from math import floor, log, pi
from typing import Any, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange
from torch import Tensor, einsum
from torch.utils.data import DataLoader,Dataset

from .utils import closest_power_2, default, exists, groupby
from .transformer import PositionalEncoding1D
from .diffusion import LinearSchedule, UniformDistribution, VSampler, XDiffusion, XDiffusion_x,KDiffusion_mod,LogNormalDistribution,ADPM2Sampler,KarrasSchedule
from .modules import STFT, SinusoidalEmbedding, XUNet1d, rand_bool, Encoder1d
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing import text, sequence
from tensorflow.keras.preprocessing.text import Tokenizer

from sklearn.metrics import r2_score
import time
#from tqdm import tqdm, trange
from tqdm.notebook import trange, tqdm
import seaborn as sns

def pad_sequence_lastchannel (output_xyz, max_length_l, device):         #pad
    output=torch.zeros((output_xyz.shape[0] , output_xyz.shape[1], max_length_l )).to(device)
    output[:,:,:output_xyz.shape[-1]]=output_xyz  
    return output.to(device)

class QMDiffusionForward(nn.Module):

    def __init__(self,  
                 max_length=1024,
                 channels=128,
                 pred_dim=1,
                 unet=None,
                 context_embedding_max_length=32,
                 unet_type='cfg', #"base"
                 pos_emb_fourier=True,
                 pos_emb_fourier_add=False,
                 text_embed_dim = 1024,
                 embed_dim_position=64,
                ):
        super(QMDiffusionForward, self).__init__()
        
        self.unet_type=unet_type    
        
        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last
        
        self.GELUact= nn.GELU()
        
        self.pos_emb_fourier=pos_emb_fourier
        self.pos_emb_fourier_add=pos_emb_fourier_add
        
        if self.pos_emb_fourier:
            if self.pos_emb_fourier_add==False:
                text_embed_dim=text_embed_dim+embed_dim_position
                
            self.p_enc_1d = PositionalEncoding1D(embed_dim_position)        
        
        self.max_length= max_length
        self.pred_dim=pred_dim
         
        
        if self.unet_type=='cfg':
            if exists (unet):
                self.unet=unet
            else:
                self.unet = XUNet1d( type=unet_type,
                    in_channels=pred_dim,

                    channels=channels,
                    patch_size=4,

                    multipliers=[1, 2, 4,   ],
                    factors    =[4, 4,   ],
                    num_blocks= [3, 3,   ],
                    attentions= [2, 2,   ],
                    attention_heads=8,
                    attention_features=64,
                    attention_multiplier=2,
                    attention_use_rel_pos=False,

                    context_embedding_features=text_embed_dim ,
                    context_embedding_max_length= context_embedding_max_length ,
                )

            # Either use KDiffusion
            self.diffusion = XDiffusion_x(type='k',
                net=self.unet,
                sigma_distribution=LogNormalDistribution(mean = -1.2, std = 1.2),
                sigma_data=0.1,
                dynamic_threshold=0.0,
            )

        if self.unet_type=='base':
            if exists (unet):
                self.unet=unet
            else:
                self.unet = XUNet1d( type=unet_type,
                    in_channels=pred_dim,

                    channels=channels,
                         patch_size=8,
                        multipliers=[1, 2, 4,   ],
                        factors    =[4, 4,   ],
                        num_blocks= [2, 2,   ],
                        attentions= [1, 1,   ],
                        attention_heads=8,
                        attention_features=64,
                        attention_multiplier=2,
                        attention_use_rel_pos=False,

                )

            # Either use KDiffusion
            self.diffusion = XDiffusion_x(type='k',
                net=self.unet,
                sigma_distribution=LogNormalDistribution(mean = -1.2, std = 1.2),
                sigma_data=0.1,
                dynamic_threshold=0.0,
            )
          
        
    def forward(self, sequences, output ): #sequences=conditioning, output=prediction 
         
        ########################## conditioning ####################################
        x= sequences.float().unsqueeze(2)
        
        x= self.fc1(x)
        x= self.GELUact(x) 
        
        
        if self.pos_emb_fourier:
            pos_fourier_xy=self.p_enc_1d(x) 
 
            if self.pos_emb_fourier_add:
                x=x+pos_fourier_xy
            
            else:
                x= torch.cat( (x,   pos_fourier_xy), 2)
        ########################## END conditioning ####################################
           
        if self.unet_type=='cfg':
            loss = self.diffusion(output,embedding=x)
        if self.unet_type=='base':
            loss = self.diffusion(output )
        
        return loss
    
    
    def sample (self, sequences,device,cond_scale=1.,timesteps=100,clamp=False,):
    
        ########################## conditioning ####################################
        x=sequences.float().unsqueeze(2)
        x= self.fc1(x)
        x= self.GELUact(x) 
        
       
        if self.pos_emb_fourier:
            
            pos_fourier_xy=self.p_enc_1d(x) 
            
             
            if self.pos_emb_fourier_add:
                x=x+pos_fourier_xy
            
            else:
                x= torch.cat( (x,   pos_fourier_xy), 2)
        ########################## END conditioning ####################################
            
        noise = torch.randn(x.shape[0], self.pred_dim,  self.max_length)  .to(device)
        
        if self.unet_type=='cfg':
        
            output = self.diffusion.sample(num_steps=timesteps, # Suggested range 2-100, higher better quality but takes longer
                sampler=ADPM2Sampler(rho=1),                                   
               sigma_schedule=KarrasSchedule(sigma_min=0.001, sigma_max=9.0, rho=3.),clamp=clamp,
                                           noise = noise,embedding=x, embedding_scale=cond_scale)
            
        if self.unet_type=='base':
            
            output = self.diffusion.sample(num_steps=timesteps, # Suggested range 2-100, higher better quality but takes longer
                sampler=ADPM2Sampler(rho=1),
                
                sigma_schedule=KarrasSchedule(sigma_min=0.001, sigma_max=9.0, rho=3.),clamp=clamp,
                                           noise = noise, )
        return output
         
def train_loop_forward (model,
                train_loader,test_loader,
                optimizer=None,
                print_every=10,
                epochs= 300,
                start_ep=0,
                start_step=0,
                save_loss_images=False,
                print_loss=10,
                cond_scales=[1.],
                num_samples=2,
                timesteps=100,
                clamp=False,
                save_model=False,
                show_jointplot=False,
                max_length = 32,
                prefix='./',
                device='cpu',
                loss_list=[],
               ):
    
    steps=start_step
    start = time.time()
    
    loss_total=0
    for e in range(1, epochs+1):
            start = time.time()

            torch.cuda.empty_cache()
           
            train_epoch_loss = 0
            model.train()
            
            for item  in tqdm (train_loader): #X=smiles, y=conditioning 

                X_train_batch= item[0].unsqueeze(1).to(device) #prediction SMILES
                y_train_batch=item[1].to(device) #conditiiong are properties
                  #SIZES: torch.Size([1024, 1, 64]) torch.Size([1024, 12])
                
                X_train_batch= item[1].unsqueeze(1).to(device) #prediction is now properties
                y_train_batch=item[0].squeeze().to(device) #SMILES is now conditioning 
                 
                #Pad prediction: was 22 must now be longer
                X_train_batch=  pad_sequence_lastchannel (X_train_batch, max_length, device)          #pad
                
                optimizer.zero_grad()
                
                loss=model ( y_train_batch , X_train_batch )   #sequences=conditioning, output=prediction  SMILES
                loss.backward( ) #sequences, output ): #sequences=conditioning, output=prediction 
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                optimizer.step()

                loss_total=loss_total+loss.item()

                if steps>0:
                    if steps % print_loss == 0:
                        norm_loss=loss_total/print_loss
                        print (f"\nTOTAL LOSS at epoch={e}, step={steps}: {norm_loss}")

                        loss_list.append (norm_loss)
                        loss_total=0

                        plt.plot (loss_list, label='Loss')
                        plt.legend()


                        if save_loss_images:
                            outname = prefix+ f"loss_{e}_{steps}.jpg"
                            plt.savefig(outname, dpi=200)
                        plt.show()
                        
                        
                        R2=sample_loop_forward (model,device ,
                                test_loader,
                                cond_scales=cond_scales, #list of cond scales - each sampled...
                                num_samples=num_samples, #how many samples produced every time tested.....
                                timesteps=timesteps,clamp=clamp,show_jointplot=show_jointplot, )
                        
                        print (f"\n\n-------------------\nTime passed for {print_loss} at {steps} = {(time.time()-start)/60} mins\n-------------------")
                        R2_list.append (R2)
                        plt.plot (R2_list, label = 'R2')
                        plt.legend()
                        plt.show ()
                        
                        start = time.time()
                      
                        if save_model:
                           
                            fname=f"{prefix}statedict_save-model-epoch_{e}.pt"
                            torch.save(model.state_dict(), fname)
                            print (f"Model saved: ", fname)
                steps=steps+1

def sample_loop_forward (model,device,
                 train_loader,
                 tokenizer_X=None,
                 cond_scales=[7.5], #list of cond scales - each sampled...
                 num_samples=2, #how many samples produced every time tested.....
                 num_batches=1,
                 timesteps=100,
                 flag=0, 
                 clamp=False,
                 show_jointplot=False,
                 draw_molecules=False,
                 draw_all=False,mols_per_row=8,
                 max_length = 32,
                 X_norm_factor= 1.,        
                 context_embedding_max_length = 12,   
                 prefix='./',
               ):
    steps=0
    e=flag
    
    for item  in train_loader:
            
        X_train_batch= item[1] #smiles
        y_train_batch=item[0].to(device)#properties

        GT=X_train_batch.squeeze().cpu().numpy()  #smiles

        num_samples = min (num_samples,y_train_batch.shape[0] )
        
        for iisample in range (len (cond_scales)):
            result=model.sample ( y_train_batch,device,
                                     cond_scale=cond_scales[iisample],
                                     timesteps=timesteps,clamp=clamp,
                                      )

            result=result.squeeze().cpu().numpy()
           
            if show_jointplot:
                sns.jointplot(y=result[:num_samples,:context_embedding_max_length].flatten(), x=GT[:num_samples,:context_embedding_max_length].flatten (), kind ='kde')
                plt.show()
                sns.jointplot(y=result[:num_samples,:context_embedding_max_length].flatten(), x=GT[:num_samples,:context_embedding_max_length].flatten (), kind ='scatter')
                plt.show()
            else:
                plt.plot (result[:num_samples,:context_embedding_max_length].flatten(), GT[:num_samples,:context_embedding_max_length].flatten () , 'r.')
                plt.show ()
            
            R2=r2_score(result[:num_samples,:context_embedding_max_length].flatten(),  GT[:num_samples,:context_embedding_max_length].flatten ())
            print ("OVERALL R2: ", R2)
            
            print (f"sample result {result.shape} GT shape {GT.shape}, conditioning: {y_train_batch.shape}")
            
            GT_smiles=y_train_batch.cpu().numpy() 
            GT_untokenized=reverse_tokenize  (tokenizer_X, GT_smiles, X_norm_factor=X_norm_factor)
            print ("GT as SMILES:     ",GT_untokenized[:num_samples])
            
            if draw_molecules:
                for i in range (num_samples):
                    draw_and_save (smi = GT_untokenized [i],
                                   fname=f'{prefix}/sample_{flag}_{i}.png', add_Hs=False)
                    
            if draw_all:
                draw_and_save_set (smiles =  GT_untokenized [:num_samples],
                 fname=f'{prefix}/sample_all_{flag}.png',mols_per_row=mols_per_row,
                  figsize=1,
                )
                    
        steps=steps+1
        if steps>num_batches-1:
            return R2
    return R2


#################

def predict_properties_from_SMILES (model,device, SMILES, scaler,
                
                cond_scales=[7.5], #list of cond scales - each sampled...
                
                timesteps=100,
                 flag=0, 
                 clamp=False,
                 X_norm_factor=1.,
                 draw_molecules=False,
                         draw_all=False,mols_per_row=8,
                                    tokenizer_X=None,max_length = 64,context_embedding_max_length=12,
                                    verbose=False,
               ):
    steps=0
    
    if verbose:
        print (f"Number of SMILES strings: {len (SMILES)}")
    
    data_tokenized = tokenizer_X.texts_to_sequences(SMILES)
    data_tokenized = sequence.pad_sequences(data_tokenized,  maxlen=max_length, padding='post', truncating='post')  
    
    data_tokenized = data_tokenized/X_norm_factor
    data_tokenized = torch.Tensor (data_tokenized).to(device)
    
    if verbose:
        print ("##########################################")
        
    for iisample in range (len (cond_scales)):
        result=model.sample ( data_tokenized,device,
                                 cond_scale=cond_scales[iisample],
                                 timesteps=timesteps,clamp=clamp,
                                  )
        
        result=result.squeeze().cpu().numpy()
        result=result[:,:context_embedding_max_length]
    
    if verbose:
        for i in range (len (SMILES)):
            print (f"For {SMILES[i]}, result={result[i]}")
        
    result_unscaled=scaler.inverse_transform(result)
    
    if verbose:
        for i in range (len (SMILES)):
            print (f"For {SMILES[i]}, unscaled result={result_unscaled[i]}")
        print ("##########################################")    
    
    return result,result_unscaled
            
##################################################################   
# Generative inverse models ######################################
##################################################################

#Diffusion
class QMDiffusion(nn.Module):

    def __init__(self,  
               
                 max_length=1024,
                 channels=128,
                 pred_dim=1,
                
                 context_embedding_max_length=32,
                 unet_type='cfg', #"base"
                 pos_emb_fourier=True,
                 pos_emb_fourier_add=False,
                 text_embed_dim = 1024,
                 
                 embed_dim_position=64,
               
                ):
        super(QMDiffusion, self).__init__()
        
        self.unet_type=unet_type    
        print ("Using unet type: ", self.unet_type)
        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last
        
        self.GELUact= nn.GELU()
        
        self.pos_emb_fourier=pos_emb_fourier
        self.pos_emb_fourier_add=pos_emb_fourier_add
        
        if self.pos_emb_fourier:
            if self.pos_emb_fourier_add==False:
                text_embed_dim=text_embed_dim+embed_dim_position
                
            self.p_enc_1d = PositionalEncoding1D(embed_dim_position)        
        
        self.max_length= max_length
        self.pred_dim=pred_dim
         
        if self.unet_type=='cfg':
            self.unet = XUNet1d( type=unet_type,
                in_channels=pred_dim,

                pre_transformer=2,#number of self attention pre-transformer layers before unet begins
                                
                channels=channels,
                patch_size=1,

                    multipliers=[1, 2, 4,   ],
                    factors    =[4, 4,   ],
                    num_blocks= [3, 3,   ],
                    attentions= [4, 4,   ],
                    attention_heads=8,
                    attention_features=64,
                    attention_multiplier=2,
                    attention_use_rel_pos=False,

                context_embedding_features=text_embed_dim ,
                context_embedding_max_length= context_embedding_max_length ,
            )

            # Either use KDiffusion
            self.diffusion = XDiffusion_x(type='k',
                net=self.unet,
                sigma_distribution=LogNormalDistribution(mean = -1.2, std = 1.2),
                sigma_data=0.1,
                dynamic_threshold=0.0,
            )

        if self.unet_type=='base':
            self.unet = XUNet1d( type=unet_type,
                in_channels=pred_dim,
                                
               # pre_transformer=2,#number of self attention pre-transformer layers before unet begins

                channels=channels,
                     patch_size=8,
                    multipliers=[1, 2, 4,   ],
                    factors    =[4, 4,   ],
                    num_blocks= [2, 2,   ],
                    attentions= [1, 1,   ],
                    attention_heads=8,
                    attention_features=64,
                    attention_multiplier=2,
                    attention_use_rel_pos=False,
                
                )

            self.diffusion = XDiffusion_x(type='k',
                net=self.unet,
                sigma_distribution=LogNormalDistribution(mean = -1.2, std = 1.2),
                sigma_data=0.1,
                dynamic_threshold=0.0,
            )
          
    def forward(self, sequences, output ): #sequences=conditioning, output=prediction 
         
        x= sequences.float().unsqueeze(2)
        
        x= self.fc1(x)
        x= self.GELUact(x) 
        
        if self.pos_emb_fourier:
            pos_fourier_xy=self.p_enc_1d(x) 
 
            if self.pos_emb_fourier_add:
                x=x+pos_fourier_xy
            else:
                x= torch.cat( (x,   pos_fourier_xy), 2)
        ########################## END conditioning ####################################
           
         
        #print (output.shape, x.shape)
        if self.unet_type=='cfg':
            loss = self.diffusion(output,embedding=x)
        if self.unet_type=='base':
            loss = self.diffusion(output )
        
        return loss
    
    def sample (self, sequences,device,cond_scale=7.5,timesteps=100,clamp=False,):
    
        ########################## conditioning ####################################
         
        x=sequences.float().unsqueeze(2)
        x= self.fc1(x)
        x= self.GELUact(x) 
        
        if self.pos_emb_fourier:
            
            pos_fourier_xy=self.p_enc_1d(x) 
            
             
            if self.pos_emb_fourier_add:
                x=x+pos_fourier_xy
            
            else:
                x= torch.cat( (x,   pos_fourier_xy), 2)
        ########################## END conditioning ####################################
            
        noise = torch.randn(x.shape[0], self.pred_dim,  self.max_length)  .to(device)
        
        if self.unet_type=='cfg':
        
            output = self.diffusion.sample(num_steps=timesteps, # Suggested range 2-100, higher better quality but takes longer
                sampler=ADPM2Sampler(rho=1),                                   
               sigma_schedule=KarrasSchedule(sigma_min=0.001, sigma_max=9.0, rho=3.),clamp=clamp,
                                           noise = noise,embedding=x, embedding_scale=cond_scale)
            
        if self.unet_type=='base':
            
            output = self.diffusion.sample(num_steps=timesteps, # Suggested range 2-100, higher better quality but takes longer
                sampler=ADPM2Sampler(rho=1),
                
                sigma_schedule=KarrasSchedule(sigma_min=0.001, sigma_max=9.0, rho=3.),clamp=clamp,
                                           noise = noise, )
             
        return output 
    
##############################################################
#Tools and helpers
##############################################################
    
import rdkit
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from rdkit.Chem.Draw import IPythonConsole
from rdkit.Chem import rdFMCS
from rdkit.Chem.Draw import rdDepictor
from PIL import Image

rdDepictor.SetPreferCoordGen(True)
IPythonConsole.drawOptions.minFontSize=20

def view_difference(mol1, mol2):
    mcs = rdFMCS.FindMCS([mol1,mol2])
    mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
    match1 = mol1.GetSubstructMatch(mcs_mol)
    target_atm1 = []
    for atom in mol1.GetAtoms():
        if atom.GetIdx() not in match1:
            target_atm1.append(atom.GetIdx())
    match2 = mol2.GetSubstructMatch(mcs_mol)
    target_atm2 = []
    for atom in mol2.GetAtoms():
        if atom.GetIdx() not in match2:
            target_atm2.append(atom.GetIdx())
    return Draw.MolsToGridImage([mol1, mol2],highlightAtomLists=[target_atm1, target_atm2])

def draw_and_save (smi = 'CC(=CC(C)(C)CCCCCCCC(=O)O)C1CCC2C(=CC=C3CC(O)CC(O)C3)CCCC21C', 
                   fname='out.png',
                  add_Hs=False, 
                   plot_title=None,
                  figsize=1,
                  GTsmile =None):
    
    valid=False
    if GTsmile==None:
    
        if plot_title==None:
            plot_title=fname
        molecule = Chem.MolFromSmiles(smi)
        
        if molecule != None:
            valid=True
            if add_Hs:
                molecule = Chem.AddHs(molecule)

            fig = Draw.MolToMPL(molecule)
            fig.set_size_inches(figsize,figsize)
            plt.title(plot_title)
            plt.grid(False)
            plt.axis('off')

            fig.savefig(fname, bbox_inches='tight')
            plt.show ()
    else:
        molecule1 = Chem.MolFromSmiles(smi)
        molecule2 = Chem.MolFromSmiles(GTsmile)
        
        if molecule1 != None and  molecule2!=None:
            valid=True
            img = Draw.MolsToGridImage((molecule1,molecule2), subImgSize=(600,600), returnPNG=True)
           
            png = img.data
           
            with open(fname,'wb+') as outf:
                outf.write(png) 

            im=Image.open(fname)
            plt.imshow (im)
            plt.title(plot_title)
            plt.grid(False)
            plt.axis('off')
            plt.show()
        
    return valid

def draw_and_save_set (smiles = ['CC(=CC(C)(C)CCCCCCCC(=O)O)C1CCC2C(=CC=C3CC(O)CC(O)C3)CCCC21C',
                                 'CC(=CC(C)(C)CCCCCC(O)CC(O)C3)CCCC21C'],
                   fname='out.png',
                    plot_title=None,
                  figsize=1,
                       mols_per_row=8,
                ):
    
    mols = [Chem.MolFromSmiles(smi) for smi in smiles]
    
    img = Draw.MolsToGridImage(mols, subImgSize=(600,600), returnPNG=True, molsPerRow=mols_per_row,
                              maxMols=mols_per_row*len (mols))
  
    png = img.data
  
    with open(fname,'wb+') as outf:
        outf.write(png) 

    im=Image.open(fname)
    plt.imshow (im)
    plt.title(plot_title)
    plt.grid(False)
    plt.axis('off')
    plt.show()
    
def pad_sequence_end (output_xyz, max_length_l):         #pad
    output=torch.zeros((output_xyz.shape[0] , max_length_l,  output_xyz.shape[2])).to(device)
    output[:,:output_xyz.shape[-2],:]=output_xyz  
    return output.to(device)    
    
### Dataset loaders, processing, etc.

class MoleculeDataset(Dataset):
    
    def __init__(self, X_data, y_data):
        self.X_data = X_data
        self.y_data = y_data
       
    def __getitem__(self, index):
        
        return self.X_data[index], self.y_data[index]
        
    def __len__ (self):
        return len(self.X_data)

    
def get_data_loaders (X_scaled,  y_data_scaled,  split=0.1, batch_size_=16):

    X_train, X_test, y_train, y_test,     = train_test_split(torch.Tensor (X_scaled),    torch.Tensor (y_data_scaled) ,
                                                             test_size=split,random_state=235)
    print (f"Shapes= {X_scaled.shape}, {y_data_scaled.shape}")
    
    print ("Shapes of training/test for X and y:")
    print(X_train.shape)
    print(y_train.shape)
    print(X_test.shape)
    print(y_test.shape)
    train_dataset = MoleculeDataset(X_train, y_train,  )

    test_dataset = MoleculeDataset(X_test,y_test, )

    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size_, shuffle=True)
    train_loader_noshuffle = DataLoader(dataset=train_dataset, batch_size=batch_size_, shuffle=False)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size_)

    return train_loader,train_loader_noshuffle, test_loader    

def is_novel (ALL_SMILES, smi):
    if smi in ALL_SMILES:
        return False
    else:
        return True
    
def reverse_tokenize (tokenizer_X, X_data, X_norm_factor=1):
    
    X_data_tokenized_reversed=tokenizer_X.sequences_to_texts ( (X_data*X_norm_factor).astype(int) )
    k = []
    
    for i in X_data_tokenized_reversed:
        i=str(i).replace(' ','')

        k.append(i)
    return k

######################## Generative model training loop, etc.

def train_loop_generative (model,
                train_loader,test_loader,
                optimizer=None,
                print_every=10,
                epochs= 300,
                start_ep=0,
                start_step=0,
                save_loss_images=False,
                print_loss=10,
                cond_scales=[1.],
                num_samples=2,
                timesteps=100,
                clamp=False,
                save_model=False,
                show_jointplot=False,
                prefix='./',
                ALL_SMILES=[''],
                model_forward=None,
                scaler=None,
                X_norm_factor=1.,
                device='cpu',
                loss_list=[],
               ):
    

    steps=start_step
    start = time.time()

    loss_total=0
    for e in range(1, epochs+1):
            start = time.time()

            torch.cuda.empty_cache()
           
            train_epoch_loss = 0
            model.train()
            
            for item  in tqdm (train_loader): #X=smiles, y=conditioning 

                X_train_batch= item[0].to(device) #prediction SMILES
                y_train_batch= item[1].to(device) #conditiiong
                
                optimizer.zero_grad()
                
                X_train_batch=torch.permute (X_train_batch , (0,2,1))
                 
                loss=model ( y_train_batch , X_train_batch )   #sequences=conditioning, output=prediction  SMILES
                loss.backward( ) #sequences, output ): #sequences=conditioning, output=prediction 
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)

                optimizer.step()

                loss_total=loss_total+loss.item()

                if steps>0:
                    if steps % print_loss == 0:
                        norm_loss=loss_total/print_loss
                        print (f"\nTOTAL LOSS at epoch={e}, step={steps}: {norm_loss}")

                        loss_list.append (norm_loss)
                        loss_total=0

                        plt.plot (loss_list, label='Loss')
                        plt.legend()


                        if save_loss_images:
                            outname = prefix+ f"loss_{e}_{steps}.jpg"
                            plt.savefig(outname, dpi=200)
                        plt.show()
                        
                        
                        sample_loop_generative (model,device ,
                                test_loader,
                                cond_scales=cond_scales, #list of cond scales - each sampled...
                                num_samples=num_samples, #how many samples produced every time tested.....
                                timesteps=timesteps,clamp=clamp,show_jointplot=show_jointplot,
                                model_forward=model_forward,scaler=scaler,X_norm_factor=X_norm_factor,
                                               )
                        
                        print (f"\n\n-------------------\nTime passed for {print_loss} epochs at {steps} = {(time.time()-start)/60} mins\n-------------------")
                        start = time.time()
                        #print (f"\n\n-------------------\nTime for epoch {e}={(time.time()-start)/60} mins\n-------------------")
                        if save_model:
                            
                            fname=f"{prefix}statedict_save-model-epoch_{e+start_ep}.pt"
                            torch.save(model.state_dict(), fname)
                            print (f"Model saved: ", fname)
                steps=steps+1


def sample_loop_generative (model,device,
                 train_loader,model_forward=None,
                 cond_scales=[7.5], #list of cond scales - each sampled...
                 num_samples=2, #how many samples produced every time tested.....
                 num_batches=1,
                 timesteps=100,
                 flag=0, 
                 clamp=False,
                 show_jointplot=False,
                 draw_molecules=False,
                 prefix='./',
                 tokenizer_X=None,
                 ALL_SMILES=[''],
                 scaler=None,
                 X_norm_factor=1,
               ):
    steps=0
    e=flag
    novel_count=0
 
    for item  in train_loader:
            
        X_train_batch= item[0]
        y_train_batch=item[1].to(device)#conditinojng

        GT=torch.argmax(X_train_batch, dim=2).squeeze().cpu().numpy() 
        

        num_samples = min (num_samples,y_train_batch.shape[0] )
        
        for iisample in range (len (cond_scales)):
            result=model.sample ( y_train_batch,device,
                                     cond_scale=cond_scales[iisample],
                                     timesteps=timesteps,clamp=clamp,
                                      )
        
            result=torch.permute (result , (0,2,1))
            result=torch.argmax(result, dim=2)
            
            result=result.cpu().numpy()
  
            if show_jointplot:
                sns.jointplot(y=result[:num_samples].flatten(), x=GT[:num_samples].flatten (), kind ='kde')
                plt.show()
            else:
                plt.plot (result[:num_samples].flatten(), GT[:num_samples].flatten () , 'r.')
                plt.show ()
                
            print (f"sample result {result.shape} GT shape {GT.shape}, conditioning: {y_train_batch.shape}")
            
            result_untokenized=reverse_tokenize  (tokenizer_X, result, X_norm_factor=1)
            print ("Result as SMILES: ", result_untokenized[:num_samples])
            GT_untokenized=reverse_tokenize  (tokenizer_X, GT, X_norm_factor=1)
            print ("GT as SMILES:     ",GT_untokenized[:num_samples])
            
            if draw_molecules:
                l_res=[]
                l_GT=[]
                for i in range (num_samples):
                    
                    res= result_untokenized[i] 
                    GT_s=  GT_untokenized[i] 
    
                    novel_flag=is_novel (ALL_SMILES, res)
    
                    print ("SMILES result=", res, "GT: ", GT_s, " is novel: ", novel_flag)
                    if novel_flag:
                        novel_count=novel_count+1
                    
                    draw_and_save (smi =res, 
                                   GTsmile = GT_s,
                                   fname=f'{prefix}/sample_{flag}_{i}.png', add_Hs=False)
                    
                    prop,_=predict_properties_from_SMILES (model_forward,device, SMILES=[GT_s,res],scaler=scaler,
                                                           tokenizer_X=tokenizer_X,
                                                           X_norm_factor=X_norm_factor,
                 
                        cond_scales=[1.], #list of cond scales - each sampled...

                        timesteps=100,
                         flag=0, 
                         clamp=False,

                         draw_molecules=False,
                                 draw_all=False,mols_per_row=8,
                       )
                    
                    
                    
                    print ('R2 score= ', r2_score (prop[0,:],prop[1,:]))
                    plt.plot (prop[0,:],prop[1,:] , 'r.')
                    plt.plot ([-1,1], [-1,1], 'k')
                    plt.axis('square')
                    plt.xlabel ('GT')
                    plt.ylabel ('Prediction')
                    plt.show ()      
                    
                    l_res.append (prop[1,:])
                    l_GT.append (prop[0,:])
                    
                l_res=torch.Tensor (l_res).flatten().numpy()
                l_GT=torch.Tensor (l_GT).flatten().numpy()
                plt.plot (l_GT, l_res, 'r.')
                plt.plot ([-1,1], [-1,1], 'k')
                plt.axis('square')
                plt.xlabel ('GT')
                plt.ylabel ('Prediction')
                plt.show ()   
                print ('R2 score_overall= ', r2_score (l_res, l_GT) )  
                
                print ("Fraction of novel structures: ", novel_count/num_samples, f"{novel_count} out of {num_samples}")
                
        steps=steps+1
        if steps>num_batches-1:
            break
