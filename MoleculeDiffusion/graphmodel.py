#https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py

from torch import nn
import numpy as np
import torch

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        return emb[None, :, :orig_ch].repeat(batch_size, 1, 1)


class PositionalEncodingPermute1D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x) instead of (batchsize, x, ch)
        """
        super(PositionalEncodingPermute1D, self).__init__()
        self.penc = PositionalEncoding1D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding2D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1).unsqueeze(1)
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        return emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)


class PositionalEncodingPermute2D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(PositionalEncodingPermute2D, self).__init__()
        self.penc = PositionalEncoding2D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels


class PositionalEncoding3D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = (
            torch.cat((sin_inp_x.sin(), sin_inp_x.cos()), dim=-1)
            .unsqueeze(1)
            .unsqueeze(1)
        )
        emb_y = torch.cat((sin_inp_y.sin(), sin_inp_y.cos()), dim=-1).unsqueeze(1)
        emb_z = torch.cat((sin_inp_z.sin(), sin_inp_z.cos()), dim=-1)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        return emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)


class PositionalEncodingPermute3D(nn.Module):
    def __init__(self, channels):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(PositionalEncodingPermute3D, self).__init__()
        self.penc = PositionalEncoding3D(channels)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels


class FixEncoding(nn.Module):
    """
    :param pos_encoder: instance of PositionalEncoding1D, PositionalEncoding2D or PositionalEncoding3D
    :param shape: shape of input, excluding batch and embedding size
    Example:
    p_enc_2d = FixEncoding(PositionalEncoding2D(32), (x, y)) # for where x and y are the dimensions of your image
    inputs = torch.randn(64, 128, 128, 32) # where x and y are 128, and 64 is the batch size
    p_enc_2d(inputs)
    """

    def __init__(self, pos_encoder, shape):
        super(FixEncoding, self).__init__()
        self.shape = shape
        self.dim = len(shape)
        self.pos_encoder = pos_encoder
        self.pos_encoding = pos_encoder(
            torch.ones(1, *shape, self.pos_encoder.org_channels)
        )
        self.batch_size = 0

    def forward(self, tensor):
        if self.batch_size != tensor.shape[0]:
            self.repeated_pos_encoding = self.pos_encoding.to(tensor.device).repeat(
                tensor.shape[0], *(self.dim + 1) * [1]
            )
            self.batch_size = tensor.shape[0]
        return self.repeated_pos_encoding    
    
#################################
# MAIN MODEL
#################################

#pred_dim = 3 #x, y, z or angle, sec str. 
#pred_dim = 25 #x, y, z or angle, sec str. 
#loss_type = 0

from .modules import XUNet1d
from .diffusion import ADPM2Sampler,XDiffusion_x,KDiffusion_mod,LogNormalDistribution
from .diffusion import LinearSchedule, UniformDistribution, VSampler, XDiffusion,KarrasSchedule
from .modules import STFT, SinusoidalEmbedding, XUNet1d, rand_bool, Encoder1d
from .utils import (
    closest_power_2,
    default,
    downsample,
    exists,
    groupby,
    to_list,
    upsample,
)

def pad_sequence (output_xyz, max_length):         #pad
    output=torch.zeros((output_xyz.shape[0],  output_xyz.shape[1] , max_length))#.to(device)
    output[:,:,:output_xyz.shape[2]]=output_xyz #just positions for now....
    return output
 
class AnalogDiffusionSparse(nn.Module):
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
            
                 predict_neighbors=False,
                ):
        super(AnalogDiffusionSparse, self).__init__()
        self.predict_neighbors=predict_neighbors
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
        
        print ("Text emb dim: ", text_embed_dim)
        
        if self.unet_type=='cfg':
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

                context_embedding_features=text_embed_dim ,
                context_embedding_max_length= context_embedding_max_length ,
            )

            self.diffusion = XDiffusion_x(type='k',
                net=self.unet,
                sigma_distribution=LogNormalDistribution(mean = -1.2, std = 1.2),
                sigma_data=0.1,
                dynamic_threshold=0.0,
            )

        if self.unet_type=='base':
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

            self.diffusion = XDiffusion_x(type='k',
                net=self.unet,
                sigma_distribution=LogNormalDistribution(mean = -1.2, std = 1.2),
                sigma_data=0.1,
                dynamic_threshold=0.0,
            )
           

    def forward(self, sequences, output ): #sequences=conditioning, output=prediction 
       
        output_nodenumbers =output[:,0, :] #not used
        output_xyz =output[:,1:4, :]
        output_neighbors =output[:,4:4+max_neighbors, :]
     
        output= pad_sequence (output_xyz, self.max_length)          #pad
       
        if self.predict_neighbors:
            
            output_neighbors= pad_sequence (output_neighbors, self.max_length)
           
            output=torch.cat( (output,   output_neighbors), 1)
                
        ########################## conditioning ####################################
        x=sequences.float().unsqueeze(2)
        x= self.fc1(x)
        x=self.GELUact(x) 
        
       
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
    
    
    def sample (self, sequences,device,cond_scale=7.5,timesteps=100,clamp=False):
        
        ########################## conditioning ####################################
        x=sequences.float().unsqueeze(2)
        x= self.fc1(x)
        x=self.GELUact(x) 
        
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
    
class AnalogDiffusionFull(nn.Module):

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
               
                 predict_neighbors=True,
                ):
        super(AnalogDiffusionFull, self).__init__()
        self.predict_neighbors=predict_neighbors
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
         # UNet used to denoise our 1D (audio) data
        
        
        if self.unet_type=='cfg':
            self.unet = XUNet1d( type=unet_type,
                in_channels=pred_dim,

                channels=channels,
                patch_size=4,

                      
                    multipliers=[1, 2, 4,   ],
                    factors    =[4, 4,   ],
                    num_blocks= [3, 3,   ],
                    attentions= [1, 1,   ],
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
            #self.diffusion = XDiffusion_x(type='v',
            #    net=self.unet,
            #    sigma_distribution=UniformDistribution (),
                 
            #    )


        


        
    def forward(self, sequences, output ): #sequences=conditioning, output=prediction 
       
        #Now need to process the output so that it is in the correct format
        #here we'll just use xyz
        #x_pos = self.emb_data( x[:,:,0].long())
        #print (x.shape)
        #x_pos = torch.squeeze(x_pos, 1)
        
        
        output_nodenumbers =output[:,0, :] #not used
        output_xyz =output[:,1:4, :]
        output_neighbors =output[:,4:4+self.max_length, :]
       
        
        if self.predict_neighbors:
         
            output=torch.cat( (output_xyz,   output_neighbors), 1)
                
        ########################## conditioning ####################################
        
        #print (x.shape)
        x=sequences.float().unsqueeze(2)
        x= self.fc1(x)
        x=self.GELUact(x) 
        
       
        if self.pos_emb_fourier:
            #pos_matrix_i_=self.pos_matrix_i.repeat(x.shape[0], 1, 1).to(device=device) 
            #pos_emb_x = self.pos_emb_x( pos_matrix_i_)
            #pos_emb_x = torch.squeeze(pos_emb_x, 1)
            pos_fourier_xy=self.p_enc_1d(x) 

            #pos_emb=self.sinu_pos_emb (pos_matrix_i_)
            if self.pos_emb_fourier_add:
                x=x+pos_fourier_xy
            #print (f"x shape, pos shape ", x.shape, pos_fourier_xy.shape)
            else:
                x= torch.cat( (x,   pos_fourier_xy), 2)
        ########################## END conditioning ####################################
           
         
        #print ("output and embedding shape", output.shape, x.shape)
        if self.unet_type=='cfg':
            loss = self.diffusion(output,embedding=x)
        if self.unet_type=='base':
            loss = self.diffusion(output )
        
        return loss
    
    
    def sample (self, sequences,device,cond_scale=7.5,timesteps=100,clamp=False,):
        
    
        ########################## conditioning ####################################
        
        #print (x.shape)
        x=sequences.float().unsqueeze(2)
        x= self.fc1(x)
        x=self.GELUact(x) 
        print (x.shape)
       
        if self.pos_emb_fourier:
            #pos_matrix_i_=self.pos_matrix_i.repeat(x.shape[0], 1, 1).to(device=device) 
            #pos_emb_x = self.pos_emb_x( pos_matrix_i_)
            #pos_emb_x = torch.squeeze(pos_emb_x, 1)
            pos_fourier_xy=self.p_enc_1d(x) 
            
            print (pos_fourier_xy.shape)

            #pos_emb=self.sinu_pos_emb (pos_matrix_i_)
            if self.pos_emb_fourier_add:
                x=x+pos_fourier_xy
            #print (f"x shape, pos shape ", x.shape, pos_fourier_xy.shape)
            else:
                x= torch.cat( (x,   pos_fourier_xy), 2)
        ########################## END conditioning ####################################
            
        noise = torch.randn(x.shape[0], self.pred_dim,  self.max_length)  .to(device)
        
        if self.unet_type=='cfg':
        
            output = self.diffusion.sample(num_steps=timesteps, # Suggested range 2-100, higher better quality but takes longer
                sampler=ADPM2Sampler(rho=1),
                #sampler=KarrasSampler( s_tmin=0.05, s_tmax=50, s_noise=1.003, s_churn=80.),                                       
               sigma_schedule=KarrasSchedule(sigma_min=0.001, sigma_max=9.0, rho=3.),clamp=clamp,
                                           noise = noise,embedding=x, embedding_scale=cond_scale)
            
        if self.unet_type=='base':
            
            output = self.diffusion.sample(num_steps=timesteps, # Suggested range 2-100, higher better quality but takes longer
                sampler=ADPM2Sampler(rho=1),
                #sampler=KarrasSampler( s_tmin=0.05, s_tmax=50, s_noise=1.003, s_churn=80.),  
                sigma_schedule=KarrasSchedule(sigma_min=0.001, sigma_max=9.0, rho=3.),clamp=clamp,
                                           noise = noise, )
            #output = self.diffusion.sample(num_steps=timesteps, # Suggested range 2-100, higher better quality but takes longer
            #    sampler=VSampler(),
            #    sigma_schedule=LinearSchedule( ),
            #                              clamp=clamp,
            #                               noise = noise,)

         
        return output 