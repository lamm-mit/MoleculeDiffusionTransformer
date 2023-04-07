# MoleculeDiffusionTransformer

![image](https://user-images.githubusercontent.com/101393859/230594718-d818b4a4-6af9-4df3-a7a9-ae2d86918c16.png)

## Installation

```
conda create -n MoleculeDiffusionTransformer python=3.8
conda activate MoleculeDiffusionTransformer
```
Clone repository
```
git clone https://github.com/lamm-mit/MoleculeDiffusionTransformer/
cd MoleculeDiffusionTransformer
```
To install MoleculeDiffusionTransformer:
```
pip install -e .
```
Start Jupyter Lab (or Jupyter Notebook):
```
jupyter-lab --no-browser
```

## Datasets

The QM9 dataset is used for training. Download via and place in the home folder:

```
wget https://www.dropbox.com/s/gajj3euub7k9p9j/qm9_.csv?dl=0 -O qm9_.csv
df=pd.read_csv("qm9_.csv")
df.describe()
```

## Model overview

1. Forward diffusion model (predicts molecular properties from SMILES input)
2. Generative inverse diffusion model (predicts molecular designs via SMILES codes from molecular properties input, solving the inverse problem)
3. Generative inverse transformer model (predicts molecular designs via SMILES codes from molecular properties input, solving the inverse problem)

### Pretrained weights

- Weights for model 1: https://www.dropbox.com/s/wft4uhcj8287ojt/statedict_save-model-epoch_78.pt?dl=0 (place in 'diffusion_forward' folder)

- Weights for model 2: https://www.dropbox.com/s/xzb2bb4eo1m859p/statedict_save-model-epoch_4851.pt?dl=0 (place in 'QM_generative_diffusion_inverse' folder)

- Weights for model 3: https://www.dropbox.com/s/fqu6mogj4yw2rcc/statedict_save-model-epoch_2861.pt?dl=0 (place in 'QM_generative_transformer_inverse' folder)

Models 2 and 3 solve the same task, albeit with distinct neural network architectures and strategies. 

#### Download all weights and place in proper folders:

```
wget https://www.dropbox.com/s/wft4uhcj8287ojt/statedict_save-model-epoch_78.pt?dl=0 -O ./diffusion_forward/statedict_save-model-epoch_78.pt
wget https://www.dropbox.com/s/xzb2bb4eo1m859p/statedict_save-model-epoch_4851.pt?dl=0 -O ./QM_generative_diffusion_inverse/statedict_save-model-epoch_4851.pt
wget https://www.dropbox.com/s/fqu6mogj4yw2rcc/statedict_save-model-epoch_2861.pt?dl=0 -O ./QM_generative_transformer_inverse/statedict_save-model-epoch_2861.pt
```
## Sample results

Both generative models can generate novel molecular structure that meet a set of properties. The image below shows a comparison of the predicted properties with the set of required properties, along with the molecular structure. 

![image](https://user-images.githubusercontent.com/101393859/230594632-11e80aab-05ba-497e-9ed3-01d58b6c3d21.png)

## General use of the models

The models are set up in a flexible way so they can be used to generate any kind of sqeuence data from conditioning, using both the diffusion or transformer models. Below are examples of how this can be done. To implement this for systems other than, say, SMILES representations, the traing, sampling and property prediction methods need to be rewritten. 

### Forward diffusion model (predicts molecular properties from  input): Basic model setup 

```
from   MoleculeDiffusion import QMDiffusionForward,predict_properties_from_SMILES,ADPM2Sampler

pred_dim=1 #Prediction embedding dimension, 1 here since we're predicting a max_featuresx1 tensor with the properties
max_length_forward=64
 
context_embedding_max_length=y_data.shape[1]
model_forward =QMDiffusionForward( 
        max_length=max_length_forward, #length of predicted data
        pred_dim=pred_dim,
        channels=64,
        unet_type='cfg', #'base', #'cfg',
        context_embedding_max_length=max_length_forward, #length of conditioning 
        pos_emb_fourier=True,
        pos_emb_fourier_add=False,
        text_embed_dim = 64,
        embed_dim_position=64,
        ) .to(device)  
```

### Generative inverse diffusion mode: Basic model setup 

```
from MoleculeDiffusion import QMDiffusion 

device='cpu'
max_length = 64
pred_dim=16 #dimension equals number of unique tokens
context_embedding_max_length=12 #dimension equals length of conditioning, i.e. number of molecular features to be considered

model =QMDiffusion( 
        max_length=max_length,#length of predicted results, i.e. max length of the SMILES string
        pred_dim=pred_dim,
        channels=64,
        unet_type='cfg', #'base', #'cfg',
        context_embedding_max_length=context_embedding_max_length,#length of conditioning 
        pos_emb_fourier=True,
        pos_emb_fourier_add=False,
        text_embed_dim = 64,
        embed_dim_position=64,
        )  .to(device)

sequences= torch.randn(4, context_embedding_max_length ).to (device) #conditioning sequence; note, max_text_len=12, 
output=torch.randint (0,pred_dim, (4, pred_dim , max_length)).to(device).float() #batch, number of tokens, length (length is flexible)
 
loss=model(sequences=sequences, #conditioning sequence (set of floating points)
           output=output, #desired result (e.g. one-hot encoded sequence
        )
loss.backward()
loss

#Generate
generated=model.sample (sequences,
              device,
              cond_scale=1.,
              timesteps=64,
              clamp=False,
              )
 
print (generated.shape) #(b, pred_dim, max_length])
```

### Generative inverse transformer model: Basic model setup 

#### Model that takes input in the form (batch, num_tokens, length); MSE loss

In this case, the input and output dimension is the same.
```
from  MoleculeDiffusion import MoleculeTransformer 

logits_dim = 32 #number of tokens
MolTrans = MoleculeTransformer(
        dim=128,
        depth=6,
        logits_dim=logits_dim, #number of tokens, and also input/output dimension
        dim_head = 16,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        text_embed_dim = 32,
        cond_drop_prob = 0.25,
        max_text_len = 12, #max length of conditioning sequence
        pos_fourier_graph_dim= 32, #entire graph fourier embedding, will be added to logits_dim
        
).cuda()

sequences= torch.randn(4, 12 ).cuda() #conditioning sequence; note, max_text_len=12, 
output=torch.randint (0,logits_dim, (4, logits_dim , 128)).cuda().float() #batch, number of tokens, length (length is flexible)
 
loss=MolTrans(
        sequences=sequences,#conditioning sequence
        output=output,
        text_mask = None,
        return_loss = True,
)
loss.backward()
loss

#Generate
generated = MolTrans.generate (   sequences=sequences,#conditioning
                                 tokens_to_generate=128, #can also generate less....
                                 cond_scale = 1., temperature=1,  
                              )  
print (generated.shape) #(b, number_tokens, tokens_to_generate])
```

#### Model that takes input in the form of a sequence (batch, length); Cross Entropy loss (used in the paper)

```
from   MoleculeDiffusion import MoleculeTransformerSequence, count_parameters
logits_dim = 32 #number of tokens

model = MoleculeTransformerSequence(
        dim=128,
        depth=6,
        logits_dim=logits_dim, #number of tokens  
        dim_head = 16,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        text_embed_dim = 32,
        cond_drop_prob = 0.25,
        max_text_len = 12, #max length of conditioning sequence
        pos_fourier_graph_dim= 32, #entire graph fourier embedding, will be added to logits_dim
              
).cuda()

sequences= torch.randn(4, 12 ).cuda() #conditioning sequence; note, max_text_len=12, 
output=torch.randint (0,logits_dim, (4,  23)).cuda().long() #batch, length (length is flexible)
print (output.shape)
loss=model(
          sequences=sequences,#conditioning sequence
          output=output,
          text_mask = None,
          return_loss = True,
          )
loss.backward()
loss

#if no start token provided: Model will randomly select one
generated = model.generate(    sequences=sequences,#conditioning
        tokens_to_generate=32, #can also generate less....
        cond_scale = 1., #temperature=3,  
        )  
     
#Generate start token
output_start=torch.randint (0,logits_dim, (4,  1)).cuda().long() #batch, length (length is flexible)

generated = model.generate(sequences=sequences,#conditioning
                           output=output_start, #this is the sequence to start with...
                           tokens_to_generate=32, #can also generate less....
                           cond_scale = 1., temperature=1,  
                           )  
print (generated.shape) #(b, tokens_to_generate+1) 
```

## Utility functions (e.g. drawing SMILES representations) 

```
from MoleculeDiffusion import draw_and_save, draw_and_save_set
#this function draws and saves a set of SMILES codes
draw_and_save_set (smiles = ['CC(=CC(C)(C)CCCCCCCC(=O)O)C1CCC2C(=CC=C3CC(O)CC(O)C3)CCCC21C',
                             'CCCC(C)'],
                   fname='out.png',
                   plot_title=None,
                   figsize=1,
                   mols_per_row=2,
                )

#This function compates a predicted SMILES code with a ground truth one
draw_and_save (smi = 'CC=C', GTsmile = 'CNC=C', fname='out.png', add_Hs=True)
```
