# MoleculeDiffusionTransformer

## Transformer

Model that takes input in the form (batch, num_tokens, length). I this case, the input and output dimension is the same.
```
MolTrans = MoleculeTransformer(
        dim=128,
        depth=3,
        logits_dim=32, #number of tokens 
        dim_head = 16,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
     
        text_embed_dim = 32,
        cond_drop_prob = 0.25,
        max_text_len = 128,
        pos_fourier_graph_dim= 32, #entire graph fouruer ebd
).cuda()

sequences= torch.randn(4, 12 ).cuda()
 
output=torch.randint (0,max_length, (4, 32 , 128)).cuda().float() #batch, number of tokens, length 
 
loss=MolTrans(
        sequences=sequences,#conditioning
        output=output,
        text_mask = None,
     
        
        return_loss = True,
   
    
     
)
loss.backward()
loss
images = MolTrans.generate(        sequences=sequences,#conditioning
        tokens_to_generate=128, #can also generate less....
        cond_scale = 1., #temperature=3,  
     )  
print (images.shape) #(b, number_tokens, tokens_to_generate])
```
