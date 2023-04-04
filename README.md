# MoleculeDiffusionTransformer

## Transformer

#### Model that takes input in the form (batch, num_tokens, length); MSE loss

In this case, the input and output dimension is the same.
```
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
images = MolTrans.generate(        sequences=sequences,#conditioning
        tokens_to_generate=128, #can also generate less....
        cond_scale = 1., #temperature=3,  
     )  
print (images.shape) #(b, number_tokens, tokens_to_generate])
```

#### Model that takes input in the form of a sequence (batch, length); Cross Entropy loss

