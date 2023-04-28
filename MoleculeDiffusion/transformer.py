#code based on https://github.com/lucidrains/parti-pytorch and other sources

from typing import List
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.nn import MultiheadAttention 
import torchvision.transforms as T

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
 
import numpy as np
import torch
from tqdm.notebook import trange, tqdm

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# normalization

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 2d relative positional bias

class RelPosBias2d(nn.Module):
    def __init__(self, size, heads):
        super().__init__()
        self.pos_bias = nn.Embedding((2 * size - 1) ** 2, heads)

        arange = torch.arange(size)

        pos = torch.stack(torch.meshgrid(arange, arange, indexing = 'ij'), dim = -1)
        pos = rearrange(pos, '... c -> (...) c')
        rel_pos = rearrange(pos, 'i c -> i 1 c') - rearrange(pos, 'j c -> 1 j c')

        rel_pos = rel_pos + size - 1
        h_rel, w_rel = rel_pos.unbind(dim = -1)
        pos_indices = h_rel * (2 * size - 1) + w_rel
        self.register_buffer('pos_indices', pos_indices)

    def forward(self, qk):
        i, j = qk.shape[-2:]

        bias = self.pos_bias(self.pos_indices[:i, :(j - 1)])
        bias = rearrange(bias, 'i j h -> h i j')

        bias = F.pad(bias, (j - bias.shape[-1], 0), value = 0.) # account for null key / value for classifier free guidance
        return bias

# feedforward

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden, bias = False),
        nn.GELU(),
        LayerNorm(dim_hidden),
        nn.Linear(dim_hidden, dim, bias = False)
    )

class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2
    
class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate)
    
    
class CausalDSConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_kernel_FF=3, dilation=1 ):
        super().__init__()
        self.ds_conv = nn.Conv1d(in_ch, out_ch, conv_kernel_FF, bias = False, groups = in_ch,stride=1,)
        self.conv_kernel_FF=conv_kernel_FF
        self.dilation=dilation

    def forward(self, x):
        x = rearrange(x, 'b n c -> b c n')
        #print (x.shape)  #b, c, n
        x = F.pad(x, ((self.conv_kernel_FF - 1) * self.dilation, 0))
        #print (x.shape) #b, c, n
        #print (x)
        x = self.ds_conv(x)
        return rearrange(x, 'b c n -> b n c')
    
class FeedForward_CNN(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        glu = False,
        swish = False,
        relu_squared = False,
        post_act_ln = False,
        dropout = 0.,
        no_bias = False,
        zero_init_output = False,
        conv_kernel_FF = 0, #if >0 use causal conv1d
        FF_inner_conv = 0 #if>0 use causal conv1d in FF layer (sandwiched) 
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()
        
        self.FF_inner_conv= FF_inner_conv
            
        self.project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, bias = not no_bias),
           # CausalDSConv(inner_dim, inner_dim, conv_kernel_FF=FF_inner_conv ) ,
            activation,
        ) if not glu else GLU(dim, inner_dim, activation)
        
        

        self.ff = nn.Sequential(
            #project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = not no_bias)
        )
        
        if self.FF_inner_conv>0:
            self.inner_conv_resnetblock1 = nn.Sequential(
            #nn.Linear(dim, inner_dim, bias = not no_bias),
            CausalDSConv(inner_dim, inner_dim, conv_kernel_FF=FF_inner_conv ) ,
            activation,
            CausalDSConv(inner_dim, inner_dim, conv_kernel_FF=FF_inner_conv ) ,
            ) 
        
            
        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])
            
        self.conv_kernel_FF=conv_kernel_FF
        if self.conv_kernel_FF>0: #will only work in non-causal model
            #define two resnet blocks
            self.resnetblock1 = nn.Sequential(
                CausalDSConv(dim, dim, conv_kernel_FF=conv_kernel_FF ),#, padding='same'),
                activation,
                CausalDSConv(dim, dim, conv_kernel_FF=conv_kernel_FF ),#, padding='same'),
            )
            self.resnetblock2 = nn.Sequential(
                CausalDSConv(dim_out, dim_out, conv_kernel_FF=conv_kernel_FF ),#, padding='same'),
                activation,
                CausalDSConv(dim_out, dim_out, conv_kernel_FF=conv_kernel_FF ),#, padding='same'),
            )

    def forward(self, x):
        
 
        if self.conv_kernel_FF>0:
            #x=self.resnetblock1(x.permute (0,2,1)).permute (0,2,1)+x
            x=self.resnetblock1(x )+x
            
        
        x=self.project_in (x)
        
        if self.FF_inner_conv>0:
            x=self.inner_conv_resnetblock1(x)+x

        x=self.ff(x)
        
        if self.conv_kernel_FF>0:
            #x=self.resnetblock2(x.permute (0,2,1)).permute (0,2,1)+x
            x=self.resnetblock2(x )+x
        
        return x
    
# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        norm_context = False,
        rel_pos_bias = False,
        encoded_fmap_size = None
    ):
        super().__init__()
        self.causal = causal
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)

        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, inner_dim, bias = False),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        # needed for classifier free guidance for transformers
        # by @crowsonkb, adopted by the paper

        self.null_kv = nn.Parameter(torch.randn(dim_head))

        # one-headed key / value attention, from Shazeer's multi-query paper, adopted by Alphacode and PaLM

        self.to_kv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, dim_head, bias = False)
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

        # positional bias

        self.rel_pos_bias = None

        if rel_pos_bias:
            assert exists(encoded_fmap_size)
            self.rel_pos_bias = RelPosBias2d(encoded_fmap_size, heads)

    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        batch, device = x.shape[0], x.device

        x = self.norm(x)

        q = self.to_q(x) * self.scale

        context = default(context, x)
        context = self.norm_context(context)

        kv = self.to_kv(context)

        null_kv = repeat(self.null_kv, 'd -> b 1 d', b = batch)
        kv = torch.cat((null_kv, kv), dim = 1)

        sim = einsum('b h i d, b j d -> b h i j', q, kv)

        if exists(self.rel_pos_bias):
            pos_bias = self.rel_pos_bias(sim)
            sim = sim + pos_bias

        mask_value = -torch.finfo(sim.dtype).max

        if exists(context_mask):
            context_mask = F.pad(context_mask, (1, 0), value = True)
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        out = einsum('b h i j, b j d -> b h i d', attn, kv)

        return self.to_out(out)
    
#########################################################################################################  
#https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py
#########################################################################################################
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
    
    
    #############################################
    
    # classes
def pad_sequence (output_xyz, max_length):         #pad
    
    device = output_xyz.device
    output=torch.zeros((output_xyz.shape[0],  output_xyz.shape[1] , max_length)).to(device)
    output[:,:,:output_xyz.shape[2]]=output_xyz #just positions for now....
    return output


class MoleculeTransformer (nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        logits_dim=32, 
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        text_embed_dim = None,
        cond_drop_prob = 0.25,
        max_text_len = 128,
        pos_fourier_graph_dim= 32, #entire graph fourier embedding, has to be == dim in case of embedding since we add fourier to signal
        
    ):
        super().__init__()
        use_embedding = False
      
        self.text_embed_dim =  text_embed_dim 
        self.pos_fourier_graph_dim=pos_fourier_graph_dim

        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last. This converts sequence conditioning to higher dimension
        self.GELUact= nn.GELU()
        ################
        
        self.p_enc_1d = PositionalEncoding1D(text_embed_dim)     #fourier encoding for conditioning sequence
        self.p_enc_1d_graph = PositionalEncodingPermute1D(self.pos_fourier_graph_dim)     #fourier encoding for input
        
        self.max_text_len = max_text_len
    
        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob # classifier free guidance for transformers - @crowsonkb
        
        self.start_token = nn.Parameter(torch.randn(dim))

        # projecting to logits
        self.logits_dim=logits_dim
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, causal = True, rel_pos_bias = False, 
                          dim_head = dim_head, heads = heads, dropout = dropout),
                Attention(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
            ]))

        self.final_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)
         
        self.to_dim = nn.Linear( self.logits_dim+self.pos_fourier_graph_dim, dim, bias = False)
        
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        sequences=None,#conditioning
         text_mask=None,
        *,
            tokens_to_generate=32,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
    ):
        device = next(self.parameters()).device
  
        batch = sequences.shape[0]
     
        image_seq_len=tokens_to_generate
        
        print (f"Generating {image_seq_len} tokens...")
    
        output = torch.empty((batch, self.logits_dim, 0), device = device )
        
        for _ in tqdm(range(image_seq_len)):

            sampled  = self.forward_with_cond_scale(
            
                sequences = sequences,
                text_mask = text_mask,
                output = output,

            )

            sampled=sampled[:, -1, :]
            sampled = rearrange(sampled, 'b h -> b h 1')
            output = torch.cat((output, sampled), dim = -1)
        
        return output
 

    def forward_with_cond_scale(self, *args, cond_scale = 3, **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        texts: List[str] = None,
        sequences = None,
        output=None,
        text_mask = None,
        cond_drop_prob = None,
        return_loss = False
    ):
       
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        device = next(self.parameters()).device
        ########################## conditioning #################################### 
        
        cond_x=sequences.float().unsqueeze(2)
   
        # encoding 
        cond_x= self.fc1(cond_x)
        cond_x=self.GELUact(cond_x)  
        pos_fourier_xy=self.p_enc_1d(cond_x) 
        cond_x=cond_x+pos_fourier_xy
        
        start_tokens = repeat(self.start_token, 'd -> b 1 d', b = output.shape[0])

        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           self.pos_fourier_graph_dim,
                                                           output.shape[2] ).to(device) )  
        output=torch.cat( (output, pos_fourier_graph ), 1)
        output=torch.permute(output, (0,2,1)  )

        output = self.to_dim(output)            
        
        output = torch.cat((start_tokens, output), dim = 1)
        
        if return_loss:
            output, target = output[:, :-1,:], output[:, 1:,:self.logits_dim]
            
        # enforce max text len
        if not exists(text_mask):
            text_mask = torch.ones(cond_x.shape[:2], dtype = torch.bool).to(device)
            
        cond_x, text_mask = map(lambda t: t[:, :self.max_text_len], (cond_x, text_mask))        

        # classifier free guidance conditional dropout
        batch=output.shape[0]
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        # attend

        x = self.init_norm(output)

        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context = cond_x, context_mask = text_mask) + x
            x = ff(x) + x

        x = self.final_norm(x)

        # to logits

        logits = self.to_logits(x)
         
        if not return_loss:
            return logits

        loss = F.mse_loss(
                    # logits, output
                    logits, target
                )

        return loss
    
class MoleculeTransformerSequence(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        logits_dim=32, 
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        text_embed_dim = None,
        cond_drop_prob = 0.25,
        max_text_len = 128,
       # pos_fourier_graph_dim= 32, #entire graph fourier embedding, has to be == dim in case of embedding since we add fourier to signal

    ):
        super().__init__()

        pos_fourier_graph_dim=dim
     
        self.text_embed_dim =  text_embed_dim 
        self.pos_fourier_graph_dim=pos_fourier_graph_dim

            
        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last. This converts sequence conditioning to higher dimension
        self.GELUact= nn.GELU()
        ################
        
        self.p_enc_1d = PositionalEncoding1D(text_embed_dim)     #fourier encoding for conditioning sequence
        self.p_enc_1d_graph = PositionalEncoding1D(self.pos_fourier_graph_dim)     #fourier encoding for input
                        #(batch_size, x, ch)
       
        self.max_text_len = max_text_len

        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob # classifier free guidance for transformers - @crowsonkb

        self.start_token = nn.Parameter(torch.randn(dim))

        # projecting to logits
        self.logits_dim=logits_dim
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                
                Attention(dim, causal = True, rel_pos_bias = False, 
                          dim_head = dim_head, heads = heads, dropout = dropout),
                
                
                
                Attention(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
            ]))

        self.final_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)
         
        self.token_embed = nn.Embedding(logits_dim, dim)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        sequences=None,#conditioning
         text_mask=None,
        #texts,
        *,
        tokens_to_generate=32,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
        output=None, #should provide it with at least a start token
    ):
        device = next(self.parameters()).device

        batch = sequences.shape[0]

        image_seq_len=tokens_to_generate
        
        if output==None:
            output= torch.randint (0,self.logits_dim, (batch,  1), device = device  )
            print ("Since start token not provided, generating random token.")
        
        for _ in tqdm(range(image_seq_len)):
            
            
            sampled  = self.forward_with_cond_scale(
                
                    sequences = sequences,
                    text_mask = text_mask,
                    output = output,
                     
                )
            sampled=sampled[:, -1 ]
            
            filtered_logits = top_k(sampled, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            sampled = rearrange(sampled, 'b -> b 1')
            
            output = torch.cat((output, sampled), dim = -1)

        return output
 

    def forward_with_cond_scale(self, *args, cond_scale = 3, **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        texts: List[str] = None,
        sequences = None,
        output=None,
        text_mask = None,
    
        cond_drop_prob = None,
        return_loss = False
    ):
       
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        device = next(self.parameters()).device
        ########################## conditioning #################################### 
        
        cond_x=sequences.float().unsqueeze(2)

        # encoding 
        cond_x= self.fc1(cond_x)
        cond_x=self.GELUact(cond_x)  
        pos_fourier_xy=self.p_enc_1d(cond_x) 
        cond_x=cond_x+pos_fourier_xy
         
        if return_loss:
            labels  = output [:, 1:] #labels are non embedded
        
        output = self.token_embed(output.long())
        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           output.shape[1],
                                                          self.pos_fourier_graph_dim).to(device) )  
  
        output=output+pos_fourier_graph
        
        # enforce max text len

        if not exists(text_mask):
            text_mask = torch.ones(cond_x.shape[:2], dtype = torch.bool).to(device)
            
        cond_x, text_mask = map(lambda t: t[:, :self.max_text_len], (cond_x, text_mask))        

        # classifier free guidance conditional dropout
        batch=output.shape[0]
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        # attend
        x = self.init_norm(output)
        for self_attn, cross_attn, ff in self.layers:
            
            x = self_attn(x) + x
            x = cross_attn(x, context = cond_x, context_mask = text_mask) + x
            s_causal=False
            
            x = ff(x) + x

        x = self.final_norm(x)

        # to logits
        logits = self.to_logits(x)
         
        if not return_loss:
            return logits
        
        logits=logits[:,:-1,:]
       
        loss = F.cross_entropy(
            rearrange(logits , 'b n c -> b c n'),
            labels ,
            # ignore_index = 0
            )
        
        return loss

############## same but with flexible internal dimension (in previous: embed_dim is equal to dim)
class MoleculeTransformerSequenceInternaldim(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        max_tokens=32,
        logits_dim=32, 
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        embed_dim = 16, #for input sequence
        text_embed_dim = None, #for cnditining 
        cond_drop_prob = 0.25,
        max_text_len = 128,
        one_kv_head=True,
        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.text_embed_dim =  text_embed_dim 
        self.pos_fourier_graph_dim=embed_dim

        #sequence to embedding    
        self.token_embed = nn.Embedding(max_tokens, embed_dim)
        #embedding to dim
        self.to_dim = nn.Linear( self.embed_dim, dim, bias = False)
            
        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last. This converts sequence conditioning to higher dimension
        self.GELUact= nn.GELU()
        ################
        
        self.p_enc_1d = PositionalEncoding1D(text_embed_dim)     #fourier encoding for conditioning sequence
        self.p_enc_1d_graph = PositionalEncoding1D(self.pos_fourier_graph_dim)     #fourier encoding for input
                        #(batch_size, x, ch)
       
        self.max_text_len = max_text_len

        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob # classifier free guidance for transformers - @crowsonkb

        self.start_token = nn.Parameter(torch.randn(dim))

        # projecting to logits
        self.logits_dim=logits_dim
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                
                AttentionQKV(dim, causal = True, one_kv_head=one_kv_head,
                          dim_head = dim_head, heads = heads, dropout = dropout),
                AttentionQKV(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout, one_kv_head=one_kv_head,),
                
                FeedForward(dim, mult = ff_mult, dropout = dropout)
                
            ]))

        self.final_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)
        

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        sequences=None,#conditioning
         text_mask=None,
        #texts,
        *,
        tokens_to_generate=32,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
        output=None, #should provide it with at least a start token
    ):
        device = next(self.parameters()).device

        batch = sequences.shape[0]

        image_seq_len=tokens_to_generate
        
        if output==None:
            output= torch.randint (0,self.logits_dim, (batch,  1), device = device  )
            print ("Since start token not provided, generating random token.")
        
        for _ in tqdm(range(image_seq_len)):
            
            
            sampled  = self.forward_with_cond_scale(
                
                    sequences = sequences,
                    text_mask = text_mask,
                    output = output,
                     
                )
            sampled=sampled[:, -1 ]
            
            filtered_logits = top_k(sampled, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            sampled = rearrange(sampled, 'b -> b 1')
            
            output = torch.cat((output, sampled), dim = -1)

        return output


    def forward_with_cond_scale(self, *args, cond_scale = 3, **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        texts: List[str] = None,
        sequences = None,
        output=None,
        text_mask = None,
    
        cond_drop_prob = None,
        return_loss = False
    ):
       
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        device = next(self.parameters()).device
        ########################## conditioning #################################### 
        
        cond_x=sequences.float().unsqueeze(2)

        # encoding 
        cond_x= self.fc1(cond_x)
        cond_x=self.GELUact(cond_x)  
        pos_fourier_xy=self.p_enc_1d(cond_x) 
        cond_x=cond_x+pos_fourier_xy
         
        if return_loss:
            labels  = output [:, 1:] #labels are non embedded
        
        output = self.token_embed(output.long())
        
        
        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           output.shape[1],
                                                          self.pos_fourier_graph_dim).to(device) )  
  
        output=output+pos_fourier_graph
               
        # enforce max text len

        if not exists(text_mask):
            text_mask = torch.ones(cond_x.shape[:2], dtype = torch.bool).to(device)
            
        cond_x, text_mask = map(lambda t: t[:, :self.max_text_len], (cond_x, text_mask))        

        # classifier free guidance conditional dropout
        batch=output.shape[0]
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        x = self.to_dim (output)
        
        x = self.init_norm(x)
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context = cond_x, context_mask = text_mask) + x
            x = ff(x) + x

        x = self.final_norm(x)

        # to logits
        logits = self.to_logits(x)
         
        if not return_loss:
            return logits
        
        logits=logits[:,:-1,:]
       
        loss = F.cross_entropy(
            rearrange(logits , 'b n c -> b c n'),
            labels ,
            # ignore_index = 0
            )
        
        return loss

############### transformer encoder ###################

from functools import partial, wraps
def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner



    
#######################################################################

class MoleculeTransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        logits_dim=32, 
        logits_dim_length = None , #if None entire length is used. Otherwise, max_length is projected to logits_dim_length
        max_length = None,         #see previous - OUTPUT IS (b, logits_dim, logits_dim_length)
        max_tokens = 32, #max tokens for embedding layer
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
      
        embed_dim = 16,
     
        padding_token=0,#will be used to create mask

    ):
        super().__init__()

      #  pos_fourier_graph_dim=dim
        self.padding_token=padding_token
     
        self.embed_dim=embed_dim #embedding dim of sequence
        self.max_length=max_length
        self.logits_dim_length=logits_dim_length

        self.GELUact= nn.GELU()
        ################
        
        self.p_enc_1d_graph = PositionalEncoding1D(self.embed_dim)     #fourier encoding for input
                        #(batch_size, x, ch)
       
        # projecting to logits
        self.logits_dim = logits_dim
        self.init_norm  = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                #AttentionQKV(dim, causal = False, 
                #          dim_head = dim_head, heads = heads, dropout = dropout),
                MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                #Attention(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout) 
            ]))

        self.final_norm = LayerNorm(dim)
         
        #sequence to embedding    
        self.token_embed = nn.Embedding(max_tokens, embed_dim)
        #embedding to dim
        self.to_dim = nn.Linear( self.embed_dim, dim, bias = False)
        #dim to logits
        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)
        
        if exists (self.logits_dim_length):
            assert exists (self.max_length), 'max_length and logits_dim_length must be set to nonzero value'
            self.to_logits_dim_length = nn.Linear(self.max_length, self.logits_dim_length, bias = False)

    def forward(
        self,
    
        input_sequence,
        text_mask = None,
        return_hidden = False,
        squeeze_output=False,
    
    ):
    
        device = next(self.parameters()).device
        ########################## conditioning #################################### 
       
        output = self.token_embed(input_sequence)   
        
        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           output.shape[1],
                                                           self.embed_dim).to(device) )  
  
        output=output+pos_fourier_graph
    
        x = self.to_dim (output)
    
        if exists (self.max_length):  # enforce max text len
            if not exists(text_mask):
                text_mask = torch.ones(x.shape[:2], dtype = torch.bool).to(device)
                text_mask = input_sequence.eq(self.padding_token) #use  0 will turn to True, others to False. All values with True are masked out.
            
        x, text_mask = map(lambda t: t[:, :self.max_length], (x, text_mask))     
       
        x = self.init_norm(x)
        #for self_attn, cross_attn, ff in self.layers:
        for self_attn, ff in self.layers:
            #x = self_attn(x, context_mask = text_mask) + x
            attout, _ = self_attn(x, x, x, key_padding_mask  = text_mask)
            x= attout + x
            
            #x = cross_attn(x, context = cond_x, context_mask = text_mask) + x
            x = ff(x) + x

        x = self.final_norm(x)

        # to logits
        logits = self.to_logits(x)
        
        if exists(self.logits_dim_length):
            logits=logits.permute (0,2,1)
            logits = self.to_logits_dim_length(logits)
            
         
        if return_hidden:
            
            return  x
    
        else:
            if squeeze_output:
                logits=logits.squeeze()
            
            return logits
        
        
######################### GPT with Graph, etc. #######################################

class AttentionQKV(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        norm_context = False,
        one_kv_head=True,
        use_null_kv=True,
        GNN_layers = 0,
        GNN_aggr = 'add',
        GNN_have_skip = True,
        GNN_att_threshold_min=0,#if >0 then set all values in att_matrix to 0 that are smaller
        GNN_att_threshold_max=1,#if <1 then set all values in att_matrix to 1 that are larger
        GNN_add_identity= True, #whether to add identity
        GNN_clamp_att_after_identity = True, #whether to clamp attention after identity is added
        
    ):
        super().__init__()
        self.causal = causal
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)
        self.use_null_kv=use_null_kv
        self.GNN_layers=GNN_layers
        self.GNN_att_threshold_min=GNN_att_threshold_min
        self.GNN_att_threshold_max=GNN_att_threshold_max
        self.GNN_add_identity= GNN_add_identity
        self.heads=heads
        self.GNN_clamp_att_after_identity=GNN_clamp_att_after_identity

        context_dim = default(context_dim, dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()
        
        self.one_kv_head = one_kv_head #If true: Use one KV head, multiple query heads

        # needed for classifier free guidance for transformers
        # by @crowsonkb, adopted by the paper
        
        q_dim = k_dim = dim_head * heads
        v_dim = out_dim = dim_head * heads

        # one-headed key / value attention, from Shazeer's multi-query paper, adopted by Alphacode and PaLM
        # https://arxiv.org/abs/1911.02150 - only have one head for the key / values, multi-headed queries
        self.one_kv_head = one_kv_head
        if one_kv_head:
            k_dim = dim_head
            v_dim = dim_head
            out_dim = v_dim * heads

        self.null_k = nn.Parameter(torch.randn(k_dim))
        self.null_v = nn.Parameter(torch.randn(v_dim))
        
        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, q_dim, bias = False),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )
        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, k_dim, bias = False)
        )
        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, v_dim, bias = False)
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
          
            nn.Linear(out_dim, dim, bias = False),
            LayerNorm(dim)
        )
        
         
        if self.GNN_layers>0:
            self.GNN_layers=GNN_layers
            #GNN_hidden_channels=dim
            #print ("q_dim ",q_dim) 
            self.GNN_net = GraphConvLayers(node_features_in=dim_head,#q_dim, 
                                    num_node_features_out=dim_head,#,q_dim, 
                                    hidden_channels=dim_head,#q_dim,
                                    depth=GNN_layers,
                                   # aggr = GNN_aggr,
                                    have_skip = GNN_have_skip,
                                   )



    def forward(
        self,
        x,
        context = None,
        context_mask = None,
        plot_attn= False,
        
        
    ):
        batch, device = x.shape[0], x.device

        x = self.norm(x)

        q = self.to_q(x) * self.scale

        context = default(context, x)
        context = self.norm_context(context)

        k = self.to_k(context)
        v = self.to_v(context)

        if self.use_null_kv:
            null_k = repeat(self.null_k, 'd -> b 1 d', b = batch)
            k = torch.cat((null_k, k), dim = 1)
            null_v = repeat(self.null_v, 'd -> b 1 d', b = batch)
            v = torch.cat((null_v, v), dim = 1)

        
        if not self.one_kv_head:
            k, v = map(lambda t: maybe(rearrange)(t, 'b n (h d) -> b h n d', h = self.heads), (k, v))
            
        kv_einsum_eq = 'b h j d' if not self.one_kv_head else 'b j d'

        #dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

        sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) #* scale

        mask_value = -torch.finfo(sim.dtype).max

        if exists(context_mask):
            if self.use_null_kv:
                context_mask = F.pad(context_mask, (1, 0), value = True)
            
            #original
            #context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            
            if context_mask.ndim == 2:#original ... 
                #attn_mask = rearrange(attn_mask, 'i j -> 1 1 i j')
                context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            #elif context_mask.ndim == 3:#not sure if works...
            #    #attn_mask = rearrange(attn_mask, 'h i j -> 1 h i j')
            #    context_mask = rearrange(context_mask, 'b j h -> b 1 j h')
            #context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            
            sim = sim.masked_fill(~context_mask, mask_value)
            #dots = dots.masked_fill(~input_mask, mask_value)
            
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)
            
            #causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            #dots = dots.masked_fill(causal_mask, mask_value)            

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        
        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)
        
        ######### Graph neural net ###############################
        #GNN layers are added after causal mask is applied to maintain causality
        #GNN is constructed by using attn as adjancy matrix and/or edge features, shape is d,d
        #    node features are defined by v here: shape is n,d
        #    attn can be processed with min/max threshold, identity added, etc.
        #    TO DO: implement other aggregation schemes, multi hop message passing....
        if self.GNN_layers>0:
            #print ("out shape" , out.shape)
            #print ("Att bef shape: ", attn.shape)
            # attn=  rearrange (attn,'b h n d -> b n (h d)')
            attn_comb=  rearrange (attn,'b h n d -> (b h) n d') #these are the attention matrices - batch and head grouped
            if self.GNN_add_identity:
                attn_comb = attn_comb + torch.eye(attn_comb.shape[1]).unsqueeze(0).repeat(attn_comb.shape[0],1,1).to(device)
                if self.GNN_clamp_att_after_identity:
                    attn_comb = torch.clamp(attn_comb, min=0, max=1)

                #attn_comb =attn_comb.softmax (dim=-1, dtype = torch.float32)
                
            if self.GNN_att_threshold_min>0:
                attn_comb[attn_comb<self.GNN_att_threshold_min] = 0
            if self.GNN_att_threshold_max<1:
                attn_comb[attn_comb>self.GNN_att_threshold_max] = 1
                
            if plot_attn:
                grid_disc=int (attn_comb.shape [0]**0.5)
                fig, axes = plt.subplots(nrows=grid_disc, ncols=grid_disc, figsize=(8,8))
                c_i=0
                for i in range (grid_disc):
                    for j in range (grid_disc):
                        axes[i,j].imshow (attn_comb[c_i,:].detach().numpy() )
                        #axes[i,j].colorbar()
                        c_i+=1

                # fig.colorbar (ax=fig.get_axes())
                plt.show()
            
            #v_comb=  rearrange (v,'b h n d -> b n (h d)')
            v_comb=  rearrange (v,'b h n d -> (b h) n d') #these are the node embeddings - batch and head grouped
            
            #print ("v_comb.shape",v_comb.shape, "att_comb shape shape: ", attn_comb.shape)
            #print (edge_index.shape, edge_weight.shape)

            #data=torch.randn (4, 16, dim) #b, n, c
            out_from_GNN = self.GNN_net(v_comb, attn_comb,) 
            
            
            
            #print (out_from_GNN.shape)
            #attn_comb=  rearrange (attn,'(b h) n d -> b h n d')
            out = out +  rearrange (out_from_GNN,'(b h) n d -> b h n d', h = self.heads)
            
            
            #print (out_from_GNN.shape, attn.shape)

        
        #out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)
        

        return self.to_out(out)
    
#######################################################################

class MoleculeTransformerGPT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        max_tokens=32,
        logits_dim=32, 
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        embed_dim = 16, #for input sequence
        text_embed_dim = 16,# None, #not used 
        max_text_len = 128,
        one_kv_head=True,
        concat_pos_encoding = False, #if True, then pos encoding will be added to the input; and pos_fourier_graph_dim=embed_dim
        pos_fourier_graph_dim  = None,  #only used if concat_pos_encoding == True
        use_null_kv=True, #use null_kv for CFG
        FF_conv_kernel=0, #if >0 then use conv in FF layer, at beginning and end
        FF_inner_conv_kernel=0,
        FF_glu=False,
        GNN_layers = 0, #whether or not to use GNN_layers after softmax
        GNN_att_threshold_min=0.,
        GNN_att_threshold_max=1.,
        
    ):
        super().__init__()

        # pos_fourier_graph_dim=embed_dim
        self.embed_dim = embed_dim
        self.concat_pos_encoding=concat_pos_encoding
        self.use_null_kv=use_null_kv
        #  self.text_embed_dim =  text_embed_dim 
        if concat_pos_encoding == False:
            self.pos_fourier_graph_dim=embed_dim
        else:
            self.pos_fourier_graph_dim=pos_fourier_graph_dim
            assert pos_fourier_graph_dim !=None, "pos_fourier_graph_dim has to be set if concatenating pos embedding"
            print ("Concatenate positional encoding...")

        #sequence to embedding    
        self.token_embed = nn.Embedding(max_tokens, embed_dim)
        #embedding to dim
        
        dim_in = self.embed_dim + self.concat_pos_encoding*self.pos_fourier_graph_dim
        self.to_dim = nn.Linear(dim_in, 
                                dim, bias = False)
        print ("Input dimension (signal and pos encoding): ", dim_in)
        
            
        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last. This converts sequence conditioning to higher dimension -- NOT USED, TODO: Remove
        self.GELUact= nn.GELU()
        ################
        self.p_enc_1d_graph = PositionalEncoding1D(self.pos_fourier_graph_dim)     #fourier encoding for input
                        #(batch_size, x, ch)
       
        self.max_text_len = max_text_len

         # projecting to logits
        self.logits_dim=logits_dim
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                
                AttentionQKV(dim, causal = True, one_kv_head=one_kv_head,
                          dim_head = dim_head, heads = heads, dropout = dropout, 
                            use_null_kv=self.use_null_kv, 
                            GNN_layers=GNN_layers, 
                             GNN_att_threshold_min=GNN_att_threshold_min,
                             GNN_att_threshold_max=GNN_att_threshold_max,
                            ),
                #AttentionQKV(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout, one_kv_head=one_kv_head,),
                #FeedForward(dim, mult = ff_mult, dropout = dropout)
                FeedForward(dim, mult = ff_mult, dropout = dropout) if (conv_kernel_FF==0 and FF_inner_conv==0) else FeedForward_CNN (dim,  mult = ff_mult, dropout = dropout, conv_kernel_FF=FF_conv_kernel, FF_inner_conv=FF_inner_conv_kernel, glu=FF_glu  )               
            ]))

        self.final_norm = LayerNorm(dim)#code based on https://github.com/lucidrains/parti-pytorch and other sources

from typing import List
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.nn import MultiheadAttention 
import torchvision.transforms as T

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
 
import numpy as np
import torch
from tqdm.notebook import trange, tqdm

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# normalization

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 2d relative positional bias

class RelPosBias2d(nn.Module):
    def __init__(self, size, heads):
        super().__init__()
        self.pos_bias = nn.Embedding((2 * size - 1) ** 2, heads)

        arange = torch.arange(size)

        pos = torch.stack(torch.meshgrid(arange, arange, indexing = 'ij'), dim = -1)
        pos = rearrange(pos, '... c -> (...) c')
        rel_pos = rearrange(pos, 'i c -> i 1 c') - rearrange(pos, 'j c -> 1 j c')

        rel_pos = rel_pos + size - 1
        h_rel, w_rel = rel_pos.unbind(dim = -1)
        pos_indices = h_rel * (2 * size - 1) + w_rel
        self.register_buffer('pos_indices', pos_indices)

    def forward(self, qk):
        i, j = qk.shape[-2:]

        bias = self.pos_bias(self.pos_indices[:i, :(j - 1)])
        bias = rearrange(bias, 'i j h -> h i j')

        bias = F.pad(bias, (j - bias.shape[-1], 0), value = 0.) # account for null key / value for classifier free guidance
        return bias

# feedforward

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden, bias = False),
        nn.GELU(),
        LayerNorm(dim_hidden),
        nn.Linear(dim_hidden, dim, bias = False)
    )

class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2
    
class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate)
    
    
class CausalDSConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_kernel_FF=3, dilation=1 ):
        super().__init__()
        self.ds_conv = nn.Conv1d(in_ch, out_ch, conv_kernel_FF, bias = False, groups = in_ch,stride=1,)
        self.conv_kernel_FF=conv_kernel_FF
        self.dilation=dilation

    def forward(self, x):
        x = rearrange(x, 'b n c -> b c n')
        #print (x.shape)  #b, c, n
        x = F.pad(x, ((self.conv_kernel_FF - 1) * self.dilation, 0))
        #print (x.shape) #b, c, n
        #print (x)
        x = self.ds_conv(x)
        return rearrange(x, 'b c n -> b n c')
    
class FeedForward_CNN(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        glu = False,
        swish = False,
        relu_squared = False,
        post_act_ln = False,
        dropout = 0.,
        no_bias = False,
        zero_init_output = False,
        conv_kernel_FF = 0, #if >0 use causal conv1d
        FF_inner_conv = 0 #if>0 use causal conv1d in FF layer (sandwiched) 
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()
        
        self.FF_inner_conv= FF_inner_conv
            
        self.project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, bias = not no_bias),
           # CausalDSConv(inner_dim, inner_dim, conv_kernel_FF=FF_inner_conv ) ,
            activation,
        ) if not glu else GLU(dim, inner_dim, activation)
        
        

        self.ff = nn.Sequential(
            #project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = not no_bias)
        )
        
        if self.FF_inner_conv>0:
            self.inner_conv_resnetblock1 = nn.Sequential(
            #nn.Linear(dim, inner_dim, bias = not no_bias),
            CausalDSConv(inner_dim, inner_dim, conv_kernel_FF=FF_inner_conv ) ,
            activation,
            CausalDSConv(inner_dim, inner_dim, conv_kernel_FF=FF_inner_conv ) ,
            ) 
        
            
        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])
            
        self.conv_kernel_FF=conv_kernel_FF
        if self.conv_kernel_FF>0: #will only work in non-causal model
            #define two resnet blocks
            self.resnetblock1 = nn.Sequential(
                CausalDSConv(dim, dim, conv_kernel_FF=conv_kernel_FF ),#, padding='same'),
                activation,
                CausalDSConv(dim, dim, conv_kernel_FF=conv_kernel_FF ),#, padding='same'),
            )
            self.resnetblock2 = nn.Sequential(
                CausalDSConv(dim_out, dim_out, conv_kernel_FF=conv_kernel_FF ),#, padding='same'),
                activation,
                CausalDSConv(dim_out, dim_out, conv_kernel_FF=conv_kernel_FF ),#, padding='same'),
            )

    def forward(self, x):
        
 
        if self.conv_kernel_FF>0:
            #x=self.resnetblock1(x.permute (0,2,1)).permute (0,2,1)+x
            x=self.resnetblock1(x )+x
            
        
        x=self.project_in (x)
        
        if self.FF_inner_conv>0:
            x=self.inner_conv_resnetblock1(x)+x

        x=self.ff(x)
        
        if self.conv_kernel_FF>0:
            #x=self.resnetblock2(x.permute (0,2,1)).permute (0,2,1)+x
            x=self.resnetblock2(x )+x
        
        return x
    
# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        norm_context = False,
        rel_pos_bias = False,
        encoded_fmap_size = None
    ):
        super().__init__()
        self.causal = causal
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)

        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, inner_dim, bias = False),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        # needed for classifier free guidance for transformers
        # by @crowsonkb, adopted by the paper

        self.null_kv = nn.Parameter(torch.randn(dim_head))

        # one-headed key / value attention, from Shazeer's multi-query paper, adopted by Alphacode and PaLM

        self.to_kv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, dim_head, bias = False)
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

        # positional bias

        self.rel_pos_bias = None

        if rel_pos_bias:
            assert exists(encoded_fmap_size)
            self.rel_pos_bias = RelPosBias2d(encoded_fmap_size, heads)

    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        batch, device = x.shape[0], x.device

        x = self.norm(x)

        q = self.to_q(x) * self.scale

        context = default(context, x)
        context = self.norm_context(context)

        kv = self.to_kv(context)

        null_kv = repeat(self.null_kv, 'd -> b 1 d', b = batch)
        kv = torch.cat((null_kv, kv), dim = 1)

        sim = einsum('b h i d, b j d -> b h i j', q, kv)

        if exists(self.rel_pos_bias):
            pos_bias = self.rel_pos_bias(sim)
            sim = sim + pos_bias

        mask_value = -torch.finfo(sim.dtype).max

        if exists(context_mask):
            context_mask = F.pad(context_mask, (1, 0), value = True)
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        out = einsum('b h i j, b j d -> b h i d', attn, kv)

        return self.to_out(out)
    
#########################################################################################################  
#https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py
#########################################################################################################
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
    
    
    #############################################
    
    # classes
def pad_sequence (output_xyz, max_length):         #pad
    
    device = output_xyz.device
    output=torch.zeros((output_xyz.shape[0],  output_xyz.shape[1] , max_length)).to(device)
    output[:,:,:output_xyz.shape[2]]=output_xyz #just positions for now....
    return output


class MoleculeTransformer (nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        logits_dim=32, 
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        text_embed_dim = None,
        cond_drop_prob = 0.25,
        max_text_len = 128,
        pos_fourier_graph_dim= 32, #entire graph fourier embedding, has to be == dim in case of embedding since we add fourier to signal
        
    ):
        super().__init__()
        use_embedding = False
      
        self.text_embed_dim =  text_embed_dim 
        self.pos_fourier_graph_dim=pos_fourier_graph_dim

        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last. This converts sequence conditioning to higher dimension
        self.GELUact= nn.GELU()
        ################
        
        self.p_enc_1d = PositionalEncoding1D(text_embed_dim)     #fourier encoding for conditioning sequence
        self.p_enc_1d_graph = PositionalEncodingPermute1D(self.pos_fourier_graph_dim)     #fourier encoding for input
        
        self.max_text_len = max_text_len
    
        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob # classifier free guidance for transformers - @crowsonkb
        
        self.start_token = nn.Parameter(torch.randn(dim))

        # projecting to logits
        self.logits_dim=logits_dim
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, causal = True, rel_pos_bias = False, 
                          dim_head = dim_head, heads = heads, dropout = dropout),
                Attention(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
            ]))

        self.final_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)
         
        self.to_dim = nn.Linear( self.logits_dim+self.pos_fourier_graph_dim, dim, bias = False)
        
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        sequences=None,#conditioning
         text_mask=None,
        *,
            tokens_to_generate=32,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
    ):
        device = next(self.parameters()).device
  
        batch = sequences.shape[0]
     
        image_seq_len=tokens_to_generate
        
        print (f"Generating {image_seq_len} tokens...")
    
        output = torch.empty((batch, self.logits_dim, 0), device = device )
        
        for _ in tqdm(range(image_seq_len)):

            sampled  = self.forward_with_cond_scale(
            
                sequences = sequences,
                text_mask = text_mask,
                output = output,

            )

            sampled=sampled[:, -1, :]
            sampled = rearrange(sampled, 'b h -> b h 1')
            output = torch.cat((output, sampled), dim = -1)
        
        return output
 

    def forward_with_cond_scale(self, *args, cond_scale = 3, **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        texts: List[str] = None,
        sequences = None,
        output=None,
        text_mask = None,
        cond_drop_prob = None,
        return_loss = False
    ):
       
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        device = next(self.parameters()).device
        ########################## conditioning #################################### 
        
        cond_x=sequences.float().unsqueeze(2)
   
        # encoding 
        cond_x= self.fc1(cond_x)
        cond_x=self.GELUact(cond_x)  
        pos_fourier_xy=self.p_enc_1d(cond_x) 
        cond_x=cond_x+pos_fourier_xy
        
        start_tokens = repeat(self.start_token, 'd -> b 1 d', b = output.shape[0])

        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           self.pos_fourier_graph_dim,
                                                           output.shape[2] ).to(device) )  
        output=torch.cat( (output, pos_fourier_graph ), 1)
        output=torch.permute(output, (0,2,1)  )

        output = self.to_dim(output)            
        
        output = torch.cat((start_tokens, output), dim = 1)
        
        if return_loss:
            output, target = output[:, :-1,:], output[:, 1:,:self.logits_dim]
            
        # enforce max text len
        if not exists(text_mask):
            text_mask = torch.ones(cond_x.shape[:2], dtype = torch.bool).to(device)
            
        cond_x, text_mask = map(lambda t: t[:, :self.max_text_len], (cond_x, text_mask))        

        # classifier free guidance conditional dropout
        batch=output.shape[0]
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        # attend

        x = self.init_norm(output)

        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context = cond_x, context_mask = text_mask) + x
            x = ff(x) + x

        x = self.final_norm(x)

        # to logits

        logits = self.to_logits(x)
         
        if not return_loss:
            return logits

        loss = F.mse_loss(
                    # logits, output
                    logits, target
                )

        return loss
    
class MoleculeTransformerSequence(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        logits_dim=32, 
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        text_embed_dim = None,
        cond_drop_prob = 0.25,
        max_text_len = 128,
       # pos_fourier_graph_dim= 32, #entire graph fourier embedding, has to be == dim in case of embedding since we add fourier to signal

    ):
        super().__init__()

        pos_fourier_graph_dim=dim
     
        self.text_embed_dim =  text_embed_dim 
        self.pos_fourier_graph_dim=pos_fourier_graph_dim

            
        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last. This converts sequence conditioning to higher dimension
        self.GELUact= nn.GELU()
        ################
        
        self.p_enc_1d = PositionalEncoding1D(text_embed_dim)     #fourier encoding for conditioning sequence
        self.p_enc_1d_graph = PositionalEncoding1D(self.pos_fourier_graph_dim)     #fourier encoding for input
                        #(batch_size, x, ch)
       
        self.max_text_len = max_text_len

        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob # classifier free guidance for transformers - @crowsonkb

        self.start_token = nn.Parameter(torch.randn(dim))

        # projecting to logits
        self.logits_dim=logits_dim
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                
                Attention(dim, causal = True, rel_pos_bias = False, 
                          dim_head = dim_head, heads = heads, dropout = dropout),
                
                
                
                Attention(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
            ]))

        self.final_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)
         
        self.token_embed = nn.Embedding(logits_dim, dim)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        sequences=None,#conditioning
         text_mask=None,
        #texts,
        *,
        tokens_to_generate=32,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
        output=None, #should provide it with at least a start token
    ):
        device = next(self.parameters()).device

        batch = sequences.shape[0]

        image_seq_len=tokens_to_generate
        
        if output==None:
            output= torch.randint (0,self.logits_dim, (batch,  1), device = device  )
            print ("Since start token not provided, generating random token.")
        
        for _ in tqdm(range(image_seq_len)):
            
            
            sampled  = self.forward_with_cond_scale(
                
                    sequences = sequences,
                    text_mask = text_mask,
                    output = output,
                     
                )
            sampled=sampled[:, -1 ]
            
            filtered_logits = top_k(sampled, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            sampled = rearrange(sampled, 'b -> b 1')
            
            output = torch.cat((output, sampled), dim = -1)

        return output
 

    def forward_with_cond_scale(self, *args, cond_scale = 3, **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        texts: List[str] = None,
        sequences = None,
        output=None,
        text_mask = None,
    
        cond_drop_prob = None,
        return_loss = False
    ):
       
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        device = next(self.parameters()).device
        ########################## conditioning #################################### 
        
        cond_x=sequences.float().unsqueeze(2)

        # encoding 
        cond_x= self.fc1(cond_x)
        cond_x=self.GELUact(cond_x)  
        pos_fourier_xy=self.p_enc_1d(cond_x) 
        cond_x=cond_x+pos_fourier_xy
         
        if return_loss:
            labels  = output [:, 1:] #labels are non embedded
        
        output = self.token_embed(output.long())
        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           output.shape[1],
                                                          self.pos_fourier_graph_dim).to(device) )  
  
        output=output+pos_fourier_graph
        
        # enforce max text len

        if not exists(text_mask):
            text_mask = torch.ones(cond_x.shape[:2], dtype = torch.bool).to(device)
            
        cond_x, text_mask = map(lambda t: t[:, :self.max_text_len], (cond_x, text_mask))        

        # classifier free guidance conditional dropout
        batch=output.shape[0]
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        # attend
        x = self.init_norm(output)
        for self_attn, cross_attn, ff in self.layers:
            
            x = self_attn(x) + x
            x = cross_attn(x, context = cond_x, context_mask = text_mask) + x
            s_causal=False
            
            x = ff(x) + x

        x = self.final_norm(x)

        # to logits
        logits = self.to_logits(x)
         
        if not return_loss:
            return logits
        
        logits=logits[:,:-1,:]
       
        loss = F.cross_entropy(
            rearrange(logits , 'b n c -> b c n'),
            labels ,
            # ignore_index = 0
            )
        
        return loss

############## same but with flexible internal dimension (in previous: embed_dim is equal to dim)
class MoleculeTransformerSequenceInternaldim(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        max_tokens=32,
        logits_dim=32, 
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        embed_dim = 16, #for input sequence
        text_embed_dim = None, #for cnditining 
        cond_drop_prob = 0.25,
        max_text_len = 128,
        one_kv_head=True,
        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.text_embed_dim =  text_embed_dim 
        self.pos_fourier_graph_dim=embed_dim

        #sequence to embedding    
        self.token_embed = nn.Embedding(max_tokens, embed_dim)
        #embedding to dim
        self.to_dim = nn.Linear( self.embed_dim, dim, bias = False)
            
        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last. This converts sequence conditioning to higher dimension
        self.GELUact= nn.GELU()
        ################
        
        self.p_enc_1d = PositionalEncoding1D(text_embed_dim)     #fourier encoding for conditioning sequence
        self.p_enc_1d_graph = PositionalEncoding1D(self.pos_fourier_graph_dim)     #fourier encoding for input
                        #(batch_size, x, ch)
       
        self.max_text_len = max_text_len

        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob # classifier free guidance for transformers - @crowsonkb

        self.start_token = nn.Parameter(torch.randn(dim))

        # projecting to logits
        self.logits_dim=logits_dim
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                
                AttentionQKV(dim, causal = True, one_kv_head=one_kv_head,
                          dim_head = dim_head, heads = heads, dropout = dropout),
                AttentionQKV(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout, one_kv_head=one_kv_head,),
                
                FeedForward(dim, mult = ff_mult, dropout = dropout)
                
            ]))

        self.final_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)
        

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        sequences=None,#conditioning
         text_mask=None,
        #texts,
        *,
        tokens_to_generate=32,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
        output=None, #should provide it with at least a start token
    ):
        device = next(self.parameters()).device

        batch = sequences.shape[0]

        image_seq_len=tokens_to_generate
        
        if output==None:
            output= torch.randint (0,self.logits_dim, (batch,  1), device = device  )
            print ("Since start token not provided, generating random token.")
        
        for _ in tqdm(range(image_seq_len)):
            
            
            sampled  = self.forward_with_cond_scale(
                
                    sequences = sequences,
                    text_mask = text_mask,
                    output = output,
                     
                )
            sampled=sampled[:, -1 ]
            
            filtered_logits = top_k(sampled, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            sampled = rearrange(sampled, 'b -> b 1')
            
            output = torch.cat((output, sampled), dim = -1)

        return output


    def forward_with_cond_scale(self, *args, cond_scale = 3, **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        texts: List[str] = None,
        sequences = None,
        output=None,
        text_mask = None,
    
        cond_drop_prob = None,
        return_loss = False
    ):
       
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        device = next(self.parameters()).device
        ########################## conditioning #################################### 
        
        cond_x=sequences.float().unsqueeze(2)

        # encoding 
        cond_x= self.fc1(cond_x)
        cond_x=self.GELUact(cond_x)  
        pos_fourier_xy=self.p_enc_1d(cond_x) 
        cond_x=cond_x+pos_fourier_xy
         
        if return_loss:
            labels  = output [:, 1:] #labels are non embedded
        
        output = self.token_embed(output.long())
        
        
        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           output.shape[1],
                                                          self.pos_fourier_graph_dim).to(device) )  
  
        output=output+pos_fourier_graph
               
        # enforce max text len

        if not exists(text_mask):
            text_mask = torch.ones(cond_x.shape[:2], dtype = torch.bool).to(device)
            
        cond_x, text_mask = map(lambda t: t[:, :self.max_text_len], (cond_x, text_mask))        

        # classifier free guidance conditional dropout
        batch=output.shape[0]
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        x = self.to_dim (output)
        
        x = self.init_norm(x)
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context = cond_x, context_mask = text_mask) + x
            x = ff(x) + x

        x = self.final_norm(x)

        # to logits
        logits = self.to_logits(x)
         
        if not return_loss:
            return logits
        
        logits=logits[:,:-1,:]
       
        loss = F.cross_entropy(
            rearrange(logits , 'b n c -> b c n'),
            labels ,
            # ignore_index = 0
            )
        
        return loss

############### transformer encoder ###################

from functools import partial, wraps
def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner



    
#######################################################################

class MoleculeTransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        logits_dim=32, 
        logits_dim_length = None , #if None entire length is used. Otherwise, max_length is projected to logits_dim_length
        max_length = None,         #see previous - OUTPUT IS (b, logits_dim, logits_dim_length)
        max_tokens = 32, #max tokens for embedding layer
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
      
        embed_dim = 16,
     
        padding_token=0,#will be used to create mask

    ):
        super().__init__()

      #  pos_fourier_graph_dim=dim
        self.padding_token=padding_token
     
        self.embed_dim=embed_dim #embedding dim of sequence
        self.max_length=max_length
        self.logits_dim_length=logits_dim_length

        self.GELUact= nn.GELU()
        ################
        
        self.p_enc_1d_graph = PositionalEncoding1D(self.embed_dim)     #fourier encoding for input
                        #(batch_size, x, ch)
       
        # projecting to logits
        self.logits_dim = logits_dim
        self.init_norm  = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                #AttentionQKV(dim, causal = False, 
                #          dim_head = dim_head, heads = heads, dropout = dropout),
                MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                #Attention(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout) 
            ]))

        self.final_norm = LayerNorm(dim)
         
        #sequence to embedding    
        self.token_embed = nn.Embedding(max_tokens, embed_dim)
        #embedding to dim
        self.to_dim = nn.Linear( self.embed_dim, dim, bias = False)
        #dim to logits
        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)
        
        if exists (self.logits_dim_length):
            assert exists (self.max_length), 'max_length and logits_dim_length must be set to nonzero value'
            self.to_logits_dim_length = nn.Linear(self.max_length, self.logits_dim_length, bias = False)

    def forward(
        self,
    
        input_sequence,
        text_mask = None,
        return_hidden = False,
        squeeze_output=False,
    
    ):
    
        device = next(self.parameters()).device
        ########################## conditioning #################################### 
       
        output = self.token_embed(input_sequence)   
        
        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           output.shape[1],
                                                           self.embed_dim).to(device) )  
  
        output=output+pos_fourier_graph
    
        x = self.to_dim (output)
    
        if exists (self.max_length):  # enforce max text len
            if not exists(text_mask):
                text_mask = torch.ones(x.shape[:2], dtype = torch.bool).to(device)
                text_mask = input_sequence.eq(self.padding_token) #use  0 will turn to True, others to False. All values with True are masked out.
            
        x, text_mask = map(lambda t: t[:, :self.max_length], (x, text_mask))     
       
        x = self.init_norm(x)
        #for self_attn, cross_attn, ff in self.layers:
        for self_attn, ff in self.layers:
            #x = self_attn(x, context_mask = text_mask) + x
            attout, _ = self_attn(x, x, x, key_padding_mask  = text_mask)
            x= attout + x
            
            #x = cross_attn(x, context = cond_x, context_mask = text_mask) + x
            x = ff(x) + x

        x = self.final_norm(x)

        # to logits
        logits = self.to_logits(x)
        
        if exists(self.logits_dim_length):
            logits=logits.permute (0,2,1)
            logits = self.to_logits_dim_length(logits)
            
         
        if return_hidden:
            
            return  x
    
        else:
            if squeeze_output:
                logits=logits.squeeze()
            
            return logits
        
        
######################### GPT with Graph, etc. #######################################

class AttentionQKV(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        norm_context = False,
        one_kv_head=True,
        use_null_kv=True,
        GNN_layers = 0,
        GNN_aggr = 'add',
        GNN_have_skip = True,
        GNN_att_threshold_min=0,#if >0 then set all values in att_matrix to 0 that are smaller
        GNN_att_threshold_max=1,#if <1 then set all values in att_matrix to 1 that are larger
        GNN_add_identity= True, #whether to add identity
        GNN_clamp_att_after_identity = True, #whether to clamp attention after identity is added
        
    ):
        super().__init__()
        self.causal = causal
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)
        self.use_null_kv=use_null_kv
        self.GNN_layers=GNN_layers
        self.GNN_att_threshold_min=GNN_att_threshold_min
        self.GNN_att_threshold_max=GNN_att_threshold_max
        self.GNN_add_identity= GNN_add_identity
        self.heads=heads
        self.GNN_clamp_att_after_identity=GNN_clamp_att_after_identity

        context_dim = default(context_dim, dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()
        
        self.one_kv_head = one_kv_head #If true: Use one KV head, multiple query heads

        # needed for classifier free guidance for transformers
        # by @crowsonkb, adopted by the paper
        
        q_dim = k_dim = dim_head * heads
        v_dim = out_dim = dim_head * heads

        # one-headed key / value attention, from Shazeer's multi-query paper, adopted by Alphacode and PaLM
        # https://arxiv.org/abs/1911.02150 - only have one head for the key / values, multi-headed queries
        self.one_kv_head = one_kv_head
        if one_kv_head:
            k_dim = dim_head
            v_dim = dim_head
            out_dim = v_dim * heads

        self.null_k = nn.Parameter(torch.randn(k_dim))
        self.null_v = nn.Parameter(torch.randn(v_dim))
        
        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, q_dim, bias = False),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )
        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, k_dim, bias = False)
        )
        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, v_dim, bias = False)
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
          
            nn.Linear(out_dim, dim, bias = False),
            LayerNorm(dim)
        )
        
         
        if self.GNN_layers>0:
            self.GNN_layers=GNN_layers
            #GNN_hidden_channels=dim
            #print ("q_dim ",q_dim) 
            self.GNN_net = GraphConvLayers(node_features_in=dim_head,#q_dim, 
                                    num_node_features_out=dim_head,#,q_dim, 
                                    hidden_channels=dim_head,#q_dim,
                                    depth=GNN_layers,
                                   # aggr = GNN_aggr,
                                    have_skip = GNN_have_skip,
                                   )



    def forward(
        self,
        x,
        context = None,
        context_mask = None,
        plot_attn= False,
        
        
    ):
        batch, device = x.shape[0], x.device

        x = self.norm(x)

        q = self.to_q(x) * self.scale

        context = default(context, x)
        context = self.norm_context(context)

        k = self.to_k(context)
        v = self.to_v(context)

        if self.use_null_kv:
            null_k = repeat(self.null_k, 'd -> b 1 d', b = batch)
            k = torch.cat((null_k, k), dim = 1)
            null_v = repeat(self.null_v, 'd -> b 1 d', b = batch)
            v = torch.cat((null_v, v), dim = 1)

        
        if not self.one_kv_head:
            k, v = map(lambda t: maybe(rearrange)(t, 'b n (h d) -> b h n d', h = self.heads), (k, v))
            
        kv_einsum_eq = 'b h j d' if not self.one_kv_head else 'b j d'

        #dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

        sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) #* scale

        mask_value = -torch.finfo(sim.dtype).max

        if exists(context_mask):
            if self.use_null_kv:
                context_mask = F.pad(context_mask, (1, 0), value = True)
            
            #original
            #context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            
            if context_mask.ndim == 2:#original ... 
                #attn_mask = rearrange(attn_mask, 'i j -> 1 1 i j')
                context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            #elif context_mask.ndim == 3:#not sure if works...
            #    #attn_mask = rearrange(attn_mask, 'h i j -> 1 h i j')
            #    context_mask = rearrange(context_mask, 'b j h -> b 1 j h')
            #context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            
            sim = sim.masked_fill(~context_mask, mask_value)
            #dots = dots.masked_fill(~input_mask, mask_value)
            
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)
            
            #causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            #dots = dots.masked_fill(causal_mask, mask_value)            

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        
        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)
        
        ######### Graph neural net ###############################
        #GNN layers are added after causal mask is applied to maintain causality
        #GNN is constructed by using attn as adjancy matrix and/or edge features, shape is d,d
        #    node features are defined by v here: shape is n,d
        #    attn can be processed with min/max threshold, identity added, etc.
        #    TO DO: implement other aggregation schemes, multi hop message passing....
        if self.GNN_layers>0:
            #print ("out shape" , out.shape)
            #print ("Att bef shape: ", attn.shape)
            # attn=  rearrange (attn,'b h n d -> b n (h d)')
            attn_comb=  rearrange (attn,'b h n d -> (b h) n d') #these are the attention matrices - batch and head grouped
            if self.GNN_add_identity:
                attn_comb = attn_comb + torch.eye(attn_comb.shape[1]).unsqueeze(0).repeat(attn_comb.shape[0],1,1).to(device)
                if self.GNN_clamp_att_after_identity:
                    attn_comb = torch.clamp(attn_comb, min=0, max=1)

                #attn_comb =attn_comb.softmax (dim=-1, dtype = torch.float32)
                
            if self.GNN_att_threshold_min>0:
                attn_comb[attn_comb<self.GNN_att_threshold_min] = 0
            if self.GNN_att_threshold_max<1:
                attn_comb[attn_comb>self.GNN_att_threshold_max] = 1
                
            if plot_attn:
                grid_disc=int (attn_comb.shape [0]**0.5)
                fig, axes = plt.subplots(nrows=grid_disc, ncols=grid_disc, figsize=(8,8))
                c_i=0
                for i in range (grid_disc):
                    for j in range (grid_disc):
                        axes[i,j].imshow (attn_comb[c_i,:].detach().numpy() )
                        #axes[i,j].colorbar()
                        c_i+=1

                # fig.colorbar (ax=fig.get_axes())
                plt.show()
            
            #v_comb=  rearrange (v,'b h n d -> b n (h d)')
            v_comb=  rearrange (v,'b h n d -> (b h) n d') #these are the node embeddings - batch and head grouped
            
            #print ("v_comb.shape",v_comb.shape, "att_comb shape shape: ", attn_comb.shape)
            #print (edge_index.shape, edge_weight.shape)

            #data=torch.randn (4, 16, dim) #b, n, c
            out_from_GNN = self.GNN_net(v_comb, attn_comb,) 
            
            
            
            #print (out_from_GNN.shape)
            #attn_comb=  rearrange (attn,'(b h) n d -> b h n d')
            out = out +  rearrange (out_from_GNN,'(b h) n d -> b h n d', h = self.heads)
            
            
            #print (out_from_GNN.shape, attn.shape)

        
        #out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)
        

        return self.to_out(out)
    
#######################################################################

class MoleculeTransformerGPT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        max_tokens=32,
        logits_dim=32, 
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        embed_dim = 16, #for input sequence
        text_embed_dim = 16,# None, #not used 
        max_text_len = 128,
        one_kv_head=True,
        concat_pos_encoding = False, #if True, then pos encoding will be added to the input; and pos_fourier_graph_dim=embed_dim
        pos_fourier_graph_dim  = None,  #only used if concat_pos_encoding == True
        use_null_kv=True, #use null_kv for CFG
        FF_conv_kernel=0, #if >0 then use conv in FF layer, at beginning and end
        FF_inner_conv_kernel=0,
        FF_glu=False,
        GNN_layers = 0, #whether or not to use GNN_layers after softmax
        GNN_att_threshold_min=0.,
        GNN_att_threshold_max=1.,
        
    ):
        super().__init__()

        # pos_fourier_graph_dim=embed_dim
        self.embed_dim = embed_dim
        self.concat_pos_encoding=concat_pos_encoding
        self.use_null_kv=use_null_kv
        #  self.text_embed_dim =  text_embed_dim 
        if concat_pos_encoding == False:
            self.pos_fourier_graph_dim=embed_dim
        else:
            self.pos_fourier_graph_dim=pos_fourier_graph_dim
            assert pos_fourier_graph_dim !=None, "pos_fourier_graph_dim has to be set if concatenating pos embedding"
            print ("Concatenate positional encoding...")

        #sequence to embedding    
        self.token_embed = nn.Embedding(max_tokens, embed_dim)
        #embedding to dim
        
        dim_in = self.embed_dim + self.concat_pos_encoding*self.pos_fourier_graph_dim
        self.to_dim = nn.Linear(dim_in, 
                                dim, bias = False)
        print ("Input dimension (signal and pos encoding): ", dim_in)
        
            
        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last. This converts sequence conditioning to higher dimension -- NOT USED, TODO: Remove
        self.GELUact= nn.GELU()
        ################
        self.p_enc_1d_graph = PositionalEncoding1D(self.pos_fourier_graph_dim)     #fourier encoding for input
                        #(batch_size, x, ch)
       
        self.max_text_len = max_text_len

         # projecting to logits
        self.logits_dim=logits_dim
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                
                AttentionQKV(dim, causal = True, one_kv_head=one_kv_head,
                          dim_head = dim_head, heads = heads, dropout = dropout, 
                            use_null_kv=self.use_null_kv, 
                            GNN_layers=GNN_layers, 
                             GNN_att_threshold_min=GNN_att_threshold_min,
                             GNN_att_threshold_max=GNN_att_threshold_max,
                            ),
                #AttentionQKV(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout, one_kv_head=one_kv_head,),
                #FeedForward(dim, mult = ff_mult, dropout = dropout)
                FeedForward(dim, mult = ff_mult, dropout = dropout) if (conv_kernel_FF==0 and FF_inner_conv==0) else FeedForward_CNN (dim,  mult = ff_mult, dropout = dropout, conv_kernel_FF=FF_conv_kernel, FF_inner_conv=FF_inner_conv_kernel, glu=FF_glu  )               
            ]))

        self.final_norm = LayerNorm(dim)
        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        *,
        output=None, #should provide it with at least a start token
        tokens_to_generate=32,
        filter_thres = 0.9,
        temperature = 1.,
       
        use_gumbel_sample=True, #whether or not ot use gumbel sampling 
    ):
        device = next(self.parameters()).device

        batch = output.shape[0]

        image_seq_len=tokens_to_generate
        
        if output==None:
            output= torch.randint (0,self.logits_dim, (batch,  1), device = device  )
            print ("Since start token not provided, generating random token.")
        
        for _ in tqdm(range(image_seq_len)):
            
            sampled  = self.forward(
                output = output,
                )
            sampled=sampled[:, -1 ]
            
            if use_gumbel_sample:
                filtered_logits = top_k(sampled, thres = filter_thres)
                sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
                sampled = rearrange(sampled, 'b -> b 1')
            else:
                sampled = sampled.argmax(dim = -1)
                sampled = rearrange(sampled, 'b -> b 1')
                
            output = torch.cat((output, sampled), dim = -1)

        return output
 
    def forward(
        self,
        output=None,
        return_loss = False,
        ignore_padding_zeros=False,
        mask_prob=0.,# if > 0: Apply BERT style LLM mask
        context_mask = None,
        
    ):
       
        device = next(self.parameters()).device
        
        if return_loss:
            labels  = output [:, 1:] #labels are non embedded
        
        output = self.token_embed(output.long())
        
        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           output.shape[1],
                                                          self.pos_fourier_graph_dim).to(device) )  
                                    #(batch_size, x, ch)
            
            
        if self.concat_pos_encoding == False: #if False then additive ... 
            output=output+pos_fourier_graph
        else:
            output= torch.cat((output, pos_fourier_graph), dim = -1)  
            
        batch=output.shape[0]
       
        x = self.to_dim (output)
        
        x = self.init_norm(x)
       
        
        #Apply BERT style LLM mask if selected
        
        if mask_prob > 0.:
            rand = torch.randn( (output.shape[0], x.shape[1]), device = x.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max # first token should not be masked out
            
            num_mask = min(int(output.shape[1] * mask_prob), output.shape[1] - 1)
            
            indices = rand.topk(num_mask, dim = -1).indices #torch.topk
                                    #Returns the k largest elements of the given 
                                    #input tensor along a given dimension
                    
            context_mask = ~torch.zeros_like(output[:,:,0]).squeeze().scatter(1, indices, 1.).bool()
                            #torch.scatter(input, dim, index, src) 
                            #the ones selected by random via indices, i.e. 1s, are going to be False
            
            #now repeat in channel direction
            #context_mask = repeat(context_mask, 'b n -> b n c', c = output.shape[2])
            
            #sim = sim.masked_fill(~context_mask, mask_value)
            
            #kwargs.update(self_attn_context_mask = mask)  
            
        
        
        
        for self_attn, ff in self.layers:
            x = self_attn(x, context_mask=context_mask) + x
            x = ff(x) + x
        x = self.final_norm(x)

        # to logits
        logits = self.to_logits(x)
         
        if not return_loss:
            return logits
        
        logits=logits[:,:-1,:]
       
        if ignore_padding_zeros:
            loss = F.cross_entropy(
                rearrange(logits , 'b n c -> b c n'),
                labels ,
                ignore_index = 0,
                )            
        else:
            
            loss = F.cross_entropy(
                rearrange(logits , 'b n c -> b c n'),
                labels ,
                #ignore_index = 0
                )

        return loss

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_edge_index,dense_to_sparse

class GCNLayer(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True) #used for normalization 
        node_feats = self.projection(node_feats) #Apply linear layer (filter) to each of the node features (same weights!)
        node_feats = torch.bmm(adj_matrix, node_feats) #Add up all the node features  using the adj matrix
        node_feats = node_feats / num_neighbours #normalize
        return node_feats
    
class GraphConvLayers(torch.nn.Module):
    def __init__(self, 
                 node_features_in, num_node_features_out, hidden_channels,
                 depth,
              #   aggr = 'add',
                 have_skip = False,
                ):
        super(GraphConvLayers, self).__init__()
        #torch.manual_seed(12345)
        #assert node_features_in == hidden_channels if have_skip, 'If use skip connection, must have node_features_in== hidden_channels'
        self.layers = []
        
        self.have_skip = have_skip
        
        for i in range(depth):
            self.layers.append(
                
#                GraphConv(hidden_channels, hidden_channels, aggr=aggr) if i>0 else GraphConv(node_features_in, hidden_channels, aggr=aggr)
                GCNLayer(hidden_channels, hidden_channels, ) if i>0 else GCNLayer(node_features_in, hidden_channels, )
            )
           
        self.layers = nn.ModuleList(self.layers)
        
    
        self.lin = Linear(hidden_channels, num_node_features_out)
        self.GELUact= nn.GELU()
        
        #self.mlp = Sequential(Linear(hidden_channels, 30), ReLU(), Linear(30, 15), ReLU(), Linear(15, dataset.num_classes))

    def forward(self, x, adj_matrix):
        
        #forward(x: Union[Tensor, Tuple[Tensor, Optional[Tensor]]], 
        #edge_index: Union[Tensor, SparseTensor], edge_weight: Optional[Tensor] = None, size: Optional[Tuple[int, int]] = None)→ Tensor
        
        for GNNlayer in self.layers:
            #x = GNNlayer(x, edge_index, edge_weight) + x*self.have_skip
            x = GNNlayer(x, adj_matrix) + x*self.have_skip
            x = self.GELUact(x)
         
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        
        
        return x    
######################### PyTorch implementation of GPT #########################
#There may be a bug in the MHA module in PyTorch (needs to be tested)

class MoleculeTransformerGPTPyTorch(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        max_tokens=32,
        logits_dim=32, 
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        embed_dim = 16, #for input sequence
        text_embed_dim = None, #not used 
        max_text_len = 128,
        one_kv_head=True,
        concat_pos_encoding = False, #if True, then pos encoding will be added to the input; and pos_fourier_graph_dim=embed_dim
        pos_fourier_graph_dim  = None,  #only used if concat_pos_encoding == True
    ):
        super().__init__()

       # pos_fourier_graph_dim=embed_dim
        self.embed_dim = embed_dim
        self.concat_pos_encoding=concat_pos_encoding
        #  self.text_embed_dim =  text_embed_dim 
        if concat_pos_encoding == False:
            self.pos_fourier_graph_dim=embed_dim
        else:
            self.pos_fourier_graph_dim=pos_fourier_graph_dim
            assert pos_fourier_graph_dim !=None, "pos_fourier_graph_dim has to be set if concatenating pos embedding"
            print ("Concatenate positional encoding...")

        #sequence to embedding    
        self.token_embed = nn.Embedding(max_tokens, embed_dim)
        #embedding to dim
        
        dim_in = self.embed_dim + self.concat_pos_encoding*self.pos_fourier_graph_dim
        self.to_dim = nn.Linear(dim_in, 
                                dim, bias = False)
        print ("Input dimension (signal and pos encoding): ", dim_in)
        
            
        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last. This converts sequence conditioning to higher dimension
        self.GELUact= nn.GELU()
        ################
        self.p_enc_1d_graph = PositionalEncoding1D(self.pos_fourier_graph_dim)     #fourier encoding for input
                        #(batch_size, x, ch)
       
        self.max_text_len = max_text_len

         # projecting to logits
        self.logits_dim=logits_dim
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                
                #AttentionQKV(dim, causal = True, one_kv_head=one_kv_head,
                #          dim_head = dim_head, heads = heads, dropout = dropout),
                #AttentionQKV(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout, one_kv_head=one_kv_head,),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
                
            ]))

        self.final_norm = LayerNorm(dim)
        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        *,
        tokens_to_generate=32,
        filter_thres = 0.9,
        temperature = 1.,
        output=None, #should provide it with at least a start token
        use_gumbel_sample=True, #whether or not ot use gumbel sampling 
    ):
        device = next(self.parameters()).device

        batch = output.shape[0]

        image_seq_len=tokens_to_generate
        
        if output==None:
            output= torch.randint (0,self.logits_dim, (batch,  1), device = device  )
            print ("Since start token not provided, generating random token.")
        
        for _ in tqdm(range(image_seq_len)):
            
            sampled  = self.forward(
                output = output,
                )
            sampled=sampled[:, -1 ]
            
            if use_gumbel_sample:
                filtered_logits = top_k(sampled, thres = filter_thres)
                sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
                sampled = rearrange(sampled, 'b -> b 1')
            else:
                sampled = sampled.argmax(dim = -1)
                sampled = rearrange(sampled, 'b -> b 1')
                
            output = torch.cat((output, sampled), dim = -1)

        return output
 
    def forward(
        self,
        output=None,
        return_loss = False,
        ignore_padding_zeros=False,
    ):
       
        device = next(self.parameters()).device
        
        if return_loss:
            labels  = output [:, 1:] #labels are non embedded
        
        output = self.token_embed(output.long())
        
        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           output.shape[1],
                                                          self.pos_fourier_graph_dim).to(device) )  
                                    #(batch_size, x, ch)
            
            
        if self.concat_pos_encoding == False: #if False then additive ... 
            output=output+pos_fourier_graph
        else:
            output= torch.cat((output, pos_fourier_graph), dim = -1)  
            
        batch=output.shape[0]
       
        x = self.to_dim (output)
        
        x = self.init_norm(x)
       
        for self_attn, ff in self.layers:
            #x = self_attn(x, x, x, ) + x
            #attout, _ = self_attn(x, x, x, is_causal = True )
            attout  = self_attn(x, x, x, is_causal = True, need_weights = False)[0]
            x= attout + x
            
            x = ff(x) + x
        x = self.final_norm(x)

        # to logits
        logits = self.to_logits(x)
         
        if not return_loss:
            return logits
        
        logits=logits[:,:-1,:]#code based on https://github.com/lucidrains/parti-pytorch and other sources

from typing import List
from functools import partial

import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch.nn import MultiheadAttention 
import torchvision.transforms as T

from einops import rearrange, repeat
from einops.layers.torch import Rearrange
 
import numpy as np
import torch
from tqdm.notebook import trange, tqdm

def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out
    return inner

# sampling helpers

def log(t, eps = 1e-20):
    return torch.log(t + eps)

def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))

def gumbel_sample(t, temperature = 1., dim = -1):
    return ((t / temperature) + gumbel_noise(t)).argmax(dim = dim)

def top_k(logits, thres = 0.5):
    num_logits = logits.shape[-1]
    k = max(int((1 - thres) * num_logits), 1)
    val, ind = torch.topk(logits, k)
    probs = torch.full_like(logits, float('-inf'))
    probs.scatter_(1, ind, val)
    return probs

# classifier free guidance functions

def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device = device, dtype = torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device = device, dtype = torch.bool)
    else:
        return torch.zeros(shape, device = device).float().uniform_(0, 1) < prob

# normalization

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.register_buffer('beta', torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)

# 2d relative positional bias

class RelPosBias2d(nn.Module):
    def __init__(self, size, heads):
        super().__init__()
        self.pos_bias = nn.Embedding((2 * size - 1) ** 2, heads)

        arange = torch.arange(size)

        pos = torch.stack(torch.meshgrid(arange, arange, indexing = 'ij'), dim = -1)
        pos = rearrange(pos, '... c -> (...) c')
        rel_pos = rearrange(pos, 'i c -> i 1 c') - rearrange(pos, 'j c -> 1 j c')

        rel_pos = rel_pos + size - 1
        h_rel, w_rel = rel_pos.unbind(dim = -1)
        pos_indices = h_rel * (2 * size - 1) + w_rel
        self.register_buffer('pos_indices', pos_indices)

    def forward(self, qk):
        i, j = qk.shape[-2:]

        bias = self.pos_bias(self.pos_indices[:i, :(j - 1)])
        bias = rearrange(bias, 'i j h -> h i j')

        bias = F.pad(bias, (j - bias.shape[-1], 0), value = 0.) # account for null key / value for classifier free guidance
        return bias

# feedforward

def FeedForward(dim, mult = 4, dropout = 0.):
    dim_hidden = int(dim * mult)
    return nn.Sequential(
        LayerNorm(dim),
        nn.Linear(dim, dim_hidden, bias = False),
        nn.GELU(),
        LayerNorm(dim_hidden),
        nn.Linear(dim_hidden, dim, bias = False)
    )

class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2
    
class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim = -1)
        return x * self.act(gate)
    
    
class CausalDSConv(nn.Module):
    def __init__(self, in_ch, out_ch, conv_kernel_FF=3, dilation=1 ):
        super().__init__()
        self.ds_conv = nn.Conv1d(in_ch, out_ch, conv_kernel_FF, bias = False, groups = in_ch,stride=1,)
        self.conv_kernel_FF=conv_kernel_FF
        self.dilation=dilation

    def forward(self, x):
        x = rearrange(x, 'b n c -> b c n')
        #print (x.shape)  #b, c, n
        x = F.pad(x, ((self.conv_kernel_FF - 1) * self.dilation, 0))
        #print (x.shape) #b, c, n
        #print (x)
        x = self.ds_conv(x)
        return rearrange(x, 'b c n -> b n c')
    
class FeedForward_CNN(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        glu = False,
        swish = False,
        relu_squared = False,
        post_act_ln = False,
        dropout = 0.,
        no_bias = False,
        zero_init_output = False,
        conv_kernel_FF = 0, #if >0 use causal conv1d
        FF_inner_conv = 0 #if>0 use causal conv1d in FF layer (sandwiched) 
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()
        
        self.FF_inner_conv= FF_inner_conv
            
        self.project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, bias = not no_bias),
           # CausalDSConv(inner_dim, inner_dim, conv_kernel_FF=FF_inner_conv ) ,
            activation,
        ) if not glu else GLU(dim, inner_dim, activation)
        
        

        self.ff = nn.Sequential(
            #project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = not no_bias)
        )
        
        if self.FF_inner_conv>0:
            self.inner_conv_resnetblock1 = nn.Sequential(
            #nn.Linear(dim, inner_dim, bias = not no_bias),
            CausalDSConv(inner_dim, inner_dim, conv_kernel_FF=FF_inner_conv ) ,
            activation,
            CausalDSConv(inner_dim, inner_dim, conv_kernel_FF=FF_inner_conv ) ,
            ) 
        
            
        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.ff[-1])
            
        self.conv_kernel_FF=conv_kernel_FF
        if self.conv_kernel_FF>0: #will only work in non-causal model
            #define two resnet blocks
            self.resnetblock1 = nn.Sequential(
                CausalDSConv(dim, dim, conv_kernel_FF=conv_kernel_FF ),#, padding='same'),
                activation,
                CausalDSConv(dim, dim, conv_kernel_FF=conv_kernel_FF ),#, padding='same'),
            )
            self.resnetblock2 = nn.Sequential(
                CausalDSConv(dim_out, dim_out, conv_kernel_FF=conv_kernel_FF ),#, padding='same'),
                activation,
                CausalDSConv(dim_out, dim_out, conv_kernel_FF=conv_kernel_FF ),#, padding='same'),
            )

    def forward(self, x):
        
 
        if self.conv_kernel_FF>0:
            #x=self.resnetblock1(x.permute (0,2,1)).permute (0,2,1)+x
            x=self.resnetblock1(x )+x
            
        
        x=self.project_in (x)
        
        if self.FF_inner_conv>0:
            x=self.inner_conv_resnetblock1(x)+x

        x=self.ff(x)
        
        if self.conv_kernel_FF>0:
            #x=self.resnetblock2(x.permute (0,2,1)).permute (0,2,1)+x
            x=self.resnetblock2(x )+x
        
        return x
    
# attention

class Attention(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        norm_context = False,
        rel_pos_bias = False,
        encoded_fmap_size = None
    ):
        super().__init__()
        self.causal = causal
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)

        inner_dim = heads * dim_head
        context_dim = default(context_dim, dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()

        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, inner_dim, bias = False),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )

        # needed for classifier free guidance for transformers
        # by @crowsonkb, adopted by the paper

        self.null_kv = nn.Parameter(torch.randn(dim_head))

        # one-headed key / value attention, from Shazeer's multi-query paper, adopted by Alphacode and PaLM

        self.to_kv = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, dim_head, bias = False)
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
            nn.Linear(inner_dim, dim, bias = False),
            LayerNorm(dim)
        )

        # positional bias

        self.rel_pos_bias = None

        if rel_pos_bias:
            assert exists(encoded_fmap_size)
            self.rel_pos_bias = RelPosBias2d(encoded_fmap_size, heads)

    def forward(
        self,
        x,
        context = None,
        context_mask = None
    ):
        batch, device = x.shape[0], x.device

        x = self.norm(x)

        q = self.to_q(x) * self.scale

        context = default(context, x)
        context = self.norm_context(context)

        kv = self.to_kv(context)

        null_kv = repeat(self.null_kv, 'd -> b 1 d', b = batch)
        kv = torch.cat((null_kv, kv), dim = 1)

        sim = einsum('b h i d, b j d -> b h i j', q, kv)

        if exists(self.rel_pos_bias):
            pos_bias = self.rel_pos_bias(sim)
            sim = sim + pos_bias

        mask_value = -torch.finfo(sim.dtype).max

        if exists(context_mask):
            context_mask = F.pad(context_mask, (1, 0), value = True)
            context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~context_mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        out = einsum('b h i j, b j d -> b h i d', attn, kv)

        return self.to_out(out)
    
#########################################################################################################  
#https://github.com/tatp22/multidim-positional-encoding/blob/master/positional_encodings/positional_encodings.py
#########################################################################################################
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
    
    
    #############################################
    
    # classes
def pad_sequence (output_xyz, max_length):         #pad
    
    device = output_xyz.device
    output=torch.zeros((output_xyz.shape[0],  output_xyz.shape[1] , max_length)).to(device)
    output[:,:,:output_xyz.shape[2]]=output_xyz #just positions for now....
    return output


class MoleculeTransformer (nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        logits_dim=32, 
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        text_embed_dim = None,
        cond_drop_prob = 0.25,
        max_text_len = 128,
        pos_fourier_graph_dim= 32, #entire graph fourier embedding, has to be == dim in case of embedding since we add fourier to signal
        
    ):
        super().__init__()
        use_embedding = False
      
        self.text_embed_dim =  text_embed_dim 
        self.pos_fourier_graph_dim=pos_fourier_graph_dim

        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last. This converts sequence conditioning to higher dimension
        self.GELUact= nn.GELU()
        ################
        
        self.p_enc_1d = PositionalEncoding1D(text_embed_dim)     #fourier encoding for conditioning sequence
        self.p_enc_1d_graph = PositionalEncodingPermute1D(self.pos_fourier_graph_dim)     #fourier encoding for input
        
        self.max_text_len = max_text_len
    
        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob # classifier free guidance for transformers - @crowsonkb
        
        self.start_token = nn.Parameter(torch.randn(dim))

        # projecting to logits
        self.logits_dim=logits_dim
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, causal = True, rel_pos_bias = False, 
                          dim_head = dim_head, heads = heads, dropout = dropout),
                Attention(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
            ]))

        self.final_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)
         
        self.to_dim = nn.Linear( self.logits_dim+self.pos_fourier_graph_dim, dim, bias = False)
        
    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        sequences=None,#conditioning
         text_mask=None,
        *,
            tokens_to_generate=32,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
    ):
        device = next(self.parameters()).device
  
        batch = sequences.shape[0]
     
        image_seq_len=tokens_to_generate
        
        print (f"Generating {image_seq_len} tokens...")
    
        output = torch.empty((batch, self.logits_dim, 0), device = device )
        
        for _ in tqdm(range(image_seq_len)):

            sampled  = self.forward_with_cond_scale(
            
                sequences = sequences,
                text_mask = text_mask,
                output = output,

            )

            sampled=sampled[:, -1, :]
            sampled = rearrange(sampled, 'b h -> b h 1')
            output = torch.cat((output, sampled), dim = -1)
        
        return output
 

    def forward_with_cond_scale(self, *args, cond_scale = 3, **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        texts: List[str] = None,
        sequences = None,
        output=None,
        text_mask = None,
        cond_drop_prob = None,
        return_loss = False
    ):
       
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        device = next(self.parameters()).device
        ########################## conditioning #################################### 
        
        cond_x=sequences.float().unsqueeze(2)
   
        # encoding 
        cond_x= self.fc1(cond_x)
        cond_x=self.GELUact(cond_x)  
        pos_fourier_xy=self.p_enc_1d(cond_x) 
        cond_x=cond_x+pos_fourier_xy
        
        start_tokens = repeat(self.start_token, 'd -> b 1 d', b = output.shape[0])

        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           self.pos_fourier_graph_dim,
                                                           output.shape[2] ).to(device) )  
        output=torch.cat( (output, pos_fourier_graph ), 1)
        output=torch.permute(output, (0,2,1)  )

        output = self.to_dim(output)            
        
        output = torch.cat((start_tokens, output), dim = 1)
        
        if return_loss:
            output, target = output[:, :-1,:], output[:, 1:,:self.logits_dim]
            
        # enforce max text len
        if not exists(text_mask):
            text_mask = torch.ones(cond_x.shape[:2], dtype = torch.bool).to(device)
            
        cond_x, text_mask = map(lambda t: t[:, :self.max_text_len], (cond_x, text_mask))        

        # classifier free guidance conditional dropout
        batch=output.shape[0]
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        # attend

        x = self.init_norm(output)

        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context = cond_x, context_mask = text_mask) + x
            x = ff(x) + x

        x = self.final_norm(x)

        # to logits

        logits = self.to_logits(x)
         
        if not return_loss:
            return logits

        loss = F.mse_loss(
                    # logits, output
                    logits, target
                )

        return loss
    
class MoleculeTransformerSequence(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        logits_dim=32, 
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        text_embed_dim = None,
        cond_drop_prob = 0.25,
        max_text_len = 128,
       # pos_fourier_graph_dim= 32, #entire graph fourier embedding, has to be == dim in case of embedding since we add fourier to signal

    ):
        super().__init__()

        pos_fourier_graph_dim=dim
     
        self.text_embed_dim =  text_embed_dim 
        self.pos_fourier_graph_dim=pos_fourier_graph_dim

            
        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last. This converts sequence conditioning to higher dimension
        self.GELUact= nn.GELU()
        ################
        
        self.p_enc_1d = PositionalEncoding1D(text_embed_dim)     #fourier encoding for conditioning sequence
        self.p_enc_1d_graph = PositionalEncoding1D(self.pos_fourier_graph_dim)     #fourier encoding for input
                        #(batch_size, x, ch)
       
        self.max_text_len = max_text_len

        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob # classifier free guidance for transformers - @crowsonkb

        self.start_token = nn.Parameter(torch.randn(dim))

        # projecting to logits
        self.logits_dim=logits_dim
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                
                Attention(dim, causal = True, rel_pos_bias = False, 
                          dim_head = dim_head, heads = heads, dropout = dropout),
                
                
                
                Attention(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
            ]))

        self.final_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)
         
        self.token_embed = nn.Embedding(logits_dim, dim)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        sequences=None,#conditioning
         text_mask=None,
        #texts,
        *,
        tokens_to_generate=32,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
        output=None, #should provide it with at least a start token
    ):
        device = next(self.parameters()).device

        batch = sequences.shape[0]

        image_seq_len=tokens_to_generate
        
        if output==None:
            output= torch.randint (0,self.logits_dim, (batch,  1), device = device  )
            print ("Since start token not provided, generating random token.")
        
        for _ in tqdm(range(image_seq_len)):
            
            
            sampled  = self.forward_with_cond_scale(
                
                    sequences = sequences,
                    text_mask = text_mask,
                    output = output,
                     
                )
            sampled=sampled[:, -1 ]
            
            filtered_logits = top_k(sampled, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            sampled = rearrange(sampled, 'b -> b 1')
            
            output = torch.cat((output, sampled), dim = -1)

        return output
 

    def forward_with_cond_scale(self, *args, cond_scale = 3, **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        texts: List[str] = None,
        sequences = None,
        output=None,
        text_mask = None,
    
        cond_drop_prob = None,
        return_loss = False
    ):
       
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        device = next(self.parameters()).device
        ########################## conditioning #################################### 
        
        cond_x=sequences.float().unsqueeze(2)

        # encoding 
        cond_x= self.fc1(cond_x)
        cond_x=self.GELUact(cond_x)  
        pos_fourier_xy=self.p_enc_1d(cond_x) 
        cond_x=cond_x+pos_fourier_xy
         
        if return_loss:
            labels  = output [:, 1:] #labels are non embedded
        
        output = self.token_embed(output.long())
        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           output.shape[1],
                                                          self.pos_fourier_graph_dim).to(device) )  
  
        output=output+pos_fourier_graph
        
        # enforce max text len

        if not exists(text_mask):
            text_mask = torch.ones(cond_x.shape[:2], dtype = torch.bool).to(device)
            
        cond_x, text_mask = map(lambda t: t[:, :self.max_text_len], (cond_x, text_mask))        

        # classifier free guidance conditional dropout
        batch=output.shape[0]
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        # attend
        x = self.init_norm(output)
        for self_attn, cross_attn, ff in self.layers:
            
            x = self_attn(x) + x
            x = cross_attn(x, context = cond_x, context_mask = text_mask) + x
            s_causal=False
            
            x = ff(x) + x

        x = self.final_norm(x)

        # to logits
        logits = self.to_logits(x)
         
        if not return_loss:
            return logits
        
        logits=logits[:,:-1,:]
       
        loss = F.cross_entropy(
            rearrange(logits , 'b n c -> b c n'),
            labels ,
            # ignore_index = 0
            )
        
        return loss

############## same but with flexible internal dimension (in previous: embed_dim is equal to dim)
class MoleculeTransformerSequenceInternaldim(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        max_tokens=32,
        logits_dim=32, 
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        embed_dim = 16, #for input sequence
        text_embed_dim = None, #for cnditining 
        cond_drop_prob = 0.25,
        max_text_len = 128,
        one_kv_head=True,
        ):
        super().__init__()

        self.embed_dim = embed_dim
        self.text_embed_dim =  text_embed_dim 
        self.pos_fourier_graph_dim=embed_dim

        #sequence to embedding    
        self.token_embed = nn.Embedding(max_tokens, embed_dim)
        #embedding to dim
        self.to_dim = nn.Linear( self.embed_dim, dim, bias = False)
            
        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last. This converts sequence conditioning to higher dimension
        self.GELUact= nn.GELU()
        ################
        
        self.p_enc_1d = PositionalEncoding1D(text_embed_dim)     #fourier encoding for conditioning sequence
        self.p_enc_1d_graph = PositionalEncoding1D(self.pos_fourier_graph_dim)     #fourier encoding for input
                        #(batch_size, x, ch)
       
        self.max_text_len = max_text_len

        assert cond_drop_prob > 0.
        self.cond_drop_prob = cond_drop_prob # classifier free guidance for transformers - @crowsonkb

        self.start_token = nn.Parameter(torch.randn(dim))

        # projecting to logits
        self.logits_dim=logits_dim
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                
                AttentionQKV(dim, causal = True, one_kv_head=one_kv_head,
                          dim_head = dim_head, heads = heads, dropout = dropout),
                AttentionQKV(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout, one_kv_head=one_kv_head,),
                
                FeedForward(dim, mult = ff_mult, dropout = dropout)
                
            ]))

        self.final_norm = LayerNorm(dim)

        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)
        

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        sequences=None,#conditioning
         text_mask=None,
        #texts,
        *,
        tokens_to_generate=32,
        cond_scale = 3.,
        filter_thres = 0.9,
        temperature = 1.,
        output=None, #should provide it with at least a start token
    ):
        device = next(self.parameters()).device

        batch = sequences.shape[0]

        image_seq_len=tokens_to_generate
        
        if output==None:
            output= torch.randint (0,self.logits_dim, (batch,  1), device = device  )
            print ("Since start token not provided, generating random token.")
        
        for _ in tqdm(range(image_seq_len)):
            
            
            sampled  = self.forward_with_cond_scale(
                
                    sequences = sequences,
                    text_mask = text_mask,
                    output = output,
                     
                )
            sampled=sampled[:, -1 ]
            
            filtered_logits = top_k(sampled, thres = filter_thres)
            sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
            sampled = rearrange(sampled, 'b -> b 1')
            
            output = torch.cat((output, sampled), dim = -1)

        return output


    def forward_with_cond_scale(self, *args, cond_scale = 3, **kwargs):
        logits = self.forward(*args, cond_drop_prob = 0., **kwargs)

        if cond_scale == 1:
            return logits

        null_logits = self.forward(*args, cond_drop_prob = 1., **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        texts: List[str] = None,
        sequences = None,
        output=None,
        text_mask = None,
    
        cond_drop_prob = None,
        return_loss = False
    ):
       
        cond_drop_prob = default(cond_drop_prob, self.cond_drop_prob)
        device = next(self.parameters()).device
        ########################## conditioning #################################### 
        
        cond_x=sequences.float().unsqueeze(2)

        # encoding 
        cond_x= self.fc1(cond_x)
        cond_x=self.GELUact(cond_x)  
        pos_fourier_xy=self.p_enc_1d(cond_x) 
        cond_x=cond_x+pos_fourier_xy
         
        if return_loss:
            labels  = output [:, 1:] #labels are non embedded
        
        output = self.token_embed(output.long())
        
        
        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           output.shape[1],
                                                          self.pos_fourier_graph_dim).to(device) )  
  
        output=output+pos_fourier_graph
               
        # enforce max text len

        if not exists(text_mask):
            text_mask = torch.ones(cond_x.shape[:2], dtype = torch.bool).to(device)
            
        cond_x, text_mask = map(lambda t: t[:, :self.max_text_len], (cond_x, text_mask))        

        # classifier free guidance conditional dropout
        batch=output.shape[0]
        if cond_drop_prob > 0:
            keep_mask = prob_mask_like((batch,), 1 - cond_drop_prob, device = device)
            text_mask = rearrange(keep_mask, 'b -> b 1') & text_mask

        x = self.to_dim (output)
        
        x = self.init_norm(x)
        for self_attn, cross_attn, ff in self.layers:
            x = self_attn(x) + x
            x = cross_attn(x, context = cond_x, context_mask = text_mask) + x
            x = ff(x) + x

        x = self.final_norm(x)

        # to logits
        logits = self.to_logits(x)
         
        if not return_loss:
            return logits
        
        logits=logits[:,:-1,:]
       
        loss = F.cross_entropy(
            rearrange(logits , 'b n c -> b c n'),
            labels ,
            # ignore_index = 0
            )
        
        return loss

############### transformer encoder ###################

from functools import partial, wraps
def maybe(fn):
    @wraps(fn)
    def inner(x, *args, **kwargs):
        if not exists(x):
            return x
        return fn(x, *args, **kwargs)
    return inner



    
#######################################################################

class MoleculeTransformerSequenceEncoder(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        logits_dim=32, 
        logits_dim_length = None , #if None entire length is used. Otherwise, max_length is projected to logits_dim_length
        max_length = None,         #see previous - OUTPUT IS (b, logits_dim, logits_dim_length)
        max_tokens = 32, #max tokens for embedding layer
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
      
        embed_dim = 16,
     
        padding_token=0,#will be used to create mask

    ):
        super().__init__()

      #  pos_fourier_graph_dim=dim
        self.padding_token=padding_token
     
        self.embed_dim=embed_dim #embedding dim of sequence
        self.max_length=max_length
        self.logits_dim_length=logits_dim_length

        self.GELUact= nn.GELU()
        ################
        
        self.p_enc_1d_graph = PositionalEncoding1D(self.embed_dim)     #fourier encoding for input
                        #(batch_size, x, ch)
       
        # projecting to logits
        self.logits_dim = logits_dim
        self.init_norm  = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                #AttentionQKV(dim, causal = False, 
                #          dim_head = dim_head, heads = heads, dropout = dropout),
                MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                #Attention(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout),
                FeedForward(dim, mult = ff_mult, dropout = dropout) 
            ]))

        self.final_norm = LayerNorm(dim)
         
        #sequence to embedding    
        self.token_embed = nn.Embedding(max_tokens, embed_dim)
        #embedding to dim
        self.to_dim = nn.Linear( self.embed_dim, dim, bias = False)
        #dim to logits
        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)
        
        if exists (self.logits_dim_length):
            assert exists (self.max_length), 'max_length and logits_dim_length must be set to nonzero value'
            self.to_logits_dim_length = nn.Linear(self.max_length, self.logits_dim_length, bias = False)

    def forward(
        self,
    
        input_sequence,
        text_mask = None,
        return_hidden = False,
        squeeze_output=False,
    
    ):
    
        device = next(self.parameters()).device
        ########################## conditioning #################################### 
       
        output = self.token_embed(input_sequence)   
        
        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           output.shape[1],
                                                           self.embed_dim).to(device) )  
  
        output=output+pos_fourier_graph
    
        x = self.to_dim (output)
    
        if exists (self.max_length):  # enforce max text len
            if not exists(text_mask):
                text_mask = torch.ones(x.shape[:2], dtype = torch.bool).to(device)
                text_mask = input_sequence.eq(self.padding_token) #use  0 will turn to True, others to False. All values with True are masked out.
            
        x, text_mask = map(lambda t: t[:, :self.max_length], (x, text_mask))     
       
        x = self.init_norm(x)
        #for self_attn, cross_attn, ff in self.layers:
        for self_attn, ff in self.layers:
            #x = self_attn(x, context_mask = text_mask) + x
            attout, _ = self_attn(x, x, x, key_padding_mask  = text_mask)
            x= attout + x
            
            #x = cross_attn(x, context = cond_x, context_mask = text_mask) + x
            x = ff(x) + x

        x = self.final_norm(x)

        # to logits
        logits = self.to_logits(x)
        
        if exists(self.logits_dim_length):
            logits=logits.permute (0,2,1)
            logits = self.to_logits_dim_length(logits)
            
         
        if return_hidden:
            
            return  x
    
        else:
            if squeeze_output:
                logits=logits.squeeze()
            
            return logits
        
        
######################### GPT with Graph, etc. #######################################

class AttentionQKV(nn.Module):
    def __init__(
        self,
        dim,
        *,
        context_dim = None,
        dim_head = 64,
        heads = 8,
        causal = False,
        dropout = 0.,
        norm_context = False,
        one_kv_head=True,
        use_null_kv=True,
        GNN_layers = 0,
        GNN_aggr = 'add',
        GNN_have_skip = True,
        GNN_att_threshold_min=0,#if >0 then set all values in att_matrix to 0 that are smaller
        GNN_att_threshold_max=1,#if <1 then set all values in att_matrix to 1 that are larger
        GNN_add_identity= True, #whether to add identity
        GNN_clamp_att_after_identity = True, #whether to clamp attention after identity is added
        
    ):
        super().__init__()
        self.causal = causal
        self.scale = dim_head ** -0.5
        self.norm = LayerNorm(dim)
        self.use_null_kv=use_null_kv
        self.GNN_layers=GNN_layers
        self.GNN_att_threshold_min=GNN_att_threshold_min
        self.GNN_att_threshold_max=GNN_att_threshold_max
        self.GNN_add_identity= GNN_add_identity
        self.heads=heads
        self.GNN_clamp_att_after_identity=GNN_clamp_att_after_identity

        context_dim = default(context_dim, dim)
        self.norm_context = LayerNorm(context_dim) if norm_context else nn.Identity()
        
        self.one_kv_head = one_kv_head #If true: Use one KV head, multiple query heads

        # needed for classifier free guidance for transformers
        # by @crowsonkb, adopted by the paper
        
        q_dim = k_dim = dim_head * heads
        v_dim = out_dim = dim_head * heads

        # one-headed key / value attention, from Shazeer's multi-query paper, adopted by Alphacode and PaLM
        # https://arxiv.org/abs/1911.02150 - only have one head for the key / values, multi-headed queries
        self.one_kv_head = one_kv_head
        if one_kv_head:
            k_dim = dim_head
            v_dim = dim_head
            out_dim = v_dim * heads

        self.null_k = nn.Parameter(torch.randn(k_dim))
        self.null_v = nn.Parameter(torch.randn(v_dim))
        
        self.to_q = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, q_dim, bias = False),
            Rearrange('b n (h d) -> b h n d', h = heads)
        )
        self.to_k = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, k_dim, bias = False)
        )
        self.to_v = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(context_dim, v_dim, bias = False)
        )

        self.to_out = nn.Sequential(
            Rearrange('b h n d -> b n (h d)'),
          
            nn.Linear(out_dim, dim, bias = False),
            LayerNorm(dim)
        )
        
         
        if self.GNN_layers>0:
            self.GNN_layers=GNN_layers
            #GNN_hidden_channels=dim
            #print ("q_dim ",q_dim) 
            self.GNN_net = GraphConvLayers(node_features_in=dim_head,#q_dim, 
                                    num_node_features_out=dim_head,#,q_dim, 
                                    hidden_channels=dim_head,#q_dim,
                                    depth=GNN_layers,
                                   # aggr = GNN_aggr,
                                    have_skip = GNN_have_skip,
                                   )



    def forward(
        self,
        x,
        context = None,
        context_mask = None,
        plot_attn= False,
        
        
    ):
        batch, device = x.shape[0], x.device

        x = self.norm(x)

        q = self.to_q(x) * self.scale

        context = default(context, x)
        context = self.norm_context(context)

        k = self.to_k(context)
        v = self.to_v(context)

        if self.use_null_kv:
            null_k = repeat(self.null_k, 'd -> b 1 d', b = batch)
            k = torch.cat((null_k, k), dim = 1)
            null_v = repeat(self.null_v, 'd -> b 1 d', b = batch)
            v = torch.cat((null_v, v), dim = 1)

        
        if not self.one_kv_head:
            k, v = map(lambda t: maybe(rearrange)(t, 'b n (h d) -> b h n d', h = self.heads), (k, v))
            
        kv_einsum_eq = 'b h j d' if not self.one_kv_head else 'b j d'

        #dots = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) * scale

        sim = einsum(f'b h i d, {kv_einsum_eq} -> b h i j', q, k) #* scale

        mask_value = -torch.finfo(sim.dtype).max

        if exists(context_mask):
            if self.use_null_kv:
                context_mask = F.pad(context_mask, (1, 0), value = True)
            
            #original
            #context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            
            if context_mask.ndim == 2:#original ... 
                #attn_mask = rearrange(attn_mask, 'i j -> 1 1 i j')
                context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            #elif context_mask.ndim == 3:#not sure if works...
            #    #attn_mask = rearrange(attn_mask, 'h i j -> 1 h i j')
            #    context_mask = rearrange(context_mask, 'b j h -> b 1 j h')
            #context_mask = rearrange(context_mask, 'b j -> b 1 1 j')
            
            sim = sim.masked_fill(~context_mask, mask_value)
            #dots = dots.masked_fill(~input_mask, mask_value)
            
        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)
            
            #causal_mask = torch.ones((i, j), dtype = torch.bool, device = device).triu(j - i + 1)
            #dots = dots.masked_fill(causal_mask, mask_value)            

        attn = sim.softmax(dim = -1, dtype = torch.float32)
        
        out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)
        
        ######### Graph neural net ###############################
        #GNN layers are added after causal mask is applied to maintain causality
        #GNN is constructed by using attn as adjancy matrix and/or edge features, shape is d,d
        #    node features are defined by v here: shape is n,d
        #    attn can be processed with min/max threshold, identity added, etc.
        #    TO DO: implement other aggregation schemes, multi hop message passing....
        if self.GNN_layers>0:
            #print ("out shape" , out.shape)
            #print ("Att bef shape: ", attn.shape)
            # attn=  rearrange (attn,'b h n d -> b n (h d)')
            attn_comb=  rearrange (attn,'b h n d -> (b h) n d') #these are the attention matrices - batch and head grouped
            if self.GNN_add_identity:
                attn_comb = attn_comb + torch.eye(attn_comb.shape[1]).unsqueeze(0).repeat(attn_comb.shape[0],1,1).to(device)
                if self.GNN_clamp_att_after_identity:
                    attn_comb = torch.clamp(attn_comb, min=0, max=1)

                #attn_comb =attn_comb.softmax (dim=-1, dtype = torch.float32)
                
            if self.GNN_att_threshold_min>0:
                attn_comb[attn_comb<self.GNN_att_threshold_min] = 0
            if self.GNN_att_threshold_max<1:
                attn_comb[attn_comb>self.GNN_att_threshold_max] = 1
                
            if plot_attn:
                grid_disc=int (attn_comb.shape [0]**0.5)
                fig, axes = plt.subplots(nrows=grid_disc, ncols=grid_disc, figsize=(8,8))
                c_i=0
                for i in range (grid_disc):
                    for j in range (grid_disc):
                        axes[i,j].imshow (attn_comb[c_i,:].detach().numpy() )
                        #axes[i,j].colorbar()
                        c_i+=1

                # fig.colorbar (ax=fig.get_axes())
                plt.show()
            
            #v_comb=  rearrange (v,'b h n d -> b n (h d)')
            v_comb=  rearrange (v,'b h n d -> (b h) n d') #these are the node embeddings - batch and head grouped
            
            #print ("v_comb.shape",v_comb.shape, "att_comb shape shape: ", attn_comb.shape)
            #print (edge_index.shape, edge_weight.shape)

            #data=torch.randn (4, 16, dim) #b, n, c
            out_from_GNN = self.GNN_net(v_comb, attn_comb,) 
            
            
            
            #print (out_from_GNN.shape)
            #attn_comb=  rearrange (attn,'(b h) n d -> b h n d')
            out = out +  rearrange (out_from_GNN,'(b h) n d -> b h n d', h = self.heads)
            
            
            #print (out_from_GNN.shape, attn.shape)

        
        #out = einsum(f'b h i j, {kv_einsum_eq} -> b h i d', attn, v)
        

        return self.to_out(out)
    
#######################################################################

class MoleculeTransformerGPT(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        max_tokens=32,
        logits_dim=32, 
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        embed_dim = 16, #for input sequence
        text_embed_dim = 16,# None, #not used 
        max_text_len = 128,
        one_kv_head=True,
        concat_pos_encoding = False, #if True, then pos encoding will be added to the input; and pos_fourier_graph_dim=embed_dim
        pos_fourier_graph_dim  = None,  #only used if concat_pos_encoding == True
        use_null_kv=True, #use null_kv for CFG
        FF_conv_kernel=0, #if >0 then use conv in FF layer, at beginning and end
        FF_inner_conv_kernel=0,
        FF_glu=False,
        GNN_layers = 0, #whether or not to use GNN_layers after softmax
        GNN_att_threshold_min=0.,
        GNN_att_threshold_max=1.,
        
    ):
        super().__init__()

        # pos_fourier_graph_dim=embed_dim
        self.embed_dim = embed_dim
        self.concat_pos_encoding=concat_pos_encoding
        self.use_null_kv=use_null_kv
        #  self.text_embed_dim =  text_embed_dim 
        if concat_pos_encoding == False:
            self.pos_fourier_graph_dim=embed_dim
        else:
            self.pos_fourier_graph_dim=pos_fourier_graph_dim
            assert pos_fourier_graph_dim !=None, "pos_fourier_graph_dim has to be set if concatenating pos embedding"
            print ("Concatenate positional encoding...")

        #sequence to embedding    
        self.token_embed = nn.Embedding(max_tokens, embed_dim)
        #embedding to dim
        
        dim_in = self.embed_dim + self.concat_pos_encoding*self.pos_fourier_graph_dim
        self.to_dim = nn.Linear(dim_in, 
                                dim, bias = False)
        print ("Input dimension (signal and pos encoding): ", dim_in)
        
            
        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last. This converts sequence conditioning to higher dimension -- NOT USED, TODO: Remove
        self.GELUact= nn.GELU()
        ################
        self.p_enc_1d_graph = PositionalEncoding1D(self.pos_fourier_graph_dim)     #fourier encoding for input
                        #(batch_size, x, ch)
       
        self.max_text_len = max_text_len

         # projecting to logits
        self.logits_dim=logits_dim
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                
                AttentionQKV(dim, causal = True, one_kv_head=one_kv_head,
                          dim_head = dim_head, heads = heads, dropout = dropout, 
                            use_null_kv=self.use_null_kv, 
                            GNN_layers=GNN_layers, 
                             GNN_att_threshold_min=GNN_att_threshold_min,
                             GNN_att_threshold_max=GNN_att_threshold_max,
                            ),
                #AttentionQKV(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout, one_kv_head=one_kv_head,),
                #FeedForward(dim, mult = ff_mult, dropout = dropout)
                FeedForward(dim, mult = ff_mult, dropout = dropout) if (conv_kernel_FF==0 and FF_inner_conv==0) else FeedForward_CNN (dim,  mult = ff_mult, dropout = dropout, conv_kernel_FF=FF_conv_kernel, FF_inner_conv=FF_inner_conv_kernel, glu=FF_glu  )               
            ]))

        self.final_norm = LayerNorm(dim)
        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        *,
        output=None, #should provide it with at least a start token
        tokens_to_generate=32,
        filter_thres = 0.9,
        temperature = 1.,
       
        use_gumbel_sample=True, #whether or not ot use gumbel sampling 
    ):
        device = next(self.parameters()).device

        batch = output.shape[0]

        image_seq_len=tokens_to_generate
        
        if output==None:
            output= torch.randint (0,self.logits_dim, (batch,  1), device = device  )
            print ("Since start token not provided, generating random token.")
        
        for _ in tqdm(range(image_seq_len)):
            
            sampled  = self.forward(
                output = output,
                )
            sampled=sampled[:, -1 ]
            
            if use_gumbel_sample:
                filtered_logits = top_k(sampled, thres = filter_thres)
                sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
                sampled = rearrange(sampled, 'b -> b 1')
            else:
                sampled = sampled.argmax(dim = -1)
                sampled = rearrange(sampled, 'b -> b 1')
                
            output = torch.cat((output, sampled), dim = -1)

        return output
 
    def forward(
        self,
        output=None,
        return_loss = False,
        ignore_padding_zeros=False,
        mask_prob=0.,# if > 0: Apply BERT style LLM mask
        context_mask = None,
        
    ):
       
        device = next(self.parameters()).device
        
        if return_loss:
            labels  = output [:, 1:] #labels are non embedded
        
        output = self.token_embed(output.long())
        
        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           output.shape[1],
                                                          self.pos_fourier_graph_dim).to(device) )  
                                    #(batch_size, x, ch)
            
            
        if self.concat_pos_encoding == False: #if False then additive ... 
            output=output+pos_fourier_graph
        else:
            output= torch.cat((output, pos_fourier_graph), dim = -1)  
            
        batch=output.shape[0]
       
        x = self.to_dim (output)
        
        x = self.init_norm(x)
       
        
        #Apply BERT style LLM mask if selected
        
        if mask_prob > 0.:
            rand = torch.randn( (output.shape[0], x.shape[1]), device = x.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max # first token should not be masked out
            
            num_mask = min(int(output.shape[1] * mask_prob), output.shape[1] - 1)
            
            indices = rand.topk(num_mask, dim = -1).indices #torch.topk
                                    #Returns the k largest elements of the given 
                                    #input tensor along a given dimension
                    
            context_mask = ~torch.zeros_like(output[:,:,0]).squeeze().scatter(1, indices, 1.).bool()
                            #torch.scatter(input, dim, index, src) 
                            #the ones selected by random via indices, i.e. 1s, are going to be False
            
            #now repeat in channel direction
            #context_mask = repeat(context_mask, 'b n -> b n c', c = output.shape[2])
            
            #sim = sim.masked_fill(~context_mask, mask_value)
            
            #kwargs.update(self_attn_context_mask = mask)  
            
        
        
        
        for self_attn, ff in self.layers:
            x = self_attn(x, context_mask=context_mask) + x
            x = ff(x) + x
        x = self.final_norm(x)

        # to logits
        logits = self.to_logits(x)
         
        if not return_loss:
            return logits
        
        logits=logits[:,:-1,:]
       
        if ignore_padding_zeros:
            loss = F.cross_entropy(
                rearrange(logits , 'b n c -> b c n'),
                labels ,
                ignore_index = 0,
                )            
        else:
            
            loss = F.cross_entropy(
                rearrange(logits , 'b n c -> b c n'),
                labels ,
                #ignore_index = 0
                )

        return loss

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_edge_index,dense_to_sparse

class GCNLayer(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True) #used for normalization 
        node_feats = self.projection(node_feats) #Apply linear layer (filter) to each of the node features (same weights!)
        node_feats = torch.bmm(adj_matrix, node_feats) #Add up all the node features  using the adj matrix
        node_feats = node_feats / num_neighbours #normalize
        return node_feats
    
class GraphConvLayers(torch.nn.Module):
    def __init__(self, 
                 node_features_in, num_node_features_out, hidden_channels,
                 depth,
              #   aggr = 'add',
                 have_skip = False,
                ):
        super(GraphConvLayers, self).__init__()
        #torch.manual_seed(12345)
        #assert node_features_in == hidden_channels if have_skip, 'If use skip connection, must have node_features_in== hidden_channels'
        self.layers = []
        
        self.have_skip = have_skip
        
        for i in range(depth):
            self.layers.append(
                
#                GraphConv(hidden_channels, hidden_channels, aggr=aggr) if i>0 else GraphConv(node_features_in, hidden_channels, aggr=aggr)
                GCNLayer(hidden_channels, hidden_channels, ) if i>0 else GCNLayer(node_features_in, hidden_channels, )
            )
           
        self.layers = nn.ModuleList(self.layers)
        
    
        self.lin = Linear(hidden_channels, num_node_features_out)
        self.GELUact= nn.GELU()
        
        #self.mlp = Sequential(Linear(hidden_channels, 30), ReLU(), Linear(30, 15), ReLU(), Linear(15, dataset.num_classes))

    def forward(self, x, adj_matrix):
        
        #forward(x: Union[Tensor, Tuple[Tensor, Optional[Tensor]]], 
        #edge_index: Union[Tensor, SparseTensor], edge_weight: Optional[Tensor] = None, size: Optional[Tuple[int, int]] = None)→ Tensor
        
        for GNNlayer in self.layers:
            #x = GNNlayer(x, edge_index, edge_weight) + x*self.have_skip
            x = GNNlayer(x, adj_matrix) + x*self.have_skip
            x = self.GELUact(x)
         
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        
        
        return x    
######################### PyTorch implementation of GPT #########################
#There may be a bug in the MHA module in PyTorch (needs to be tested)

class MoleculeTransformerGPTPyTorch(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        max_tokens=32,
        logits_dim=32, 
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        embed_dim = 16, #for input sequence
        text_embed_dim = None, #not used 
        max_text_len = 128,
        one_kv_head=True,
        concat_pos_encoding = False, #if True, then pos encoding will be added to the input; and pos_fourier_graph_dim=embed_dim
        pos_fourier_graph_dim  = None,  #only used if concat_pos_encoding == True
    ):
        super().__init__()

       # pos_fourier_graph_dim=embed_dim
        self.embed_dim = embed_dim
        self.concat_pos_encoding=concat_pos_encoding
        #  self.text_embed_dim =  text_embed_dim 
        if concat_pos_encoding == False:
            self.pos_fourier_graph_dim=embed_dim
        else:
            self.pos_fourier_graph_dim=pos_fourier_graph_dim
            assert pos_fourier_graph_dim !=None, "pos_fourier_graph_dim has to be set if concatenating pos embedding"
            print ("Concatenate positional encoding...")

        #sequence to embedding    
        self.token_embed = nn.Embedding(max_tokens, embed_dim)
        #embedding to dim
        
        dim_in = self.embed_dim + self.concat_pos_encoding*self.pos_fourier_graph_dim
        self.to_dim = nn.Linear(dim_in, 
                                dim, bias = False)
        print ("Input dimension (signal and pos encoding): ", dim_in)
        
            
        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last. This converts sequence conditioning to higher dimension
        self.GELUact= nn.GELU()
        ################
        self.p_enc_1d_graph = PositionalEncoding1D(self.pos_fourier_graph_dim)     #fourier encoding for input
                        #(batch_size, x, ch)
       
        self.max_text_len = max_text_len

         # projecting to logits
        self.logits_dim=logits_dim
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                
                #AttentionQKV(dim, causal = True, one_kv_head=one_kv_head,
                #          dim_head = dim_head, heads = heads, dropout = dropout),
                #AttentionQKV(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout, one_kv_head=one_kv_head,),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
                
            ]))

        self.final_norm = LayerNorm(dim)
        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        *,
        tokens_to_generate=32,
        filter_thres = 0.9,
        temperature = 1.,
        output=None, #should provide it with at least a start token
        use_gumbel_sample=True, #whether or not ot use gumbel sampling 
    ):
        device = next(self.parameters()).device

        batch = output.shape[0]

        image_seq_len=tokens_to_generate
        
        if output==None:
            output= torch.randint (0,self.logits_dim, (batch,  1), device = device  )
            print ("Since start token not provided, generating random token.")
        
        for _ in tqdm(range(image_seq_len)):
            
            sampled  = self.forward(
                output = output,
                )
            sampled=sampled[:, -1 ]
            
            if use_gumbel_sample:
                filtered_logits = top_k(sampled, thres = filter_thres)
                sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
                sampled = rearrange(sampled, 'b -> b 1')
            else:
                sampled = sampled.argmax(dim = -1)
                sampled = rearrange(sampled, 'b -> b 1')
                
            output = torch.cat((output, sampled), dim = -1)

        return output
 
    def forward(
        self,
        output=None,
        return_loss = False,
        ignore_padding_zeros=False,
    ):
       
        device = next(self.parameters()).device
        
        if return_loss:
            labels  = output [:, 1:] #labels are non embedded
        
        output = self.token_embed(output.long())
        
        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           output.shape[1],
                                                          self.pos_fourier_graph_dim).to(device) )  
                                    #(batch_size, x, ch)
            
            
        if self.concat_pos_encoding == False: #if False then additive ... 
            output=output+pos_fourier_graph
        else:
            output= torch.cat((output, pos_fourier_graph), dim = -1)  
            
        batch=output.shape[0]
       
        x = self.to_dim (output)
        
        x = self.init_norm(x)
       
        for self_attn, ff in self.layers:
            #x = self_attn(x, x, x, ) + x
            #attout, _ = self_attn(x, x, x, is_causal = True )
            attout  = self_attn(x, x, x, is_causal = True, need_weights = False)[0]
            x= attout + x
            
            x = ff(x) + x
        x = self.final_norm(x)

        # to logits
        logits = self.to_logits(x)
         
        if not return_loss:
            return logits
        
        logits=logits[:,:-1,:]
       
        if ignore_padding_zeros:
            loss = F.cross_entropy(
                rearrange(logits , 'b n c -> b c n'),
                labels ,
                ignore_index = 0,
                )            
        else:
            
            loss = F.cross_entropy(
                rearrange(logits , 'b n c -> b c n'),
                labels ,
                #ignore_index = 0
                )

        return loss
    
    
##
       
        if ignore_padding_zeros:
            loss = F.cross_entropy(
                rearrange(logits , 'b n c -> b c n'),
                labels ,
                ignore_index = 0,
                )            
        else:
            
            loss = F.cross_entropy(
                rearrange(logits , 'b n c -> b c n'),
                labels ,
                #ignore_index = 0
                )

        return loss
    
    
##
        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        *,
        output=None, #should provide it with at least a start token
        tokens_to_generate=32,
        filter_thres = 0.9,
        temperature = 1.,
       
        use_gumbel_sample=True, #whether or not ot use gumbel sampling 
    ):
        device = next(self.parameters()).device

        batch = output.shape[0]

        image_seq_len=tokens_to_generate
        
        if output==None:
            output= torch.randint (0,self.logits_dim, (batch,  1), device = device  )
            print ("Since start token not provided, generating random token.")
        
        for _ in tqdm(range(image_seq_len)):
            
            sampled  = self.forward(
                output = output,
                )
            sampled=sampled[:, -1 ]
            
            if use_gumbel_sample:
                filtered_logits = top_k(sampled, thres = filter_thres)
                sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
                sampled = rearrange(sampled, 'b -> b 1')
            else:
                sampled = sampled.argmax(dim = -1)
                sampled = rearrange(sampled, 'b -> b 1')
                
            output = torch.cat((output, sampled), dim = -1)

        return output
 
    def forward(
        self,
        output=None,
        return_loss = False,
        ignore_padding_zeros=False,
        mask_prob=0.,# if > 0: Apply BERT style LLM mask
        context_mask = None,
        
    ):
       
        device = next(self.parameters()).device
        
        if return_loss:
            labels  = output [:, 1:] #labels are non embedded
        
        output = self.token_embed(output.long())
        
        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           output.shape[1],
                                                          self.pos_fourier_graph_dim).to(device) )  
                                    #(batch_size, x, ch)
            
            
        if self.concat_pos_encoding == False: #if False then additive ... 
            output=output+pos_fourier_graph
        else:
            output= torch.cat((output, pos_fourier_graph), dim = -1)  
            
        batch=output.shape[0]
       
        x = self.to_dim (output)
        
        x = self.init_norm(x)
       
        
        #Apply BERT style LLM mask if selected
        
        if mask_prob > 0.:
            rand = torch.randn( (output.shape[0], x.shape[1]), device = x.device)
            rand[:, 0] = -torch.finfo(rand.dtype).max # first token should not be masked out
            
            num_mask = min(int(output.shape[1] * mask_prob), output.shape[1] - 1)
            
            indices = rand.topk(num_mask, dim = -1).indices #torch.topk
                                    #Returns the k largest elements of the given 
                                    #input tensor along a given dimension
                    
            context_mask = ~torch.zeros_like(output[:,:,0]).squeeze().scatter(1, indices, 1.).bool()
                            #torch.scatter(input, dim, index, src) 
                            #the ones selected by random via indices, i.e. 1s, are going to be False
            
            #now repeat in channel direction
            #context_mask = repeat(context_mask, 'b n -> b n c', c = output.shape[2])
            
            #sim = sim.masked_fill(~context_mask, mask_value)
            
            #kwargs.update(self_attn_context_mask = mask)  
            
        
        
        
        for self_attn, ff in self.layers:
            x = self_attn(x, context_mask=context_mask) + x
            x = ff(x) + x
        x = self.final_norm(x)

        # to logits
        logits = self.to_logits(x)
         
        if not return_loss:
            return logits
        
        logits=logits[:,:-1,:]
       
        if ignore_padding_zeros:
            loss = F.cross_entropy(
                rearrange(logits , 'b n c -> b c n'),
                labels ,
                ignore_index = 0,
                )            
        else:
            
            loss = F.cross_entropy(
                rearrange(logits , 'b n c -> b c n'),
                labels ,
                #ignore_index = 0
                )

        return loss

from torch.nn import Linear
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,GraphConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.utils import to_edge_index,dense_to_sparse

class GCNLayer(nn.Module):

    def __init__(self, c_in, c_out):
        super().__init__()
        self.projection = nn.Linear(c_in, c_out)

    def forward(self, node_feats, adj_matrix):
        """
        Inputs:
            node_feats - Tensor with node features of shape [batch_size, num_nodes, c_in]
            adj_matrix - Batch of adjacency matrices of the graph. If there is an edge from i to j, adj_matrix[b,i,j]=1 else 0.
                         Supports directed edges by non-symmetric matrices. Assumes to already have added the identity connections.
                         Shape: [batch_size, num_nodes, num_nodes]
        """
        # Num neighbours = number of incoming edges
        num_neighbours = adj_matrix.sum(dim=-1, keepdims=True) #used for normalization 
        node_feats = self.projection(node_feats) #Apply linear layer (filter) to each of the node features (same weights!)
        node_feats = torch.bmm(adj_matrix, node_feats) #Add up all the node features  using the adj matrix
        node_feats = node_feats / num_neighbours #normalize
        return node_feats
    
class GraphConvLayers(torch.nn.Module):
    def __init__(self, 
                 node_features_in, num_node_features_out, hidden_channels,
                 depth,
              #   aggr = 'add',
                 have_skip = False,
                ):
        super(GraphConvLayers, self).__init__()
        #torch.manual_seed(12345)
        #assert node_features_in == hidden_channels if have_skip, 'If use skip connection, must have node_features_in== hidden_channels'
        self.layers = []
        
        self.have_skip = have_skip
        
        for i in range(depth):
            self.layers.append(
                
#                GraphConv(hidden_channels, hidden_channels, aggr=aggr) if i>0 else GraphConv(node_features_in, hidden_channels, aggr=aggr)
                GCNLayer(hidden_channels, hidden_channels, ) if i>0 else GCNLayer(node_features_in, hidden_channels, )
            )
           
        self.layers = nn.ModuleList(self.layers)
        
    
        self.lin = Linear(hidden_channels, num_node_features_out)
        self.GELUact= nn.GELU()
        
        #self.mlp = Sequential(Linear(hidden_channels, 30), ReLU(), Linear(30, 15), ReLU(), Linear(15, dataset.num_classes))

    def forward(self, x, adj_matrix):
        
        #forward(x: Union[Tensor, Tuple[Tensor, Optional[Tensor]]], 
        #edge_index: Union[Tensor, SparseTensor], edge_weight: Optional[Tensor] = None, size: Optional[Tuple[int, int]] = None)→ Tensor
        
        for GNNlayer in self.layers:
            #x = GNNlayer(x, edge_index, edge_weight) + x*self.have_skip
            x = GNNlayer(x, adj_matrix) + x*self.have_skip
            x = self.GELUact(x)
         
        x = F.dropout(x, p=0.1, training=self.training)
        x = self.lin(x)
        
        
        
        return x    
######################### PyTorch implementation of GPT #########################
#There may be a bug in the MHA module in PyTorch (needs to be tested)

class MoleculeTransformerGPTPyTorch(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        max_tokens=32,
        logits_dim=32, 
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        ff_mult = 4,
        embed_dim = 16, #for input sequence
        text_embed_dim = None, #not used 
        max_text_len = 128,
        one_kv_head=True,
        concat_pos_encoding = False, #if True, then pos encoding will be added to the input; and pos_fourier_graph_dim=embed_dim
        pos_fourier_graph_dim  = None,  #only used if concat_pos_encoding == True
    ):
        super().__init__()

       # pos_fourier_graph_dim=embed_dim
        self.embed_dim = embed_dim
        self.concat_pos_encoding=concat_pos_encoding
        #  self.text_embed_dim =  text_embed_dim 
        if concat_pos_encoding == False:
            self.pos_fourier_graph_dim=embed_dim
        else:
            self.pos_fourier_graph_dim=pos_fourier_graph_dim
            assert pos_fourier_graph_dim !=None, "pos_fourier_graph_dim has to be set if concatenating pos embedding"
            print ("Concatenate positional encoding...")

        #sequence to embedding    
        self.token_embed = nn.Embedding(max_tokens, embed_dim)
        #embedding to dim
        
        dim_in = self.embed_dim + self.concat_pos_encoding*self.pos_fourier_graph_dim
        self.to_dim = nn.Linear(dim_in, 
                                dim, bias = False)
        print ("Input dimension (signal and pos encoding): ", dim_in)
        
            
        self.fc1 = nn.Linear( 1,  text_embed_dim)  # INPUT DIM (last), OUTPUT DIM, last. This converts sequence conditioning to higher dimension
        self.GELUact= nn.GELU()
        ################
        self.p_enc_1d_graph = PositionalEncoding1D(self.pos_fourier_graph_dim)     #fourier encoding for input
                        #(batch_size, x, ch)
       
        self.max_text_len = max_text_len

         # projecting to logits
        self.logits_dim=logits_dim
        self.init_norm = LayerNorm(dim)

        self.layers = nn.ModuleList([])

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                
                #AttentionQKV(dim, causal = True, one_kv_head=one_kv_head,
                #          dim_head = dim_head, heads = heads, dropout = dropout),
                #AttentionQKV(dim, context_dim = text_embed_dim, dim_head = dim_head, heads = heads, dropout = dropout, one_kv_head=one_kv_head,),
                FeedForward(dim, mult = ff_mult, dropout = dropout)
                
            ]))

        self.final_norm = LayerNorm(dim)
        self.to_logits = nn.Linear(dim, self.logits_dim, bias = False)

    @torch.no_grad()
    @eval_decorator
    def generate(
        self,
        *,
        tokens_to_generate=32,
        filter_thres = 0.9,
        temperature = 1.,
        output=None, #should provide it with at least a start token
        use_gumbel_sample=True, #whether or not ot use gumbel sampling 
    ):
        device = next(self.parameters()).device

        batch = output.shape[0]

        image_seq_len=tokens_to_generate
        
        if output==None:
            output= torch.randint (0,self.logits_dim, (batch,  1), device = device  )
            print ("Since start token not provided, generating random token.")
        
        for _ in tqdm(range(image_seq_len)):
            
            sampled  = self.forward(
                output = output,
                )
            sampled=sampled[:, -1 ]
            
            if use_gumbel_sample:
                filtered_logits = top_k(sampled, thres = filter_thres)
                sampled = gumbel_sample(filtered_logits, temperature = temperature, dim = -1)
                sampled = rearrange(sampled, 'b -> b 1')
            else:
                sampled = sampled.argmax(dim = -1)
                sampled = rearrange(sampled, 'b -> b 1')
                
            output = torch.cat((output, sampled), dim = -1)

        return output
 
    def forward(
        self,
        output=None,
        return_loss = False,
        ignore_padding_zeros=False,
    ):
       
        device = next(self.parameters()).device
        
        if return_loss:
            labels  = output [:, 1:] #labels are non embedded
        
        output = self.token_embed(output.long())
        
        pos_fourier_graph=self.p_enc_1d_graph( torch.ones (output.shape[0],
                                                           output.shape[1],
                                                          self.pos_fourier_graph_dim).to(device) )  
                                    #(batch_size, x, ch)
            
            
        if self.concat_pos_encoding == False: #if False then additive ... 
            output=output+pos_fourier_graph
        else:
            output= torch.cat((output, pos_fourier_graph), dim = -1)  
            
        batch=output.shape[0]
       
        x = self.to_dim (output)
        
        x = self.init_norm(x)
       
        for self_attn, ff in self.layers:
            #x = self_attn(x, x, x, ) + x
            #attout, _ = self_attn(x, x, x, is_causal = True )
            attout  = self_attn(x, x, x, is_causal = True, need_weights = False)[0]
            x= attout + x
            
            x = ff(x) + x
        x = self.final_norm(x)

        # to logits
        logits = self.to_logits(x)
         
        if not return_loss:
            return logits
        
        logits=logits[:,:-1,:]
       
        if ignore_padding_zeros:
            loss = F.cross_entropy(
                rearrange(logits , 'b n c -> b c n'),
                labels ,
                ignore_index = 0,
                )            
        else:
            
            loss = F.cross_entropy(
                rearrange(logits , 'b n c -> b c n'),
                labels ,
                #ignore_index = 0
                )

        return loss
    
    
##