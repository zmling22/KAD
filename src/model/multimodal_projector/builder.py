import re
import math
from torch import nn
from typing import Optional
from torch import Tensor
import torch
from functools import partial
from timm.layers.norm_act import LayerNormAct2d
from torchvision.ops.misc import SqueezeExcitation as SELayer
from torchvision.models.mobilenetv3 import InvertedResidual, InvertedResidualConfig
import torch.nn.functional as F

class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class Minigpt(nn.Module):
    def __init__(self, config=None):
        super(Minigpt, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x):
        # x is the input tensor with shape [b, num_tokens, c]
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")

        # Reshape x to [b, num_tokens/4, c*4]
        x = x.view(b, num_tokens // 4, c * 4)

        # Apply the linear transformation
        x = self.linear(x)
        return x


class Vanilla(nn.Module):
    def __init__(self, config=None):
        super(Vanilla, self).__init__()
        # c*4 is the input size, and c is the output size for the linear layer
        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear = nn.Linear(inc * 4, ouc)

    def forward(self, x):
        b, num_tokens, c = x.shape

        # Check if num_tokens is divisible by 4
        if num_tokens % 4 != 0:
            raise ValueError("num_tokens must be divisible by 4")

        # First, reshape to [b, num_tokens//4, 4, c]
        x = x.view(b, num_tokens // 4, 4, c)

        # Then, permute to interleave the tokens
        x = x.permute(0, 1, 3, 2).contiguous()

        # Finally, reshape to [b, num_tokens//4, c*4] to interleave features of 4 tokens
        x = x.view(b, num_tokens // 4, c * 4)

        # Apply the linear transformation
        x = self.linear(x)
        return x


class LDPBlock(nn.Module):
    # Lightweight Downsample Projector Block

    def __init__(self, config=None):
        super().__init__()

        inc, ouc = config.mm_hidden_size, config.hidden_size
        layer_norm = partial(LayerNormAct2d, act_layer=None)
        se_layer = partial(SELayer, scale_activation=nn.Hardsigmoid)
        self.mlp = nn.Sequential(
            nn.Identity(), nn.Linear(inc, ouc), nn.GELU(), nn.Linear(ouc, ouc)
        )
        self.mb_block = nn.Sequential(
            nn.Identity(),
            InvertedResidual(InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 1, 1, 1), layer_norm, se_layer),
            InvertedResidual(InvertedResidualConfig(ouc, 3, ouc, ouc, True, "HS", 2, 1, 1), layer_norm, se_layer)
        )

    def forward(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        x = self.mlp(x)
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.mb_block(x)
        x = x.flatten(2).permute(0, 2, 1)
        return x


class LDPNetProjector(nn.Module):

    def __init__(self, config=None):
        super().__init__()
        self.model = LDPBlock(config)

    def forward(self, x):
        return self.model(x)


class CA_interactor(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.mm_dim = config.mm_hidden_size
        self.hidden_dim = config.hidden_size
        ca_layer = TransformerDecoderLayer_(self.hidden_dim, nhead=8)
        self.ca_decoder = nn.TransformerDecoder(ca_layer, num_layers=2)

    def forward(self, x):
        # x shape is [num_images, topk_feature, dim]
        num, topk, dim = x.shape
        output = []
        for idx in range(num):
            q = x[idx].unsqueeze(0)
            kv_indices = [j for j in range(num) if j != idx]
            kv = x[kv_indices].view(-1, dim).unsqueeze(0)

            output.append(self.ca_decoder(q, kv).squeeze(0))
        output = torch.stack(output, dim=0)
        return output

    def forward_outer(self, q, kv):
        num, topk, dim = q.shape
        kv = kv.view(-1, dim).unsqueeze(0)
        q = q.view(-1, dim).unsqueeze(0)
        return self.ca_decoder(q, kv)


class CA_adapter(nn.Module):
    def __init__(self, config=None, 
                num_queries = 90):
        super().__init__()
        self.num_queries = num_queries
        self.mm_dim = config.mm_hidden_size
        self.hidden_dim = config.hidden_size
        self.query = nn.Parameter(torch.zeros(self.num_queries, self.hidden_dim))
        self.linear_0 = \
            nn.Sequential(
                nn.Linear(self.mm_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                )
        self.ca_layer = TransformerDecoderLayer_(self.hidden_dim, nhead=8)
        self.ca_decoder = nn.TransformerDecoder(self.ca_layer, num_layers=2)
        self.linear_1 = \
            nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                )

    def forward(self, x):
        x = self.linear_0(x)
        batch_size = x.shape[0]
        return self.linear_1(
            self.ca_decoder(
            self.query.unsqueeze(0).repeat(batch_size, 1, 1), 
            x))


class SPP(nn.Module):

    def __init__(self, config=None, projector_type='v1'):
        super().__init__()

        self.projector_type = projector_type

        inc, ouc = config.mm_hidden_size, config.hidden_size
        self.linear_0 = nn.Linear(inc, inc)

        self.linear_1 = nn.Linear(inc, ouc)

        self.pooling = nn.AvgPool2d(kernel_size=2)

        self.linear_2 = nn.Linear(ouc, ouc)

    def forward(self, x):
        b, num_tokens, c = x.shape
        h = int(math.sqrt(num_tokens))
        if 'v1' in self.projector_type:
            x = self.linear_1(x)
            x = x.permute(0, 2, 1).reshape(b, -1, h, h)
            x = self.pooling(x)
            x = x.flatten(2).permute(0, 2, 1)
            x = self.linear_2(x)
        elif 'v2' in self.projector_type:
            x = self.linear_1(x)
            x = self.linear_2(x)
            x = x.permute(0, 2, 1).reshape(b, -1, h, h)
            x = self.pooling(x)
            x = x.flatten(2).permute(0, 2, 1)
        elif 'v3' in self.projector_type:
            x = self.linear_0(x)
            x = x.permute(0, 2, 1).reshape(b, -1, h, h)
            x = self.pooling(x)
            x = x.flatten(2).permute(0, 2, 1)
            x = self.linear_1(x)
            x = self.linear_2(x)
        return x

def build_bev_projector(config, delay_load=False, **kwargs):
    projector_type = "linear"
    if projector_type.startswith('linear'):
        return nn.Linear(256, config.hidden_size)

def build_interactor(config, delay_load=False, **kwargs):
    interactor_type = getattr(config, 'mm_interactor_type', "ca_interactor")
    if interactor_type.startswith('ca_interactor'):
        return CA_interactor(config)

def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'mlp2x_gelu')
    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)

    elif projector_type.startswith('mlp'):
        mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
        if mlp_gelu_match:
            mlp_depth = int(mlp_gelu_match.group(1))
            modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
            for _ in range(1, mlp_depth):
                modules.append(nn.GELU())
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
            return nn.Sequential(*modules)
        
    elif projector_type.startswith('ca_adapter'):
        return CA_adapter(config)

    elif projector_type.startswith('spp'):
        return SPP(config, projector_type)

    elif projector_type == 'ldp':
        return LDPNetProjector(config)

    elif projector_type == 'vanilla':
        return Vanilla(config)

    elif projector_type == 'minigpt':
        return Minigpt(config)

    elif projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

class MultiHeadAttention_(nn.Module):
    def __init__(
            self,
            embed_dim=1024,
            num_heads=8,
            dropout=0.1,
            bias=True,
            kdim=None,
            vdim=None,
            batch_first=False):
        super().__init__()

        self.embed_dim = embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim
        self._qkv_same_embed_dim = self.kdim == embed_dim and self.vdim == embed_dim

        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        self.scaled_dot_product_attention = F.scaled_dot_product_attention

        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
            self,
            query,
            key,
            value,
            key_padding_mask=None,
            need_weights=True,
            attn_mask=None,
            average_attn_weights=True,
            is_causal=False):

        if attn_mask is not None and is_causal:
            raise AssertionError("Only allow causal mask or attn_mask")
        is_batched = query.dim() == 3
        key_padding_mask = F._canonical_mask(
            mask=key_padding_mask,
            mask_name="key_padding_mask",
            other_type=F._none_or_dtype(attn_mask),
            other_name="attn_mask",
            target_type=query.dtype
        )

        if self.batch_first and is_batched:
            if key is value:
                if query is key:
                    query = key = value = query.transpose(1, 0)
                else:
                    query, key = [x.transpose(1, 0) for x in (query, key)]
                    value = key
            else:
                query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

        tgt_len, bsz, embed_dim = query.shape
        src_len, _, _ = key.shape

        E = query.size(-1)
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        attn_mask = F._canonical_mask(
            mask=attn_mask,
            mask_name="attn_mask",
            other_type=F._none_or_dtype(key_padding_mask),
            other_name="key_padding_mask",
            target_type=q.dtype,
            check_other=False,
        )

        if attn_mask is not None:
            # ensure attn_mask's dim is 3
            if attn_mask.dim() == 2:
                correct_2d_size = (tgt_len, src_len)
                if attn_mask.shape != correct_2d_size:
                    raise RuntimeError(
                        f"The shape of the 2D attn_mask is {attn_mask.shape}, but should be {correct_2d_size}.")
                attn_mask = attn_mask.unsqueeze(0)
            elif attn_mask.dim() == 3:
                correct_3d_size = (bsz * self.num_heads, tgt_len, src_len)
                if attn_mask.shape != correct_3d_size:
                    raise RuntimeError(
                        f"The shape of the 3D attn_mask is {attn_mask.shape}, but should be {correct_3d_size}.")
            else:
                raise RuntimeError(f"attn_mask's dimension {attn_mask.dim()} is not supported")

        if attn_mask is not None:
            if attn_mask.size(0) == 1 and attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(0)
            else:
                attn_mask = attn_mask.view(bsz, self.num_heads, -1, src_len)

        dropout_p = self.dropout if self.training else 0.

        q = q.view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(src_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
        src_len = k.size(1)
        q = q.view(bsz, self.num_heads, tgt_len, self.head_dim)
        k = k.view(bsz, self.num_heads, src_len, self.head_dim)
        v = v.view(bsz, self.num_heads, src_len, self.head_dim)
        
        energy = torch.einsum('bhqc,bhkc->bhqk', [q, k]) / math.sqrt(self.head_dim)

        attn_weights = nn.functional.softmax(energy, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_weights = nn.functional.dropout(attn_weights, p=dropout_p, training=self.training)
        
        out = torch.einsum('bhqk,bhkc->bhqc', [attn_weights, v])
        out = out.reshape(bsz, tgt_len, embed_dim)
        out = self.proj(out)
        
        return out

class TransformerDecoderLayer_(nn.Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.
    """
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation=F.relu,
                 layer_norm_eps=1e-5, batch_first = True, norm_first= False,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.self_attn = MultiHeadAttention_(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=bias)
        self.multihead_attn = MultiHeadAttention_(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 bias=bias)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        # Legacy string support for activation function.
        self.activation = activation

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        memory_is_causal: bool = False,
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.
        Args:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x

    # self-attention block
    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)

    # multihead attention block
    def _mha_block(self, x: Tensor, mem: Tensor,
                   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.multihead_attn(x, mem, mem,
                                attn_mask=attn_mask,
                                key_padding_mask=key_padding_mask,
                                is_causal=is_causal,
                                need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)