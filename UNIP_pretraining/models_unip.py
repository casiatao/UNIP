# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

from functools import partial
import torch
import torch.nn as nn
from vit import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed


class UNIP(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, drop_path=0.1,
                 embed_dim=1024, depth=24, num_heads=16, last_heads=12,
                 mlp_ratio=4., norm_layer=nn.LayerNorm, loss_type='KL'):
        super().__init__()
        
        self.depth = depth
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.loss_type = loss_type
        self.last_heads = last_heads
        
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, drop_path=drop_path, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)
            for i in range(depth-1)]+[Block(embed_dim, self.last_heads, mlp_ratio, drop_path=drop_path, qkv_bias=True, qk_scale=None, norm_layer=norm_layer)])
        self.norm = norm_layer(embed_dim)

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
 
    
    def forward_encoder(self, x):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            if i == self.depth - 1:
                x, qk = blk(x, return_attention=True)
                return qk
            else:
                x = blk(x)

        
    def forward_kd_loss(self, pred, teacher_out):
        if self.loss_type == "CE":
            loss = nn.CrossEntropyLoss(reduction='none')(pred, torch.softmax(teacher_out, dim=-1)).sum(-1)   # [B, H]
        elif self.loss_type == 'KL':
            loss = nn.KLDivLoss(reduction="none", log_target=True)(torch.log_softmax(pred, dim=-1), torch.log_softmax(teacher_out, dim=-1)).sum(-1) # [B, H]
        else:
            raise ValueError(f"Loss type {self.loss_type} is not supported!")
        return loss.mean()
    

    def forward(self, imgs, teacher_qk):
        student_qk = self.forward_encoder(imgs)
        assert student_qk.shape == teacher_qk.shape, "The outputs of student and teachers unmatched!"
        qk_loss = self.forward_kd_loss(student_qk.float(), teacher_qk.float())

        return qk_loss


def unip_vit_tiny_patch16(**kwargs):
    model = UNIP(
        patch_size=16, embed_dim=192, depth=12, num_heads=3, drop_path=0.1,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def unip_vit_small_patch16(**kwargs):
    model = UNIP(
        patch_size=16, embed_dim=384, depth=12, num_heads=6,drop_path=0.1,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model


def unip_vit_base_patch16(**kwargs):
    model = UNIP(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, drop_path=0.1,
        mlp_ratio=4, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model



