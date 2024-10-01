import os
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from mmcv_custom import load_checkpoint
from mmseg.utils import get_root_logger
from mmseg.models.builder import BACKBONES
from backbone.basevit import BaseViT
import torch.nn.functional as F

@BACKBONES.register_module()
class VisionTransformer(BaseViT):
    def __init__(self,
                 patch_size,
                 embed_dim,
                 with_fpn=True,
                 frozen_stages=-1,
                 out_indices=[3, 5, 7, 11],
                 use_checkpoint=False,
                 use_norm=False,
                 remove_norm_weight=False,
                 use_ind_norm=False,
                 use_attn_feat=False,
                 **kwargs):
        super(VisionTransformer, self).__init__(
            patch_size=patch_size, 
            embed_dim=embed_dim, 
            **kwargs)
        self.patch_size = patch_size
        self.with_fpn = with_fpn
        self.frozen_stages = frozen_stages
        self.out_indices = out_indices
        self.use_checkpoint = use_checkpoint
        self.use_norm = use_norm
        self.use_ind_norm = use_ind_norm
        self.remove_norm_weight = remove_norm_weight
        self.embed_dim = embed_dim
        self.use_attn_feat = use_attn_feat
        # if not out_with_norm:
        #     self.norm = nn.Identity()

        if with_fpn and patch_size == 16:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
                nn.SyncBatchNorm(embed_dim),
                nn.GELU(),
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn3 = nn.Identity()

            self.fpn4 = nn.MaxPool2d(kernel_size=2, stride=2)
        elif with_fpn and patch_size == 8:
            self.fpn1 = nn.Sequential(
                nn.ConvTranspose2d(embed_dim, embed_dim, kernel_size=2, stride=2),
            )

            self.fpn2 = nn.Identity()

            self.fpn3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2),
            )

            self.fpn4 = nn.Sequential(
                nn.MaxPool2d(kernel_size=4, stride=4),
            )
        else:
            logger = get_root_logger()
            logger.info('Build model without FPN.')
            
            
    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.patch_embed.eval()
            for param in self.patch_embed.parameters():
                param.requires_grad = False
            self.cls_token.requires_grad = False
            self.pos_embed.requires_grad = False
            self.pos_drop.eval()

        for i in range(1, self.frozen_stages + 1):
            
            if i  == len(self.blocks):
                norm_layer = getattr(self, 'norm') #f'norm{i-1}')
                norm_layer.eval()
                for param in norm_layer.parameters():
                    param.requires_grad = False

            m = self.blocks[i - 1]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode while keep layers freezed."""
        super(VisionTransformer, self).train(mode)
        self._freeze_stages()
        

    
            
    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.
        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        if isinstance(pretrained, str):
            self.apply(self._init_weights)
            logger = get_root_logger()
            if  os.path.isfile(pretrained):
                load_checkpoint(self, pretrained, strict=False, logger=logger)
            else:
                logger.info(f"checkpoint path {pretrained} is invalid, we skip it and initialize net randomly")
        elif pretrained is None:
            self.apply(self._init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        B, _, H, W = x.shape
        Hp, Wp = H // self.patch_size, W // self.patch_size
        x = self.prepare_tokens(x)
        features = []
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                if self.use_attn_feat and i in self.out_indices:
                    x = checkpoint.checkpoint(blk, x, False, True)
                else:  
                    x = checkpoint.checkpoint(blk, x)
            else:
                if self.use_attn_feat and i in self.out_indices:
                    x = blk(x, return_attn_feat=True)
                else:
                    x = blk(x)
            if i in self.out_indices:
                if self.use_norm:
                    if self.remove_norm_weight:
                        xp = F.layer_norm(x[:, 1:, :], (self.embed_dim,)).permute(0, 2, 1).reshape(B, -1, Hp, Wp)
                    else:
                        xp = self.norm(x[:, 1:, :]).permute(0, 2, 1).reshape(B, -1, Hp, Wp) 
                elif self.use_ind_norm:
                    if i == len(self.blocks)-1:
                        xp = self.norm(x[:, 1:, :]).permute(0, 2, 1).reshape(B, -1, Hp, Wp) 
                    else:
                        xp = self.blocks[i+1].norm1(x[:, 1:, :]).permute(0, 2, 1).reshape(B, -1, Hp, Wp) 
                else:
                    xp = x[:, 1:, :].permute(0, 2, 1).reshape(B, -1, Hp, Wp)         
                features.append(xp.contiguous())
                if len(set(self.out_indices)) == 1:
                    break
        
        if len(set(self.out_indices)) == 1:
            features = features * len(self.out_indices)

        if self.with_fpn:
            ops = [self.fpn1, self.fpn2, self.fpn3, self.fpn4]
            for i in range(len(features)):
                features[i] = ops[i](features[i])

        return tuple(features)