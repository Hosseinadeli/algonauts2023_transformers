# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Backbone modules.
"""
from collections import OrderedDict

import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List
import math

from util.misc import NestedTensor, is_main_process

from .position_encoding import build_position_encoding


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2", "layer4": "3"}
        else:
            return_layers = {'layer4': "0"}
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool):
        backbone = getattr(torchvision.models, name)(
            replace_stride_with_dilation=[False, False, dilation],
            pretrained=is_main_process(), norm_layer=FrozenBatchNorm2d)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos
    
    
from typing import Dict, Iterable, Callable

    
class Backbone_dino(nn.Module):

    def __init__(self, enc_output_layer):
        super().__init__()   
        
        self.backbone = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14')
        self.num_channels = 768
        
        for name, parameter in self.backbone.named_parameters():
            parameter.requires_grad_(False)
            
        self.qkv_feats = {'qkv_feats':torch.empty(0)}
        
        self.backbone._modules["blocks"][enc_output_layer]._modules["attn"]._modules["qkv"].register_forward_hook(self.hook_fn_forward_qkv)  #self.hook_fn_forward_qkv())
        
    def hook_fn_forward_qkv(self, module, input, output) -> Callable:
#         def fn(_, __, output):
        self.qkv_feats['qkv_feats'] = output
            

    def forward(self, tensor_list: NestedTensor):
        xs = tensor_list.tensors
#         self.qkv_feats = []    
#         qkv_feats = []
            
#         self.backbone._modules["blocks"][-1]._modules["attn"]._modules["qkv"].register_forward_hook(lambda self, input, output: qkv_feats.append(output))
        
        xs = self.backbone.get_intermediate_layers(xs)[0]

        feats = self.qkv_feats['qkv_feats']
        # Dimensions
        nh = 12 #Number of heads
        
        feats = feats.reshape(xs.shape[0], xs.shape[1]+1, 3, nh, -1 // nh).permute(2, 0, 3, 1, 4)
        q, k, v = feats[0], feats[1], feats[2]
        q = q.transpose(1, 2).reshape(xs.shape[0], xs.shape[1]+1, -1)
        
        xs = q[:,1:,:]

        xs = {'layer_top':xs}
#         xs = self.body(tensor_list.tensors)

        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None

            x = torch.reshape(x, (x.shape[0],int(math.sqrt(x.shape[1])),int(math.sqrt(x.shape[1])),self.num_channels)).permute(0,3,1,2)
            
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


def build_backbone(args):
    position_embedding = build_position_encoding(args)
    train_backbone = args.lr_backbone > 0
    return_interm_layers = True #args.masks
    
    if args.backbone == 'resnet50':
        backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
    elif args.backbone == 'dinov2':
        #backbone = Backbone(args.backbone, train_backbone, return_interm_layers, args.dilation)
        backbone = Backbone_dino(-1*args.enc_output_layer)
        
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model
