# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn

from util import box_ops
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized)

from models.backbone import build_backbone
from models.matcher import build_matcher
from models.segmentation import (DETRsegm, PostProcessPanoptic, PostProcessSegm,
                           dice_loss, sigmoid_focal_loss)
from models.transformer import build_transformer

from models.decoders import get_decoder
from .position_encoding import build_position_encoding

import numpy as np
import os
import math
        
        
def pos2posemb2d(pos, num_pos_feats=128, temperature=10000):
    scale = 2 * math.pi
    pos = pos * scale
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)
    pos_x = pos[..., 0, None] / dim_t
    pos_y = pos[..., 1, None] / dim_t
    pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
    pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
    posemb = torch.cat((pos_y, pos_x), dim=-1)
    return posemb

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
    

class DETR(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, args, backbone, transformer, num_classes, num_queries, aux_loss=False):
        """ Initializes the model.
        Parameters:
            backbone: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            num_classes: number of object classes
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        
        self.decoder_arch = args.decoder_arch
        
        self.num_queries = num_queries
        
        if self.decoder_arch == 'transformer':
            self.transformer = transformer
            self.hidden_dim = transformer.d_model
            
                    # TODO remove this or make it an option
            self.query_embed = nn.Embedding(num_queries, self.hidden_dim)
            self.query_pos = torch.zeros_like(self.query_embed.weight) #.double()
            
            
        elif self.decoder_arch == 'conv':
            self.dec_conv1 =  nn.Conv2d(768, 64, kernel_size=6, stride=6)
            self.hidden_dim = 1600
        
        self.saved_feats = args.saved_feats
        
        feature_dim = self.hidden_dim     


        self.masks = False
        self.backbone_arch = args.backbone
        
        if self.backbone_arch:
            #backbone.num_channels = 1024
            self.backbone = backbone
            
            if self.backbone_arch == 'resnet50':
                self.input_proj = nn.Conv2d(backbone.num_channels, self.hidden_dim, kernel_size=1)
            
            
        self.aux_loss = aux_loss
        

        if self.masks:
            #self.input_proj_2 = nn.Conv2d(1024, 64, kernel_size=1)
            self.input_proj_conv2 = nn.Conv2d(1024, 64, kernel_size=2, stride=2, padding=1)
            self.input_proj_conv3 = nn.Conv2d(512, 64, kernel_size=4, stride=4, padding=1)
            self.input_proj_conv4 = nn.Conv2d(256, 64, kernel_size=8, stride=8, padding=3)
        
        if self.saved_feats:
            self.position_embedding = build_position_encoding(args)
#             if self.saved_feats == 'clip':
#                 self.input_proj = nn.Conv2d(768, self.hidden_dim, kernel_size=1)
        
        #self.input_proj_2 = nn.Conv2d(1024, self.hidden_dim, kernel_size=1)
        
        self.enc_layers = args.enc_layers
        self.dec_layers = args.dec_layers        
                
        #self.class_embed = nn.Linear(2*self.hidden_dim, num_classes + 1)
        
        
        if self.enc_layers == 0 and self.dec_layers == 0:
            self.output_layer = args.output_layer
            
            if self.output_layer == 'backbone':
                feature_dim = backbone.num_channels
            
#         self.class_embed = nn.Sequential(
#             nn.Linear(2*feature_dim, feature_dim),
#             nn.ReLU(),
#             nn.Linear(feature_dim, num_classes + 1),
#         )
        
        self.readout_res = args.readout_res
        if self.readout_res == 'hemis':
        
            self.lh_embed = nn.Sequential(
    #             nn.Linear(feature_dim, 2*feature_dim),
    #             nn.ReLU(),
                nn.Linear(feature_dim, args.lh_vs),
            )

            self.rh_embed = nn.Sequential(
    #             nn.Linear(feature_dim, 2*feature_dim),
    #             nn.ReLU(),
                nn.Linear(feature_dim, args.rh_vs),
            )
        
        # self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        # decoder_recon = get_decoder('OCRA')
        # self.recon_embed = decoder_recon(args,args.im_dims,transformer.d_model,num_queries)
        
        elif self.readout_res != 'hemis':
            
            lh_vs = args.lh_vs
            rh_vs = args.rh_vs

            #stream 0
            self.lh_embed_0 = nn.Sequential(
                nn.Linear(feature_dim, lh_vs),
            )

            self.rh_embed_0 = nn.Sequential(
                nn.Linear(feature_dim, rh_vs),
            )
            # #stream 1
            # self.lh_embed_1 = nn.Sequential(
            #     nn.Linear(feature_dim, lh_vs),
            # )

            # self.rh_embed_1 = nn.Sequential(
            #     nn.Linear(feature_dim, rh_vs),
            # )
            # #stream 2
            # self.lh_embed_2 = nn.Sequential(
            #     nn.Linear(feature_dim, lh_vs),
            # )

            # self.rh_embed_2 = nn.Sequential(
            #     nn.Linear(feature_dim, rh_vs),
            # )
            # #stream 3
            # self.lh_embed_3 = nn.Sequential(
            #     nn.Linear(feature_dim, lh_vs),
            # )

            # self.rh_embed_3 = nn.Sequential(
            #     nn.Linear(feature_dim, rh_vs),
            # )
            #  #stream 4
            # self.lh_embed_4 = nn.Sequential(
            #     nn.Linear(feature_dim, lh_vs),
            # )

            # self.rh_embed_4 = nn.Sequential(
            #     nn.Linear(feature_dim, rh_vs),
            # )
            # #stream 5
            # self.lh_embed_5 = nn.Sequential(
            #     nn.Linear(feature_dim, lh_vs),
            # )

            # self.rh_embed_5 = nn.Sequential(
            #     nn.Linear(feature_dim, rh_vs),
            # )
            # #stream 6
            # self.lh_embed_6 = nn.Sequential(
            #     nn.Linear(feature_dim, lh_vs),
            # )

            # self.rh_embed_6 = nn.Sequential(
            #     nn.Linear(feature_dim, rh_vs),
            # )
            # #stream 7 
            # self.lh_embed_7 = nn.Sequential(
            #     nn.Linear(feature_dim, lh_vs),
            # )

            # self.rh_embed_7 = nn.Sequential(
            #     nn.Linear(feature_dim, rh_vs),
            # )   
        
        
        
        self.cos = nn.CosineSimilarity(dim=1, eps=1e-6)
        # self.adapt_pos2d = nn.Sequential(
            # nn.Linear(self.hidden_dim, self.hidden_dim),
            # nn.ReLU(),
            # nn.Linear(self.hidden_dim, self.hidden_dim),
        # )
        # self.adapt_pos2d = self.adapt_pos2d.float()  #.double()

    def forward(self, samples: NestedTensor):
        """ The forward expects a NestedTensor, which consists of:
               - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
               - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels
            It returns a dict with the following elements:
               - "pred_logits": the classification logits (including no-object) for all queries.
                                Shape= [batch_size x num_queries x (num_classes + 1)]
               - "pred_boxes": The normalized boxes coordinates for all queries, represented as
                               (center_x, center_y, height, width). These values are normalized in [0, 1],
                               relative to the size of each individual image (disregarding possible padding).
                               See PostProcess for information on how to retrieve the unnormalized bounding box.
               - "aux_outputs": Optional, only returned when auxilary losses are activated. It is a list of
                                dictionnaries containing the two above keys for each decoder layer.
        """
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
            
            
        if self.saved_feats:
            
            pos_embed = self.position_embedding(samples).to(samples.tensors.dtype)
            mask = samples.mask
            
            input_proj_src = samples.tensors #self.input_proj(samples.tensors)
            
            src_all = input_proj_src
            
        else:
            with torch.no_grad():
                features, pos = self.backbone(samples)

            # print(self.hidden_dim)

            # print('self.query_embed.weight.shape')
            # print(self.query_embed.weight.shape)

            src, mask = features[-1].decompose()
            assert mask is not None
            pos_embed = pos[-1]

            if self.masks:

                src_2, mask_2 = features[-2].decompose()
                src_3, mask_3 = features[-3].decompose()
                src_4, mask_4 = features[-4].decompose()

                src_2 = self.input_proj_conv2(src_2)
                src_3 = self.input_proj_conv3(src_3)
                src_4 = self.input_proj_conv4(src_4)

                src_all = torch.cat((src_2, src_3,src_4), dim=1)
            
            
            if self.backbone_arch == 'dinov2':
                
                input_proj_src = src
                src_all = input_proj_src
                
            else:
                input_proj_src = self.input_proj(src)
           
        
#         print(input_proj_src, pos_embed, mask)
            
#         print(f'src all shape : {src_all.shape} and pos_embed shape : {pos_embed.shape}')
        
#         print(f'src shape : {src.shape}')

        #self.query_embed.weight = dots_coords['coords']
        # self.query_pos = self.adapt_pos2d(coords[0]) #.double()   # cuda()
        
        _,_,h,w = pos_embed.shape
        
        
#         self.query_pos = torch.cat((pos_embed[:,:, idxs[0][0], idxs[0][1]], pos_embed[:,:, idxs[1][0], idxs[1][1]]), dim=0)
        
        # print('self.query_pos', self.query_pos.shape)
        # print(idxs)     
        
        
        if self.decoder_arch == 'transformer':
            
            if self.enc_layers == 0:

                if self.dec_layers > 0:

                    hs = self.transformer(input_proj_src, mask, self.query_embed.weight, pos_embed, self.masks, src_all) 
                    output_tokens = hs[-1]

                else:

                    if self.output_layer == 'backbone':
                        hs = src
                        output_tokens = torch.cat((hs[:,:, idxs[0][0], idxs[0][1]], hs[:,:, idxs[1][0], idxs[1][1]]),dim=0).unsqueeze(0)
                        #output_tokens = output_tokens + self.query_pos 
                        #output_tokens = output_tokens.unsqueeze(0)
                        #print('output_tokens',output_tokens.shape)

                    elif self.output_layer == 'input_proj':
                        hs = self.input_proj(src)
                        output_tokens = torch.cat((hs[:,:, idxs[0][0], idxs[0][1]], hs[:,:, idxs[1][0], idxs[1][1]]),dim=0)
                        output_tokens = output_tokens + self.query_pos 
                        output_tokens = output_tokens.unsqueeze(0)
                        #print('output_tokens',output_tokens.shape)


            else: 

                hs = self.transformer(input_proj_src, mask, self.query_embed.weight, pos_embed, self.masks, src_all) #[0] # get the decoer output for all layers 
                #self.query_embed.weight  self.query_pos coords[0]

                #print(hs[:,:, idxs[0][0], idxs[0][1]].shape)

                if self.dec_layers == 0:

                    output_tokens = torch.cat((hs[:,:, 0, 0], hs[:,:, 0, 0]),dim=0).unsqueeze(0)

        #             pred_sim = self.cos(hs[:,:, idxs[0][0], idxs[0][1]], hs[:,:, idxs[1][0], idxs[1][1]])
        #             hs = torch.cat((hs[:,:, idxs[0][0], idxs[0][1]], hs[:,:, idxs[1][0], idxs[1][1]]), dim=1)

                else:
                    output_tokens = hs[-1]

                    #output_tokens = torch.cat((hs[-1], hs[-2], hs[-3]), 0)
                    # get the last decoder layer output hs[-1]: [1,2,256]   [:,0:1,:]
                
                
        elif self.decoder_arch == 'conv':
            output_tokens = self.dec_conv1(input_proj_src)
            output_tokens = output_tokens.flatten(1).unsqueeze(1)
            output_tokens = torch.tile(output_tokens, (1,16,1))
        
        #hs = output_tokens.flatten(1) #.detach()
        
        if self.readout_res == 'hemis':
            
            lh_f_pred = self.lh_embed(output_tokens[:,0,:])
            rh_f_pred = self.rh_embed(output_tokens[:,1,:])
        
        else: #if self.readout_res == 'streams':

            lh_f_pred_0 = self.lh_embed_0(output_tokens[:,0,:])
            lh_f_pred_1 = self.lh_embed_0(output_tokens[:,1,:])
            lh_f_pred_2 = self.lh_embed_0(output_tokens[:,2,:])
            lh_f_pred_3 = self.lh_embed_0(output_tokens[:,3,:])
            lh_f_pred_4 = self.lh_embed_0(output_tokens[:,4,:])
            lh_f_pred_5 = self.lh_embed_0(output_tokens[:,5,:])
            lh_f_pred_6 = self.lh_embed_0(output_tokens[:,6,:])
            lh_f_pred_7 = self.lh_embed_0(output_tokens[:,7,:])

            rh_f_pred_0 = self.rh_embed_0(output_tokens[:,8,:])
            rh_f_pred_1 = self.rh_embed_0(output_tokens[:,9,:])
            rh_f_pred_2 = self.rh_embed_0(output_tokens[:,10,:])
            rh_f_pred_3 = self.rh_embed_0(output_tokens[:,11,:])
            rh_f_pred_4 = self.rh_embed_0(output_tokens[:,12,:])
            rh_f_pred_5 = self.rh_embed_0(output_tokens[:,13,:])
            rh_f_pred_6 = self.rh_embed_0(output_tokens[:,14,:])
            rh_f_pred_7 = self.rh_embed_0(output_tokens[:,15,:])

            lh_f_pred = torch.stack((lh_f_pred_0, lh_f_pred_1, lh_f_pred_2,lh_f_pred_3,lh_f_pred_4,lh_f_pred_5,lh_f_pred_6,lh_f_pred_7), dim=2)

            rh_f_pred = torch.stack((rh_f_pred_0, rh_f_pred_1, rh_f_pred_2,rh_f_pred_3,rh_f_pred_4,rh_f_pred_5,rh_f_pred_6,rh_f_pred_7), dim=2)
        
        
        out = {'lh_f_pred': lh_f_pred, 'rh_f_pred': rh_f_pred, 'output_tokens': output_tokens}
        
#         out = {'lh_f_pred_0': lh_f_pred_0, 'rh_f_pred_0': rh_f_pred_0, 'lh_f_pred_1': lh_f_pred_1, 'rh_f_pred_1': rh_f_pred_1, \
#                'lh_f_pred_2': lh_f_pred_2, 'rh_f_pred_2': rh_f_pred_2, 'lh_f_pred_3': lh_f_pred_3, 'rh_f_pred_3': rh_f_pred_3, \
#                'lh_f_pred_4': lh_f_pred_4, 'rh_f_pred_4': rh_f_pred_4, 'lh_f_pred_5': lh_f_pred_5, 'rh_f_pred_5': rh_f_pred_5, \
#                'lh_f_pred_6': lh_f_pred_6, 'rh_f_pred_6': rh_f_pred_6, 'lh_f_pred_7': lh_f_pred_7, 'rh_f_pred_7': rh_f_pred_7, \
#                'output_tokens': output_tokens}
               
        #pred_sim = self.cos(output_tokens[:,0,:], output_tokens[:,1,:])
        
        #outputs_class = self.class_embed(hs)
        
        # outputs_coord = self.bbox_embed(hs).sigmoid()
#         outputs_recon = self.recon_embed(hs[-1].flatten(0,1)).view(hs[0].shape[0],self.num_queries,-1)
        
        #out = {'pred_logits': outputs_class} #, 'pred_recons': outputs_recon , 'pred_boxes': outputs_coord[-1]
        #out = {'pred_logits': outputs_class, 'output_tokens': output_tokens, 'pred_sim': pred_sim}
        
        if self.aux_loss:
            out['aux_outputs'] = self._set_aux_loss(outputs_class, outputs_coord)
            
        return out

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_coord):
        # this is a workaround to make torchscript happy, as torchscript
        # doesn't support dictionary with non-homogeneous values, such
        # as a dict having both a Tensor and a list.
        return [{'pred_logits': a, 'pred_boxes': b}
                for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
    

    
# bbox_embed = MLP(64, 64, 4, 3)
# print(bbox_embed)


class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, args, num_classes, matcher, weight_dict, eos_coef, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        empty_weight = empty_weight.to(args.device)
        self.cosine_embedding_loss = nn.CosineEmbeddingLoss(reduction='mean')
        self.register_buffer('empty_weight', empty_weight)
        
        self.readout_res = args.readout_res
        
                # Load the ROI classes mapping dictionaries
        roi_mapping_files = ['mapping_prf-visualrois.npy', 'mapping_floc-bodies.npy',
            'mapping_floc-faces.npy', 'mapping_floc-places.npy',
            'mapping_floc-words.npy', 'mapping_streams.npy']
        roi_name_maps = []
        for r in roi_mapping_files:
            roi_name_maps.append(np.load(os.path.join(args.data_dir, 'roi_masks', r),
                allow_pickle=True).item())

        # Load the ROI brain surface maps
        lh_challenge_roi_files = ['lh.prf-visualrois_challenge_space.npy',
            'lh.floc-bodies_challenge_space.npy', 'lh.floc-faces_challenge_space.npy',
            'lh.floc-places_challenge_space.npy', 'lh.floc-words_challenge_space.npy',
            'lh.streams_challenge_space.npy']
        rh_challenge_roi_files = ['rh.prf-visualrois_challenge_space.npy',
            'rh.floc-bodies_challenge_space.npy', 'rh.floc-faces_challenge_space.npy',
            'rh.floc-places_challenge_space.npy', 'rh.floc-words_challenge_space.npy',
            'rh.streams_challenge_space.npy']
        lh_challenge_rois = []
        rh_challenge_rois = []
        for r in range(len(lh_challenge_roi_files)):
            lh_challenge_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
                lh_challenge_roi_files[r])))
            rh_challenge_rois.append(np.load(os.path.join(args.data_dir, 'roi_masks',
                rh_challenge_roi_files[r])))

        self.roi_name_maps = roi_name_maps
        
        self.lh_challenge_rois = torch.tensor(lh_challenge_rois).to(args.device)
        self.rh_challenge_rois = torch.tensor(rh_challenge_rois).to(args.device)
        
        args.lh_vs = len(lh_challenge_rois[args.rois_ind])
        args.rh_vs = len(rh_challenge_rois[args.rois_ind])
        
        self.rois_ind = args.rois_ind
        
        self.lh_vs = args.lh_vs 
        self.rh_v = args.rh_vs 

    def loss_labels(self, samples, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']
        
        # print('new trial')
        # print(outputs)

        # pick a specific query for the label prediction - here the last query - 10
        # pick a specific target label index, since it's only one 
        indices = [(torch.as_tensor([0], dtype=torch.int64), torch.as_tensor([0], dtype=torch.int64)) for i in range(len(targets))]
        idx = self._get_src_permutation_idx(indices)
        
        # print('idx = {}'.format(idx))

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        
        # target_classes[idx] = target_classes_o
        
        # print(target_classes_o)
        # print(src_logits[:,0,:])

        loss_ce = F.cross_entropy(src_logits, target_classes_o, self.empty_weight)
#         loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        if log:
            # TODO this should probably be a separate loss, not hacked in this one here
            losses['class_error'] = 100 - accuracy(src_logits, target_classes_o)[0]
        return losses


    def loss_cosine(self, samples, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'output_tokens' in outputs
        output_tokens = outputs['output_tokens']
        

        # pick a specific query for the label prediction - here the last query - 10
        # pick a specific target label index, since it's only one 
        indices = [(torch.as_tensor([0], dtype=torch.int64), torch.as_tensor([0], dtype=torch.int64)) for i in range(len(targets))]
        idx = self._get_src_permutation_idx(indices)
        
        # print('idx = {}'.format(idx))

        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        
        # target_classes[idx] = target_classes_o
        
        # print(target_classes_o)
        # print(src_logits[:,0,:])

        loss_cos = self.cosine_embedding_loss(output_tokens[:,0,:], output_tokens[:,1,:], (2*target_classes_o)-1)

        losses = {'loss_cos': loss_cos}

        return losses
        
        
    @torch.no_grad()
    def loss_cardinality(self, samples, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, samples, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')

        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        loss_giou = 1 - torch.diag(box_ops.generalized_box_iou(
            box_ops.box_cxcywh_to_xyxy(src_boxes),
            box_ops.box_cxcywh_to_xyxy(target_boxes)))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def loss_masks(self, samples, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses
    
    
    def loss_recons(self, samples, outputs, targets, indices, num_boxes, log=True):
        """Reconstruction loss (MSE)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_recons' in outputs        

        
        target_masks = samples.tensors
        target_masks = target_masks.flatten(1) # [128,3*64*64]      
        
        print(target_masks.shape)
        
        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs['pred_recons']#.flatten(2) #torch.Size([128, 10, 3, 64, 64])
        src_masks = src_masks[src_idx]        
        src_masks = src_masks.view(target_masks.shape[0],2,-1).contiguous().sum(1)  # [128,3*64*64]
#         src_masks = src_masks.sum(1).flatten(1)
        src_masks = torch.clip(src_masks,0.,1.)
        
        #[tgt_idx] #tgt_idx is either 0 or 1 since two possible target indeces
#         print(target_masks.shape)
#         src_masks = src_masks[:, 0].flatten(1)
#         print(src_masks.shape)
        
#         target_masks = target_masks.view(src_masks.shape)
        
        loss_recon = nn.MSELoss()(src_masks, target_masks)
        losses = {'loss_recon': loss_recon}

        log=False
        if log:
            losses['recon_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses
    
    
    def loss_mse_fmri(self, samples, outputs, targets, indices, num_boxes, log=True):
        """Reconstruction loss (MSE)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'lh_f_pred' in outputs    
        assert 'rh_f_pred' in outputs 
        
        
        if self.readout_res != 'hemis':
            lh_rois = self.lh_challenge_rois[self.rois_ind]
            rh_rois = self.rh_challenge_rois[self.rois_ind]

            lh_challenge_rois = []
            rh_challenge_rois = []
            for i in range(len(self.roi_name_maps[self.rois_ind])):
                lh_challenge_rois.append(torch.where(lh_rois == i, 1, 0))
                rh_challenge_rois.append(torch.where(rh_rois == i, 1, 0))


            lh_challenge_rois = torch.vstack(lh_challenge_rois)
            rh_challenge_rois = torch.vstack(rh_challenge_rois)

            lh_challenge_rois = torch.tile(lh_challenge_rois[:,:,None], (1,1,targets['lh_f'].shape[0])).permute(2,1,0)
            rh_challenge_rois = torch.tile(rh_challenge_rois[:,:,None], (1,1,targets['rh_f'].shape[0])).permute(2,1,0)
            
            outputs['lh_f_pred'] = torch.sum(torch.mul(lh_challenge_rois, outputs['lh_f_pred'][:,:,:len(self.roi_name_maps[self.rois_ind])]), dim=2)
            outputs['rh_f_pred'] = torch.sum(torch.mul(rh_challenge_rois, outputs['rh_f_pred'][:,:,:len(self.roi_name_maps[self.rois_ind])]), dim=2)
            
#         outputs['lh_f_pred'] 
#         outputs['rh_f_pred']

            if self.readout_res != 'streams_inc':

                outputs['lh_f_pred'] = (1*(lh_rois>0)) * outputs['lh_f_pred']
                outputs['rh_f_pred'] = (1*(rh_rois>0)) * outputs['rh_f_pred']

                targets['lh_f'] = (1*(lh_rois>0)) * targets['lh_f']
                targets['rh_f'] = (1*(rh_rois>0)) * targets['rh_f']
        
        loss_lh = nn.MSELoss()(outputs['lh_f_pred'], targets['lh_f'])
        loss_rh = nn.MSELoss()(outputs['rh_f_pred'], targets['rh_f'])
        losses = {'loss_mse_fmri': loss_lh+loss_rh}

        log=False
        if log:
            losses['loss_mse_fmri'] = loss_lh+loss_rh
        return losses
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, samples, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cosine': self.loss_cosine,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks,
            'recons': self.loss_recons,
            'mse_fmri': self.loss_mse_fmri
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](samples, outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, samples, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        # TODO make this a condition
        indices = [] # self.matcher(samples, outputs_without_aux, targets)
        
        #indices = [(torch.as_tensor([0], dtype=torch.int64), torch.as_tensor([0], dtype=torch.int64)) for i in range(len(targets))]
        
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        
        # not used for algonauts
#         num_boxes = sum(len(t["labels"]) for t in targets)
#         num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
#         if is_dist_avail_and_initialized():
#             torch.distributed.all_reduce(num_boxes)
#         num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        
        for loss in self.losses:
            losses.update(self.get_loss(loss, samples, outputs, targets, indices, num_boxes=0))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses


class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        # convert to [x0, y0, x1, y1] format
        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        # and from relative [0, 1] to absolute [0, height] coordinates
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = boxes * scale_fct[:, None, :]

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


def build(args):
    # the `num_classes` naming here is somewhat misleading.
    # it indeed corresponds to `max_obj_id + 1`, where max_obj_id
    # is the maximum id for a class in your dataset. For example,
    # COCO has a max_obj_id of 90, so we pass `num_classes` to be 91.
    # As another example, for a dataset that has a single class with id 1,
    # you should pass `num_classes` to be 2 (max_obj_id + 1).
    # For more details on this, check the following discussion
    # https://github.com/facebookresearch/detr/issues/108#issuecomment-650269223
    
    num_classes = args.num_classes  #11 if args.dataset_file != 'coco' else 91
    if args.dataset_file == "coco_panoptic":
        # for panoptic, we just add a num_classes that is large enough to hold
        # max_obj_id + 1, but the exact value doesn't really matter
        num_classes = 250
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_transformer(args)

    model = DETR(
        args,
        backbone,
        transformer,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
    )
    
    if args.masks:
        model = DETRsegm(model, freeze_detr=(args.frozen_weights is not None))
        
    matcher = build_matcher(args)
    #TODO match the weights 
    weight_dict = {'loss_mse_fmri': 1}
    
    #loss_ce': args.class_loss_coef, 'loss_cos': args.cosine_loss_coef}
                   
                  # ,'loss_bbox': args.bbox_loss_coef, 
                  # 'loss_giou' : args.giou_loss_coef
#                  'loss_recon' : args.recon_loss_coef} 
    
#     weight_dict['loss_giou'] = args.giou_loss_coef
#     if args.masks:
#         weight_dict["loss_mask"] = args.mask_loss_coef
#         weight_dict["loss_dice"] = args.dice_loss_coef
#     # TODO this is a hack
#     if args.aux_loss:
#         aux_weight_dict = {}
#         for i in range(args.dec_layers - 1):
#             aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items()})
#         weight_dict.update(aux_weight_dict)

    losses = ['mse_fmri'] # 'labels', 'cosine','recons', 'boxes', 'cardinality']
#     if args.masks:
#         losses += ["masks"]
    criterion = SetCriterion(args, num_classes, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=args.eos_coef, losses=losses)
#     criterion.to(device)
#     postprocessors = {'bbox': PostProcess()}
#     if args.masks:
#         postprocessors['segm'] = PostProcessSegm()
#         if args.dataset_file == "coco_panoptic":
#             is_thing_map = {i: i <= 90 for i in range(201)}
#             postprocessors["panoptic"] = PostProcessPanoptic(is_thing_map, threshold=0.85)

    return model, criterion #, postprocessors

