# -*- coding: utf-8 -*-
from ast import arg
from typing import Counter, List
import math
import torch
import torch.nn.functional as F
import copy
from torch import nn
from .deformable_detr import SetCriterion, MLP
from models.structures import Instances
from .deformable_transformer_ import build_deforamble_transformer
from .backbone import build_backbone
from .matcher import build_matcher
from util.box_ops import box_xyxy_to_cxcywh, box_iou, box_cxcywh_to_xyxy
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate, get_rank,
                       is_dist_avail_and_initialized, inverse_sigmoid)
from .segmentation import sigmoid_focal_loss
from models.structures import Instances, Boxes, pairwise_iou, matched_boxlist_iou


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class MOMOT(nn.Module):

    def __init__(self, backbone, transformer, num_classes, num_queries, num_feature_levels, criterion,
                 aux_loss=True, with_box_refine=False, two_stage=False, use_checkpoint=False, visibilty_thresh=0.0):
        super().__init__()

        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.num_classes = num_classes
        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_feature_levels = num_feature_levels
        self.use_checkpoint = use_checkpoint
        self.hidden_dim = hidden_dim

        # 1x1 conv
        if not two_stage:
            self.query_embed = nn.Embedding(num_queries, hidden_dim * 2)
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.strides)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(nn.Sequential(
                    nn.Conv2d(in_channels, hidden_dim, kernel_size=3, stride=2, padding=1),
                    nn.GroupNorm(32, hidden_dim),
                ))
                in_channels = hidden_dim
            self.input_proj = nn.ModuleList(input_proj_list)
        else:
            self.input_proj = nn.ModuleList([
                nn.Sequential(
                    nn.Conv2d(backbone.num_channels[0], hidden_dim, kernel_size=1),
                    nn.GroupNorm(32, hidden_dim),
                )])

        self.backbone = backbone
        self.aux_loss = aux_loss
        self.with_box_refine = with_box_refine
        self.two_stage = two_stage

        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        self.class_embed.bias.data = torch.ones(self.num_classes) * bias_value
        nn.init.constant_(self.bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self.bbox_embed.layers[-1].bias.data, 0)
        for proj in self.input_proj:
            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)

        # if two-stage, the last class_embed and bbox_embed is for region proposal generation
        num_pred = (transformer.decoder.num_layers + 1) if self.two_stage else transformer.decoder.num_layers
        if self.with_box_refine:
            self.class_embed = _get_clones(self.class_embed, num_pred)
            self.bbox_embed = _get_clones(self.bbox_embed, num_pred)
            nn.init.constant_(self.bbox_embed[0].layers[-1].bias.data[2:], -2.0)
            # hack implementation for iterative bounding box refinement
            self.transformer.decoder.bbox_embed = self.bbox_embed
        else:
            nn.init.constant_(self.bbox_embed.layers[-1].bias.data[2:], -2.0)
            self.class_embed = nn.ModuleList([self.class_embed for _ in range(num_pred)])
            self.bbox_embed = nn.ModuleList([self.bbox_embed for _ in range(num_pred)])
            self.transformer.decoder.bbox_embed = None

        if self.two_stage:
            # hack implementation for two-stage
            self.transformer.decoder.class_embed = self.class_embed
            for box_embed in self.bbox_embed:
                nn.init.constant_(box_embed.layers[-1].bias.data[2:], 0.0)
        
        # test
        self.track_instances = None
        self.track_base = RuntimeTrackerBase()
        self.criterion = criterion
        self.inference_frame_id = -1
        self.visibilty_thresh = visibilty_thresh

    def forward(self, data: dict):

        if self.training:
            self.criterion.initialize_for_single_clip(data['gt_instances'])
            self.track_instances = generate_empty_tracks(num_tracks=0, 
                                                         device=self.query_embed.weight.device, 
                                                         hidden_dim=self.hidden_dim)
        else:
            if self.track_instances is None:
                self.track_instances = generate_empty_tracks(num_tracks=0, 
                                                             device=self.query_embed.weight.device, 
                                                             hidden_dim=self.hidden_dim)

        frames = data['imgs'] 

        # init
        outputs = {
                'pred_det_logits': [],
                'pred_det_boxes': [],
                'pred_track_logits': [],
                'pred_track_boxes': [],
                'det_gt_idx':[], 
                'track_gt_idx':[],
            }
        if self.training:        
            for frame_id, frame in enumerate(frames):
                frame.requires_grad = False
                frame = nested_tensor_from_tensor_list([frame])
                frame_res = self._forward_single_image(frame)
                frame_res, outputs = self._post_process_single_image(frame_res, outputs)
            outputs['losses_dict'] = self.criterion.losses_dict

            return outputs

        else:
            assert frames.size(0) == 1
            frame_res, outputs = self.inference_single_image(frames[0], outputs)
            return outputs
            
    @torch.no_grad()
    def inference_single_image(self, frame, outputs):
        frame = nested_tensor_from_tensor_list([frame])
        frame_res = self._forward_single_image(frame)
        frame_res, outputs = self._post_process_single_image(frame_res, outputs)
        return frame_res, outputs

    def _forward_single_image(self, samples):
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        assert mask is not None

        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))
            masks.append(mask)
            assert mask is not None

        # generate res6 features
        if self.num_feature_levels > len(srcs):
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(torch.bool)[0]
                pos_l = self.backbone[1](NestedTensor(src, mask)).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                pos.append(pos_l)

        hs, init_reference, inter_references = self.transformer(srcs, masks, pos, self.query_embed.weight, 
                                                                self.track_instances.track_embed, 
                                                                self.track_instances.boxes[:, :2])
        outputs_classes = []
        outputs_coords = []
        for lvl in range(hs.shape[0]): # hs.shape[0]: num of decoder layers
            if lvl == 0:
                reference = init_reference
            else:
                reference = inter_references[lvl - 1]
            num_tracks = hs.size(2) - self.num_queries
            reference = inverse_sigmoid(reference)
            outputs_class = self.class_embed[lvl](hs[lvl])
            tmp = self.bbox_embed[lvl](hs[lvl])
            if reference.shape[-1] == 4:
                tmp += reference
            else:
                assert reference.shape[-1] == 2
                tmp[..., :2] += reference
            outputs_coord = tmp.sigmoid()
            outputs_classes.append(outputs_class)
            outputs_coords.append(outputs_coord)
        outputs_class = torch.stack(outputs_classes)
        outputs_coord = torch.stack(outputs_coords)

        class_det, class_track = torch.split(outputs_class[-1], (self.num_queries, num_tracks), dim=1)
        coord_det, coord_track = torch.split(outputs_coord[-1], (self.num_queries, num_tracks), dim=1)
        hs_det, hs_track = torch.split(hs[-1], (self.num_queries, num_tracks), dim=1)

        return {'pred_det_logits': class_det, 'pred_det_boxes': coord_det, 'hs_det': hs_det,
                'pred_track_logits': class_track, 'pred_track_boxes': coord_track, 'hs_track': hs_track, 
                'query_embed': self.query_embed.weight[:, :self.hidden_dim]}
    
    def _post_process_single_image(self, frame_res, outputs):

        frame_res['track_instances'] = self.track_instances

        if self.training:
            self.track_instances = self.criterion.match_for_single_frame(frame_res)
        else:
            self.track_instances = self.track_base.update(frame_res)
            
        # for debug purpose
        outputs['pred_det_logits'].append(frame_res['pred_det_logits'])
        outputs['pred_det_boxes'].append(frame_res['pred_det_boxes'])
        outputs['pred_track_logits'].append(frame_res['pred_track_logits'])
        outputs['pred_track_boxes'].append(frame_res['pred_track_boxes'])
        outputs['det_gt_idx'].append(frame_res['det_gt_idx'])
        outputs['track_gt_idx'].append(frame_res['track_gt_idx'])
  
        return frame_res, outputs

class ClipMatcher(SetCriterion):
    def __init__(self, num_classes, matcher, weight_dict, losses):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__(num_classes, matcher, weight_dict, losses)
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.losses = losses
        self.focal_loss = True
        self.losses_dict = {}
        self._current_frame_idx = 0
        self.miss_tolerance = 3
        self.iou_thresh = 0.5
        self.visibility_thresh = 0.0
    def initialize_for_single_clip(self, gt_instances: List[Instances]):
        self.gt_instances = gt_instances
        self.num_samples = 0
        self.sample_device = None
        self._current_frame_idx = 0
        self.losses_dict = {}

    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def match_for_single_frame(self, frame_res: dict):
        track_instances: Instances = frame_res['track_instances']

        # previous frame tracklets' gt idx
        track_gt_ids = track_instances.obj_idxes
        device = track_gt_ids.device

        gt_instances_i = self.gt_instances[self._current_frame_idx]
        all_gt_idxes_i = gt_instances_i.obj_ids

        # note: all indexes are w.r.t current frame gt instances' order
        # step1. find detection(new-born) and tracklets(old) ids

        # if idx not found in previous frame track list, record it as a new-born object 
        # current frame instances mapping to previous frame instances
        track_ids = torch.full((len(all_gt_idxes_i),), -1, dtype=torch.long, device=device)

        # keep tracking or new obj
        for i, gt_idx in enumerate(all_gt_idxes_i):
            if gt_idx in track_gt_ids and gt_idx != -1:
                # gt_idx在tracklets中的index, ie: gt 50005: new idx 7 -> old index 3
                track_ids[i] = (track_gt_ids==gt_idx).nonzero().squeeze().item()

        track_src = track_ids[track_ids>=0]
        track_tgt = (track_ids>=0).nonzero(as_tuple=True)[0]

        track_instances.disappear_time += 1
        track_instances.disappear_time[track_src] = 0

        # step2. match untracked objects
        outputs = {
            'pred_logits': frame_res['pred_det_logits'],
            'pred_boxes':  frame_res['pred_det_boxes'],
        }
        det_indices = self.matcher(outputs, [gt_instances_i])
        # matcher回傳queries與其相應的gt_idx
        src_idx = det_indices[0][0].to(device)
        tgt_idx = det_indices[0][1].to(device)

        # reorder src to current gt order(i.e: src:10,151,197,tgt:2,1,0 -> src:197,151,10,tgt:0,1,2)
        src_to_gt_order  = src_idx[torch.argsort(tgt_idx)]

        # step4. calculate losses.
        self.num_samples += len(track_src) + len(src_idx)
        self.sample_device = device
        for loss_type in self.losses:
            if 'det' in loss_type:
                new_det_loss = self.get_loss(loss_type,
                                               outputs=frame_res,
                                               gt_instances=[gt_instances_i],
                                               indices=[(src_idx, tgt_idx)],
                                               num_boxes=1)
                self.losses_dict.update(
                    {'frame_{}_{}_det'.format(self._current_frame_idx, key): value 
                        for key, value in new_det_loss.items()})
            elif 'track' in loss_type:
                #disappear_mask = torch.ones(len(track_gt_ids), dtype=torch.float, device=device)
                #disappear_mask[track_src] = gt_instances_i.visibility[track_tgt]
                prev_track_loss = self.get_loss(loss_type,
                                                outputs=frame_res,
                                                gt_instances=[gt_instances_i],
                                                indices=[(track_src, track_tgt)],
                                                num_boxes=1, 
                                                #disappear_mask = disappear_mask
                                                )
                self.losses_dict.update(
                    {'frame_{}_{}_track'.format(self._current_frame_idx, key): value 
                        for key, value in prev_track_loss.items()})
    
        # calculate iou and filter dead tracklets
        #active_track_boxes = track_instances.boxes[track_src]
        #gt_boxes = gt_instances_i.boxes[track_tgt]
        #active_track_boxes = box_cxcywh_to_xyxy(active_track_boxes)
        #gt_boxes = box_cxcywh_to_xyxy(gt_boxes)
        #active_tracks_iou = matched_boxlist_iou(Boxes(active_track_boxes), Boxes(gt_boxes))
        #active_tracks = torch.zeros(len(track_gt_ids), dtype=torch.bool, device=device)
        #active_tracks[track_src] = active_tracks_iou > self.iou_thresh
        active_tracks = (track_instances.disappear_time < self.miss_tolerance) #& active_tracks
        active_tracks = active_tracks.nonzero(as_tuple=True)[0]

        unmatched_ids =  []
        for i, gt_idx in enumerate(all_gt_idxes_i):
            if gt_idx not in track_gt_ids[active_tracks]:
                unmatched_ids.append(i)
        unmatched_ids = torch.tensor(unmatched_ids, dtype=torch.long, device=device)

        # step3. update new matching result.
        new_track_instances = generate_empty_tracks(num_tracks=len(active_tracks)+len(unmatched_ids), device=device)
        new_track_instances.track_embed = torch.cat([frame_res['hs_track'][0, active_tracks], 
                                                     frame_res['hs_det'][0, src_to_gt_order[unmatched_ids]]], dim=0)
        
        new_track_instances.boxes = torch.cat([frame_res['pred_track_boxes'][0, active_tracks], 
                                                       frame_res['pred_det_boxes'][0, src_to_gt_order[unmatched_ids]]], 
                                                       dim=0)
        new_track_instances.obj_idxes = torch.cat([track_gt_ids[active_tracks],
                                                   all_gt_idxes_i[unmatched_ids]], dim=0)
        new_track_instances.disappear_time[:len(active_tracks)] = track_instances.disappear_time[active_tracks]

        # write results for debug purpose
        frame_res['pred_det_logits'] = frame_res['pred_det_logits'][0, src_to_gt_order[unmatched_ids]]
        frame_res['pred_det_boxes'] = frame_res['pred_det_boxes'][0, src_to_gt_order[unmatched_ids]]
        frame_res['pred_track_logits'] = frame_res['pred_track_logits'][0, track_src]
        frame_res['pred_track_boxes'] = frame_res['pred_track_boxes'][0, track_src]
        frame_res['det_gt_idx'] = new_track_instances.obj_idxes[len(track_gt_ids):]
        frame_res['track_gt_idx'] = track_gt_ids[track_src]
        self._current_frame_idx += 1
        
        return new_track_instances
        
    def get_num_boxes(self, num_samples):
        num_boxes = torch.as_tensor(num_samples, dtype=torch.float, device=self.sample_device)
        if is_dist_avail_and_initialized():
            torch.distributed.all_reduce(num_boxes)
        num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()
        return num_boxes

    def forward(self, outputs):
        # losses of each frame are calculated during the model's forwarding and are outputted by the model as outputs['losses_dict].
        losses = outputs.pop("losses_dict")
        num_samples = self.get_num_boxes(self.num_samples)
        for loss_name, loss in losses.items():
            losses[loss_name] = losses[loss_name] /num_samples
        return losses

class RuntimeTrackerBase(object):
    def __init__(self):
        self.det_score_thresh = 0.7
        self.track_score_thresh = 0.7
        self.iou_thresh = 0.0
        self.miss_tolerance = 10
        self.max_obj_id = 0

        self.frame_count = 0

    def update(self, frame_res: dict):
        track_instances: Instances = frame_res['track_instances']
        track_gt_ids = track_instances.obj_idxes
        device = track_gt_ids.device

        # step1. find new-born det
        det_scores = frame_res['pred_det_logits'][0, :, 0].sigmoid() #(300,)
        track_scores = frame_res['pred_track_logits'][0, :, 0].sigmoid() #(num_tracklets, )
        det_src_ids = (det_scores >= self.det_score_thresh).nonzero(as_tuple=True)[0]
        track_src_ids = (track_scores >= self.track_score_thresh).nonzero(as_tuple=True)[0]

        # filter duplicate tracklets
        track_boxes = frame_res['pred_track_boxes'][0, track_src_ids]
        iou, _ = box_iou(box_cxcywh_to_xyxy(track_boxes), box_cxcywh_to_xyxy(track_boxes), clamp=True)
        iou = torch.triu(iou, diagonal=1)
        _, duplicate_track_ids = (iou >= 0.9).nonzero(as_tuple=True)
        duplicate_track_ids = torch.unique(duplicate_track_ids)
        filter_track_ids = list(set(range(len(track_src_ids))) - set(duplicate_track_ids.tolist()))

        track_instances.disappear_time += 1
        track_instances.disappear_time[track_src_ids[filter_track_ids]] = 0
        filter_track_ids_o = torch.zeros(len(track_gt_ids), dtype=torch.bool, device=device)
        filter_track_ids_o[track_src_ids[filter_track_ids]] = True
        active_tracks = (track_instances.disappear_time < self.miss_tolerance) & filter_track_ids_o
        active_tracks = active_tracks.nonzero(as_tuple=True)[0]

        det_boxes = frame_res['pred_det_boxes'][0, det_src_ids]
        iou, _ = box_iou(box_cxcywh_to_xyxy(det_boxes), box_cxcywh_to_xyxy(frame_res['pred_track_boxes'][0]), clamp=True)
        duplicate_det_ids, track_ids = (iou >= self.iou_thresh).nonzero(as_tuple=True)
        new_det_obj = list(set(range(len(det_src_ids))) - set(duplicate_det_ids.tolist()))
        new_det_gt_ids = torch.arange(self.max_obj_id, self.max_obj_id + len(new_det_obj), device=device)
        self.max_obj_id += len(new_det_obj)

        

        # step2. update new matching result.
        new_track_instances = generate_empty_tracks(num_tracks=len(active_tracks)+len(new_det_obj), device=device)
        new_track_instances.track_embed = torch.cat([frame_res['hs_track'][0, active_tracks], 
                                                     frame_res['hs_det'][0, det_src_ids[new_det_obj]]], dim=0)
        new_track_instances.boxes = torch.cat([frame_res['pred_track_boxes'][0, active_tracks], 
                                               frame_res['pred_det_boxes'][0, det_src_ids[new_det_obj]]], dim=0)
        new_track_instances.obj_idxes = torch.cat([track_gt_ids[active_tracks], new_det_gt_ids], dim=0)
        new_track_instances.disappear_time[:len(active_tracks)] = track_instances.disappear_time[active_tracks]

        frame_res['pred_det_logits'] = det_scores[det_src_ids[new_det_obj]]
        frame_res['pred_det_boxes'] = det_boxes[new_det_obj]
        frame_res['pred_track_logits'] = track_scores[active_tracks]
        frame_res['pred_track_boxes'] = frame_res['pred_track_boxes'][0, active_tracks]
        frame_res['det_gt_idx'] = new_det_gt_ids
        frame_res['track_gt_idx'] = track_gt_ids[active_tracks]
        self.frame_count += 1
        return new_track_instances

def generate_empty_tracks(num_tracks, device, hidden_dim=256):
    track_instances = Instances((1, 1))
    track_instances.track_embed = torch.zeros((num_tracks, hidden_dim), device=device)
    track_instances.boxes = torch.zeros((num_tracks, 4), device=device)
    track_instances.obj_idxes = torch.full((num_tracks,), -1, dtype=torch.long, device=device) # maintain gt list
    track_instances.disappear_time = torch.zeros((num_tracks,), dtype=torch.long, device=device)
    track_instances.iou = torch.zeros((num_tracks,), dtype=torch.float, device=device)
    track_instances.track_scores = torch.zeros((num_tracks,), dtype=torch.float, device=device)

    return track_instances.to(device)

def build(args):
    dataset_to_num_classes = {
        'coco': 91,
        'coco_panoptic': 250,
        'e2e_mot': 1,
        'e2e_dance': 1,
        'e2e_joint': 1,
        'e2e_static_mot': 1,
    }
    assert args.dataset_file in dataset_to_num_classes
    num_classes = dataset_to_num_classes[args.dataset_file]
    device = torch.device(args.device)

    backbone = build_backbone(args)

    transformer = build_deforamble_transformer(args)

    img_matcher = build_matcher(args)
    num_frames_per_batch = max(args.sampler_lengths)
    weight_dict = {
            'ce': args.cls_loss_coef,
            'bbox': args.bbox_loss_coef,
            'giou': args.giou_loss_coef,
        }

    # TODO this is a hack
    if args.aux_loss:
        for i in range(num_frames_per_batch):
            for j in range(args.dec_layers - 1):
                weight_dict.update({"frame_{}_aux{}_loss_ce".format(i, j): args.cls_loss_coef,
                                    'frame_{}_aux{}_loss_bbox'.format(i, j): args.bbox_loss_coef,
                                    'frame_{}_aux{}_loss_giou'.format(i, j): args.giou_loss_coef,
                                    })

    losses = args.losses
    criterion = ClipMatcher(num_classes, matcher=img_matcher, weight_dict=weight_dict, losses=losses)
    criterion.to(device)
    postprocessors = {}
    model = MOMOT(
        backbone,
        transformer,
        num_feature_levels=args.num_feature_levels,
        num_classes=num_classes,
        num_queries=args.num_queries,
        aux_loss=args.aux_loss,
        criterion=criterion,
        with_box_refine=args.with_box_refine,
        two_stage=args.two_stage,
        use_checkpoint=args.use_checkpoint,
        visibilty_thresh=args.visibility_thresh,
    )
    return model, criterion, postprocessors