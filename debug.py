import torch
import numpy as np
import copy
from detectron2.utils.events import EventStorage, get_event_storage
import matplotlib.pyplot as plt
import sys
sys.path.append('DeformTrack')
from util.box_ops import box_cxcywh_to_xyhw
import cv2
import random

class Debug():
    def __init__(self):
        self.outputs = []
        self.hooks = None
        self.plot_gt = False
        
    def register_hooks(self, model):
        # if multi-GPU replace model.module by model_
        model_ = model.module if hasattr(model, 'module') else model
        self.hooks = [
            #model_.criterion.register_forward_hook(
            #    # pick up pred mask found by HungarianMatcher -> pick up debug_idx pred mask
            #    lambda module, input, output: self.outputs.append(input[0]) 
            #),
            model_.transformer.decoder.layers[0].temporal_cross_attn.register_forward_hook(
                lambda module, input, output: self.outputs.append(module.viz)
            )
        ]
    def viz_attn(self, org_img, boxes=None, identities=None, frame_id=0):
    
        #input 0, 1, 2:
            # ref: N, num_tracks, n_heads, n_levels, n_points, 2
            # sample_loc: N, num_tracks, n_heads, n_levels, n_points, 2
            # attn_weights: N, num_tracks*n_head, 1, n_level*n_point
        def lookup_id(ids, identities=None):
            if ids == 'all':
                return torch.arange(0, self.outputs[0][0].shape[1])
            if identities is not None:
                ret_id = []
                for id in ids:
                    if id in identities:
                        ret_id.append((id==identities).nonzero(as_tuple=True)[0].item())
                return ret_id

        draw_bbox_ids = [3,16]
        viz_ids = [14]

        if self.outputs[0] is None:
            self.outputs = []
            return         
            
        # draw bboxes
        if boxes is not None and identities is not None:
            img = draw_bboxes(org_img.numpy(), boxes[lookup_id(draw_bbox_ids, identities)], identities[lookup_id(draw_bbox_ids, identities)])

        # draw temp attn weights
        h, w = img.shape[:2]
        fig, ax = plt.subplots()
        ax.imshow(img)
        
        ref_pts = self.outputs[0][0].cpu()
        sample_loc = self.outputs[0][1].flatten(3, 4).cpu()
        _, num_tracks, n_heads, _ , _ =  sample_loc.shape
        attn_weights = self.outputs[0][2].cpu().reshape(num_tracks, n_heads, -1)[[-4], :]

        attn_max = np.argmax(attn_weights, axis=-1)
        x, y = ref_pts[0, [-3], 0, 0]*w, ref_pts[0, [-3], 0, 1]*h
        plt.scatter(x, y, s=15, alpha=1, marker="+", c="green")
        sample_loc_x, sample_loc_y = sample_loc[0, [-3], ..., 0]*w, sample_loc[0, [-3], ..., 1]*h
        cm = plt.cm.get_cmap('bwr')
        plt.scatter(torch.gather(sample_loc_x, 2, attn_max[...,None]), torch.gather(sample_loc_y, 2, attn_max[...,None]), c=attn_max[...,None], cmap=cm, s=8)
        
        plt.axis('off')
        fig.savefig('demo/{}.png'.format(frame_id), bbox_inches='tight',pad_inches = 0, dpi=384)
        plt.clf()
        self.outputs = []

        
    def __call__(self):

        storage = get_event_storage()
        for hook in self.hooks:
            hook.remove()
        org_imgs = storage._debug_image['org_imgs']
        storage._debug_figure = self.plot_pred_box(org_imgs, self.outputs[0])
        storage._debug_image = {}

    @staticmethod
    def plot_pred_box(org_images, outputs, out_dir=None, plot_mode='tensorboard', plot_gt=False):
        '''
        pred_mask: can be flatten mask(N, 1, H*W) or processed mask(N, H, W)
        '''
        img_dict = {}
        for i, (_org_img, det_logits_frame,   \
                         det_boxes_frame,    \
                         track_logits_frame, \
                         track_boxes_frame,  \
                         det_gt_idx,         \
                         track_gt_idx        \
                         ) in enumerate(zip(org_images, outputs['pred_det_logits'], 
                                                        outputs['pred_det_boxes'],
                                                        outputs['pred_track_logits'],
                                                        outputs['pred_track_boxes'],
                                                        outputs['det_gt_idx'],
                                                        outputs['track_gt_idx'])):

            det_gt = det_gt_idx.clone().detach().cpu()
            track_gt = track_gt_idx.clone().detach().cpu()
            det_logits = det_logits_frame.clone().detach().cpu()
            det_boxes = det_boxes_frame.clone().detach().cpu()
            track_logits = track_logits_frame.clone().detach().cpu()
            track_boxes = track_boxes_frame.clone().detach().cpu()
            org_img = _org_img.clone().detach().cpu()
            if org_img.shape[0] == 3:
                org_img = org_img.permute(1, 2, 0)

            h, w = org_img.shape[:2]

            if plot_gt:
                gt_instnace = outputs['gt_instance'][i]
                gt_boxes = gt_instnace.gt_boxes.tensor.clone().detach().cpu()
                
                org_img_gt = copy.deepcopy(org_img)
                fig_gt, ax_gt = plt.subplots()
                
                ax_gt.imshow(org_img_gt)
                gt_boxes = box_cxcywh_to_xyhw(gt_boxes * torch.tensor([w, h, w, h]))
                if hasattr(gt_instnace, 'inst_id'):
                    gt_inst_id = gt_instnace.inst_id.clone().detach().cpu()
                    for box, gt_id in zip(gt_boxes, gt_inst_id):
                        ax_gt.add_patch(plt.Rectangle((box[0].item(), box[1].item()), box[2].item(), box[3].item(), 
                                                fill=False, edgecolor='red', linewidth=1))
                        ax_gt.text(box[0].item(), box[1].item(), gt_id.item(),fontsize=10, color='green')
                else:
                    for box in gt_boxes:
                        ax_gt.add_patch(plt.Rectangle((box[0].item(), box[1].item()), box[2].item(), box[3].item(), 
                                                fill=False, edgecolor='red', linewidth=1))
                img_dict['frame_gt_{}'.format(i)] = fig_gt

            
            fig, ax = plt.subplots()
            ax.imshow(org_img)
            det_boxes = box_cxcywh_to_xyhw(det_boxes * torch.tensor([w, h, w, h]))
            track_boxes = box_cxcywh_to_xyhw(track_boxes * torch.tensor([w, h, w, h]))

            for score, box, gt_id in zip(det_logits, det_boxes, det_gt):
                ax.add_patch(plt.Rectangle((box[0].item(), box[1].item()), box[2].item(), box[3].item(), 
                                            fill=False, edgecolor='red', linewidth=1))
                ax.text(box[0].item(), box[1].item(), gt_id.item()%10000, fontsize=10, color='red')

            for score, box, gt_id in zip(track_logits, track_boxes, track_gt):
                ax.add_patch(plt.Rectangle((box[0].item(), box[1].item()), box[2].item(), box[3].item(), 
                                            fill=False, edgecolor='blue', linewidth=1))
                ax.text(box[0].item(), box[1].item(), gt_id.item()%10000, fontsize=10, color='red')
            fig.tight_layout()
            img_dict['frame_pred_{}'.format(i)] = fig
            
            if plot_mode == 'image':
                fig.savefig(out_dir)
        return img_dict

    @staticmethod
    def plot_det_pred_box(org_images, outputs, out_dir=None, plot_mode='tensorboard'):
        '''
        pred_mask: can be flatten mask(N, 1, H*W) or processed mask(N, H, W)
        '''
        img_dict = {}
        for i, (_org_img, det_logits_frame,   \
                         det_boxes_frame,    
                         ) in enumerate(zip(org_images, outputs['pred_det_logits'], 
                                                        outputs['pred_det_boxes'])):
            det_logits = det_logits_frame.clone().detach().cpu()
            det_boxes = det_boxes_frame.clone().detach().cpu()
            org_img = _org_img.clone().detach().cpu()

            if org_img.shape[0] == 3:
                org_img = org_img.permute(1, 2, 0)

            h, w = org_img.shape[:2]

            fig, ax = plt.subplots()
            ax.imshow(org_img)
            det_boxes = box_cxcywh_to_xyhw(det_boxes * torch.tensor([w, h, w, h]))

            for score, box in zip(det_logits, det_boxes):
                ax.add_patch(plt.Rectangle((box[0].item(), box[1].item()), box[2].item(), box[3].item(), 
                                            fill=False, edgecolor='red', linewidth=1))

            fig.tight_layout()
            img_dict['frame_pred_{}'.format(i)] = fig
            
            if plot_mode == 'image':
                fig.savefig(out_dir)
        return img_dict

COLORS_10 = [(144, 238, 144), (178, 34, 34), (221, 160, 221), (0, 255, 0), (0, 128, 0), (210, 105, 30), (220, 20, 60),
             (192, 192, 192), (255, 228, 196), (50, 205, 50), (139, 0, 139), (100, 149, 237), (138, 43, 226),
             (238, 130, 238),
             (255, 0, 255), (0, 100, 0), (127, 255, 0), (255, 0, 255), (0, 0, 205), (255, 140, 0), (255, 239, 213),
             (199, 21, 133), (124, 252, 0), (147, 112, 219), (106, 90, 205), (176, 196, 222), (65, 105, 225),
             (173, 255, 47),
             (255, 20, 147), (219, 112, 147), (186, 85, 211), (199, 21, 133), (148, 0, 211), (255, 99, 71),
             (144, 238, 144),
             (255, 255, 0), (230, 230, 250), (0, 0, 255), (128, 128, 0), (189, 183, 107), (255, 255, 224),
             (128, 128, 128),
             (105, 105, 105), (64, 224, 208), (205, 133, 63), (0, 128, 128), (72, 209, 204), (139, 69, 19),
             (255, 245, 238),
             (250, 240, 230), (152, 251, 152), (0, 255, 255), (135, 206, 235), (0, 191, 255), (176, 224, 230),
             (0, 250, 154),
             (245, 255, 250), (240, 230, 140), (245, 222, 179), (0, 139, 139), (143, 188, 143), (255, 0, 0),
             (240, 128, 128),
             (102, 205, 170), (60, 179, 113), (46, 139, 87), (165, 42, 42), (178, 34, 34), (175, 238, 238),
             (255, 248, 220),
             (218, 165, 32), (255, 250, 240), (253, 245, 230), (244, 164, 96), (210, 105, 30)]

def plot_one_box(x, img, color=None, label=None, score=None, line_thickness=None):
    # Plots one bounding box on image img

    # tl = line_thickness or round(
    #     0.002 * max(img.shape[0:2])) + 1  # line thickness
    tl = 2
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1)  # filled
        cv2.putText(img,
                    label, (c1[0], c1[1] - 2),
                    0,
                    tl / 3, [225, 255, 255],
                    thickness=tf,
                    lineType=cv2.LINE_AA)
        if score is not None:
            cv2.putText(img, score, (c1[0], c1[1] + 30), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    return img


'''
deep sort 中的画图方法，在原图上进行作画
'''
def draw_bboxes(ori_img, bbox, identities=None, offset=(0, 0), cvt_color=False):
    if cvt_color:
        ori_img = cv2.cvtColor(np.asarray(ori_img), cv2.COLOR_RGB2BGR)
    img = ori_img
    for i, box in enumerate(bbox):
        x1, y1, x2, y2 = [int(i) for i in box[:4]]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        if len(box) > 4:
            score = '{:.2f}'.format(box[4])
        else:
            score = None
        # box text and bar
        id = int(identities[i]) if identities is not None else 0
        color = COLORS_10[id % len(COLORS_10)]
        label = '{:d}'.format(id)
        # t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 2 , 2)[0]
        img = plot_one_box([x1, y1, x2, y2], img, color, label, score=score)
    return img

def inverse_sigmoid(x, eps=1e-5):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)
