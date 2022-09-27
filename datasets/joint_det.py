# ------------------------------------------------------------------------
# Copyright (c) 2021 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------

"""
MOT dataset which returns image_id for evaluation.
"""
from pathlib import Path
import cv2
import numpy as np
import torch
import torch.utils.data
import os.path as osp
from PIL import Image, ImageDraw
import copy
import datasets.transforms as T
from models.structures import Instances


class DetMOTDetection:
    def __init__(self, args, data_txt_path: str, seqs_folder, dataset2transform):
        self.args = args
        self.dataset2transform = dataset2transform
        self.vis = args.vis

        with open(data_txt_path, 'r') as file:
            self.img_files = file.readlines()
            self.img_files = [osp.join(seqs_folder, x.strip()) for x in self.img_files]
            self.img_files = list(filter(lambda x: len(x) > 0, self.img_files))

        self.label_files = [(x.replace('images', 'labels_with_ids').replace('.png', '.txt').replace('.jpg', '.txt'))
                            for x in self.img_files]


    @staticmethod
    def _targets_to_instances(targets: dict, img_shape) -> Instances:
        gt_instances = Instances(tuple(img_shape))
        gt_instances.boxes = targets['boxes']
        gt_instances.labels = targets['labels']
        gt_instances.obj_ids = targets['obj_ids']
        gt_instances.area = targets['area']
        return gt_instances

    def _pre_single_frame(self, idx: int):
        img_path = self.img_files[idx]
        label_path = self.label_files[idx]
        if 'crowdhuman' in img_path:
            img_path = img_path.replace('.png', '.jpg')
        img = Image.open(img_path)
        targets = {}
        w, h = img._size
        assert w > 0 and h > 0, "invalid image {} with shape {} {}".format(img_path, w, h)
        if osp.isfile(label_path):
            labels0 = np.loadtxt(label_path, dtype=np.float32)[..., :6].reshape(-1, 6)

            # normalized cewh to pixel xyxy format
            labels = labels0.copy()
            labels[:, 2] = w * (labels0[:, 2] - labels0[:, 4] / 2)
            labels[:, 3] = h * (labels0[:, 3] - labels0[:, 5] / 2)
            labels[:, 4] = w * (labels0[:, 2] + labels0[:, 4] / 2)
            labels[:, 5] = h * (labels0[:, 3] + labels0[:, 5] / 2)
        else:
            raise ValueError('invalid label path: {}'.format(label_path))

        if 'crowdhuman' in img_path:
            targets['dataset'] = 'CrowdHuman'
        elif 'MOT17' in img_path:
            targets['dataset'] = 'MOT17'
        else:
            raise NotImplementedError()
        targets['boxes'] = []
        targets['area'] = []
        targets['iscrowd'] = []
        targets['labels'] = []
        targets['obj_ids'] = []
        targets['image_id'] = torch.as_tensor(idx)
        targets['size'] = torch.as_tensor([h, w])
        targets['orig_size'] = torch.as_tensor([h, w])
        for label in labels:
            targets['boxes'].append(label[2:6].tolist())
            targets['area'].append(label[4] * label[5])
            targets['iscrowd'].append(0)
            targets['labels'].append(0)
            targets['obj_ids'].append(-1)

        targets['area'] = torch.as_tensor(targets['area'])
        targets['iscrowd'] = torch.as_tensor(targets['iscrowd'])
        targets['labels'] = torch.as_tensor(targets['labels'])
        targets['obj_ids'] = torch.as_tensor(targets['obj_ids'])
        targets['boxes'] = torch.as_tensor(targets['boxes'], dtype=torch.float32).reshape(-1, 4)
        return img, targets

    def __getitem__(self, idx):
        img, target = self._pre_single_frame(idx)
        dataset_name = target['dataset']
        transform = self.dataset2transform[dataset_name]
        if transform is not None:
            img, target = transform([img], [target])
        target[0].pop('dataset')
        gt_instances = self._targets_to_instances(target[0], img[0].shape[1:3])
        return img[0], gt_instances

    def __len__(self):
        return len(self.img_files)

class DetMOTDetectionValidation(DetMOTDetection):
    def __init__(self, args, seqs_folder, dataset2transform):
        args.data_txt_path = args.val_data_txt_path
        super().__init__(args, seqs_folder, dataset2transform)

def make_transforms_for_mot17(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([400, 500, 600]),
                    T.FixedMotRandomCrop(384, 600),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            normalize,
        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def make_transforms_for_crowdhuman(image_set, args=None):

    normalize = T.MotCompose([
        T.MotToTensor(),
        T.MotNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    scales = [608, 640, 672, 704, 736, 768, 800, 832, 864, 896, 928, 960, 992]

    if image_set == 'train':
        return T.MotCompose([
            T.MotRandomHorizontalFlip(),
            T.FixedMotRandomShift(bs=1),
            T.MotRandomSelect(
                T.MotRandomResize(scales, max_size=1536),
                T.MotCompose([
                    T.MotRandomResize([400, 500, 600]),
                    T.FixedMotRandomCrop(384, 600),
                    T.MotRandomResize(scales, max_size=1536),
                ])
            ),
            normalize,

        ])

    if image_set == 'val':
        return T.MotCompose([
            T.MotRandomResize([800], max_size=1333),
            normalize,
        ])

    raise ValueError(f'unknown {image_set}')


def build_dataset2transform(args, image_set):
    mot17_train = make_transforms_for_mot17('train', args)
    mot17_test = make_transforms_for_mot17('val', args)

    crowdhuman_train = make_transforms_for_crowdhuman('train', args)
    dataset2transform_train = {'MOT17': mot17_train, 'CrowdHuman': crowdhuman_train}
    dataset2transform_val = {'MOT17': mot17_test, 'CrowdHuman': mot17_test}
    if image_set == 'train':
        return dataset2transform_train
    elif image_set == 'val':
        return dataset2transform_val
    else:
        raise NotImplementedError()


def build(image_set, args):
    root = Path(args.mot_path)
    assert root.exists(), f'provided MOT path {root} does not exist'
    dataset2transform = build_dataset2transform(args, image_set)
    if image_set == 'train':
        data_txt_path = args.data_txt_path_train
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform)
    if image_set == 'val':
        data_txt_path = args.data_txt_path_val
        dataset = DetMOTDetection(args, data_txt_path=data_txt_path, seqs_folder=root, dataset2transform=dataset2transform)
    return dataset