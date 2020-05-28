import albumentations as albu
from albumentations.pytorch.transforms import ToTensor, ToTensorV2
import torch
import numpy as np
import cv2


def get_augmentation(phase, width=512, height=512, min_area=0., min_visibility=0.):
    list_transforms = []

    if phase == 'train':
        list_transforms.extend([
            # albu.LongestMaxSize(
            #     max_size=width, always_apply=True),
            # albu.PadIfNeeded(min_height=height, min_width=width,
            #                  always_apply=True, border_mode=0, value=[0, 0, 0]),
            albu.RandomResizedCrop(
                height=height, width=width, p=1),
            albu.Rotate(limit=(-20, 20), p=0.5),
            # Flip the input either horizontally, vertically or both horizontally and vertically.
            albu.Flip(),
            # Transpose the input by swapping rows and columns.
            albu.Transpose(),
            albu.OneOf([
                albu.RandomBrightnessContrast(brightness_limit=0.3,
                                              contrast_limit=0.3),
                albu.RandomGamma(gamma_limit=(50, 150)),
                albu.NoOp()
            ]),
            albu.OneOf([
                albu.RGBShift(r_shift_limit=20, b_shift_limit=15,
                              g_shift_limit=15),
                albu.HueSaturationValue(hue_shift_limit=5,
                                        sat_shift_limit=5),
                albu.NoOp()
            ]),
            # albu.ChannelShuffle(p=0.5),
            # albu.CoarseDropout(max_holes=8, max_height=30, max_width=30, p=0.3),
            # albu.CLAHE(p=0.5),
            # albu.GaussNoise(p=0.5),
            albu.MedianBlur(blur_limit=5, p=0.3)
            # # 水平翻转
            # albu.HorizontalFlip(p=0.5),
            # albu.VerticalFlip(p=0.5),
        ])
    if (phase == 'test' or phase == 'valid'):
        list_transforms.extend([
            albu.Resize(height=height, width=width)
        ])
    list_transforms.extend([
        albu.Normalize(mean=(0.485, 0.456, 0.406),
                       std=(0.229, 0.224, 0.225), p=1),
        ToTensorV2()
    ])
    if (phase == 'test'):
        return albu.Compose(list_transforms)
    return albu.Compose(list_transforms, bbox_params=albu.BboxParams(
        format='pascal_voc', min_area=min_area, min_visibility=min_visibility, label_fields=['category_id']))


def detection_collate(batch):
    imgs = [s['img'] for s in batch]
    annots = [s['bboxes'] for s in batch]
    labels = [s['category_id'] for s in batch]
    scales = [s['scale'] for s in batch]

    # 保证一个batch中标注框数量一样？？？？？？
    max_num_annots = max(len(annot) for annot in annots)
    annot_padded = np.ones((len(annots), max_num_annots, 5)) * -1

    if max_num_annots > 0:
        for idx, (annot, lab) in enumerate(zip(annots, labels)):
            if len(annot) > 0:
                annot_padded[idx, :len(annot), :4] = annot
                annot_padded[idx, :len(annot), 4] = lab

    imgs = torch.stack(imgs, 0)
    annot_padded = torch.FloatTensor(annot_padded)
    scales = torch.FloatTensor(scales)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, common_size=512):
        self.common_size = common_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        height, width, _ = image.shape
        if height > width:
            scale = self.common_size / height
            resized_height = self.common_size
            resized_width = int(width * scale)
        else:
            scale = self.common_size / width
            resized_height = int(height * scale)
            resized_width = self.common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((self.common_size, self.common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image
        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image), 'annot': torch.from_numpy(annots), 'scale': scale}


class ResizerTest(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, common_size=512):
        self.common_size = common_size

    def __call__(self, sample):
        image = sample['img']

        height, width, _ = image.shape
        if height > width:
            scale = self.common_size / height
            resized_height = self.common_size
            resized_width = int(width * scale)
        else:
            scale = self.common_size / width
            resized_height = int(height * scale)
            resized_width = self.common_size

        image = cv2.resize(image, (resized_width, resized_height))

        new_image = np.zeros((self.common_size, self.common_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        return {'img': torch.from_numpy(new_image), 'scale': scale}


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


class NormalizerTest(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image = sample['img']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std)}
