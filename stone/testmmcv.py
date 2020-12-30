import mmcv
import numpy as np
from mmcv.parallel import DataContainer
img1='../data/coco/train2017/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg'
img2='../data/coco/radar/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg'
img3=mmcv.imread(img1)
img4=mmcv.imread(img2)



a=DataContainer(img3)
print(a.data.shape)

# print(img3.shape,img4.shape)
# print(type(img3))
# img5 = np.concatenate((img3,img4),axis=2)
# print(img5.shape)
# mmcv.imshow(img5[:,:,3:6])

# a=np.zeros((2,2,3))
# b=np.ones((2,2,3))
# print(a)
# print(b)
# c=np.concatenate((a,b),axis=0)
# print(c.shape)
# print(c)
import torch
d= torch.Tensor([[1,1,2,3],[2,2,2,2]])

print(d.shape)


# LoadImageFromFile(to_float32=False, color_type='color', file_client_args={'backend': 'disk'})
# 1 th composing (900, 1600, 6):
#
# ************************************************************
# ************************************************************
# LoadAnnotations(with_bbox=True, with_label=True, with_mask=False, with_seg=False)poly2mask=True)poly2mask={'backend': 'disk'})
# 2 th composing (900, 1600, 6):
#
# ************************************************************
# ************************************************************
# Resize(img_scale=[(1333, 800)], multiscale_mode=range, ratio_range=None, keep_ratio=True)bbox_clip_border=True)
# 3 th composing (750, 1333, 6):
#
# ************************************************************
# ************************************************************
# RandomFlip(flip_ratio=0.5)
# 4 th composing (750, 1333, 6):
#
# ************************************************************
# ************************************************************
# Normalize(mean=[103.53  116.28  123.675], std=[1. 1. 1.], to_rgb=False)
# 5 th composing (750, 1333, 3):
#
# ************************************************************
# ************************************************************
# Pad(size=None, size_divisor=32, pad_val=0)
# 6 th composing (768, 1344, 3):
#
# ************************************************************
# ************************************************************
# DefaultFormatBundle
# 7 th composing torch.Size([768, 1344]):
#
# ************************************************************
# ************************************************************
# Collect(keys=['img', 'gt_bboxes', 'gt_labels'], meta_keys=('filename', 'ori_filename', 'ori_shape', 'img_shape', 'pad_shape', 'scale_factor', 'flip', 'flip_direction', 'img_norm_cfg'))
# 8 th composing torch.Size([768, 1344]):
