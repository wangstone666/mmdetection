import mmcv
import numpy as np

# img1='../data/coco/train2017/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg'
# img2='../data/coco/radar/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg'
# img3=mmcv.imread(img1)
# img4=mmcv.imread(img2)
# print(img3.shape,img4.shape)
# print(type(img3))
# img5 = np.concatenate((img3,img4),axis=2)
# print(img5.shape)
# mmcv.imshow(img5[:,:,3:6])

a=np.zeros((2,2,3))
b=np.ones((2,2,3))
print(a)
print(b)
c=np.concatenate((a,b),axis=0)
print(c.shape)
print(c)