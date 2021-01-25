
from mmcv import Config, DictAction
from mmdet.apis import multi_gpu_test, single_gpu_test,init_detector,show_result_pyplot,inference_detector



if __name__ == '__main__':
    #main()
    img='/home/qinghua/Downloads/mmdetection/data/coco/train2017/n008-2018-08-01-15-16-36-0400__CAM_FRONT__1533151603512404.jpg'
    radar_img=''
    device2='cpu'
    config='/home/qinghua/Downloads/mmdetection/work_dir/fcos_center-normbbox-centeronreg-giou_r50_caffe_fpn_gn-head_4x4_1x_coco.py'
    cfg = Config.fromfile(config)
    checkpoints='/home/qinghua/Downloads/mmdetection/work_dir/epoch_24.pth'
    model=init_detector(cfg,checkpoints,device=device2)
    results=inference_detector(model,img)
    show_result_pyplot(model,img,results)
    #print(model)
    ##result=inference_detector(model)