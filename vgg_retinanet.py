#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2020/5/11 9:31
# @Author  : 白裕�?
# @File    : vgg_retinanet.py

import torch.nn as nn
import torch
import math
import numpy as np
import torch.utils.model_zoo as model_zoo
from torchvision.ops import nms
import torchvision.models as models

from . import loss
from .helper import BasicBlock, Bottleneck, BBoxTransform, ClipBoxes
from crfnet.utils.anchors import Anchors

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


def min_max_pool2d(x):
    max_x = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(x)
    min_x = min_pool2d(x)
    return torch.cat([max_x, min_x], dim=1)  # concatenate on channel


def min_max_pool2d_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] = int(shape[1] / 2)
    shape[2] = int(shape[2] / 2)
    shape[3] *= 2
    return tuple(shape)


def min_pool2d(x, padding=1):
    max_val = torch.max(x) + 1  # we gonna replace all zeros with that value
    # replace all 0s with very high numbers
    is_zero = torch.where(torch.eq(x, 0.), max_val, x)
    x = is_zero + x

    # execute pooling with 0s being replaced by a high number
    min_x = -nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2), padding=padding)(-x)

    # depending on the value we either substract the zero replacement or not
    is_result_zero = torch.where(torch.eq(min_x, max_val), max_val, min_x)
    min_x = min_x - is_result_zero

    return min_x  # concatenate on channel


def min_pool2d_output_shape(input_shape):
    shape = list(input_shape)
    shape[1] = int(shape[1] / 2)
    shape[2] = int(shape[2] / 2)
    return tuple(shape)


class PyramidFeatures(nn.Module):
    def __init__(self, C3_size, C4_size, C5_size, radar_layers=True, feature_size=256):
        super(PyramidFeatures, self).__init__()

        self.radar_layers = radar_layers
        # upsample C5 to get P5 from the FPN paper
        self.P5_1 = nn.Conv2d(C5_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P5_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P5_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P5 elementwise to C4
        self.P4_1 = nn.Conv2d(C4_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P4_upsampled = nn.Upsample(scale_factor=2, mode='nearest')
        self.P4_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # add P4 elementwise to C3
        self.P3_1 = nn.Conv2d(C3_size, feature_size, kernel_size=1, stride=1, padding=0)
        self.P3_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=1, padding=1)

        # "P6 is obtained via a 3x3 stride-2 conv on C5"
        self.P6 = nn.Conv2d(C5_size, feature_size, kernel_size=3, stride=2, padding=1)

        # "P7 is computed by applying ReLU followed by a 3x3 stride-2 conv on P6"
        self.P7_1 = nn.ReLU()
        self.P7_2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, stride=2, padding=1)

    def forward(self, inputs, radar_inputs):
        C3, C4, C5 = inputs
        radar_layers = radar_inputs

        P5_x = self.P5_1(C5)
        P5_upsampled_x = self.P5_upsampled(P5_x)
        P5_x = self.P5_2(P5_x)

        P4_x = self.P4_1(C4)
        P4_x = P5_upsampled_x + P4_x
        P4_upsampled_x = self.P4_upsampled(P4_x)
        P4_x = self.P4_2(P4_x)

        P3_x = self.P3_1(C3)
        P3_x = P3_x + P4_upsampled_x
        P3_x = self.P3_2(P3_x)

        P6_x = self.P6(C5)

        P7_x = self.P7_1(P6_x)
        P7_x = self.P7_2(P7_x)

        if self.radar_layers:
            R3 = radar_layers[2]
            R4 = radar_layers[3]
            R5 = radar_layers[4]
            R6 = radar_layers[5]
            R7 = radar_layers[6]
            P3_x = torch.cat([P3_x, R3], dim=1)
            P4_x = torch.cat([P4_x, R4], dim=1)
            P5_x = torch.cat([P5_x, R5], dim=1)
            P6_x = torch.cat([P6_x, R6], dim=1)
            P7_x = torch.cat([P7_x, R7], dim=1)
        return [P3_x, P4_x, P5_x, P6_x, P7_x]


class RegressionModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256):
        super(RegressionModel, self).__init__()

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)

        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)

        return out.contiguous().view(out.shape[0], -1, 4)


class ClassificationModel(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, prior=0.01, feature_size=256):
        super(ClassificationModel, self).__init__()

        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = self.output_act(out)

        # out is B x C x W x H, with C = n_classes + n_anchors
        out1 = out.permute(0, 2, 3, 1)

        batch_size, width, height, channels = out1.shape

        out2 = out1.view(batch_size, width, height, self.num_anchors, self.num_classes)

        return out2.contiguous().view(x.shape[0], -1, self.num_classes)


class VggMax(nn.Module):
    def __init__(self, backbone='vgg16', cfg=None):
        super(VggMax, self).__init__()

        # Read config variables
        self.fusion_blocks = cfg.fusion_blocks
        self.cfg = cfg

        self.block1_Image = nn.Sequential(
            nn.Conv2d(len(self.cfg.channels), int(64 * cfg.network_width), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(64 * cfg.network_width), int(64 * cfg.network_width), kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.block2_Image = nn.Sequential(
            nn.Conv2d(int(64 * cfg.network_width)+len(self.cfg.channels[3:]), int(128 * cfg.network_width), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(128 * cfg.network_width), int(128 * cfg.network_width), kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.block3_Image = nn.Sequential(
            nn.Conv2d(int(128 * cfg.network_width)+len(self.cfg.channels[3:]), int(256 * cfg.network_width), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(256 * cfg.network_width), int(256 * cfg.network_width), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(256 * cfg.network_width), int(256 * cfg.network_width), kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.block4_Image = nn.Sequential(
            nn.Conv2d(int(256 * cfg.network_width)+len(self.cfg.channels[3:]), int(512 * cfg.network_width), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(512 * cfg.network_width), int(512 * cfg.network_width), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(512 * cfg.network_width), int(512 * cfg.network_width), kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.block5_Image = nn.Sequential(
            nn.Conv2d(int(512 * cfg.network_width)+len(self.cfg.channels[3:]), int(1024 * cfg.network_width), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(1024 * cfg.network_width), int(1024 * cfg.network_width), kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(int(1024 * cfg.network_width), int(1024 * cfg.network_width), kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        # separate input
        if len(self.cfg.channels) > 3:
            image_input = x[:, :3, :, :]
            radar_input = x[:, 3:, :, :]
        else:
            image_input = x
            radar_input = None

        # Bock 0 Fusion
        if len(self.cfg.channels) > 3:
            if 0 in self.cfg.fusion_blocks:
                x = torch.cat([image_input, radar_input], dim=1)
            else:
                x = image_input
        else:
            x = image_input
        concat_0 = x

        # Block 1 - Image
        x = self.block1_Image(x)
        if self.cfg.pooling == 'maxmin':
            x = min_max_pool2d(x)
        else:
            x = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(x)

        block1_pool = x
        # Block 1 - Radar
        if len(self.cfg.channels) > 3:
            if self.cfg.pooling == 'min':
                y = min_pool2d(radar_input)
            elif self.cfg.pooling == 'maxmin':
                y = min_max_pool2d(radar_input)
            elif self.cfg.pooling == 'conv':
                y = nn.Conv2d(len(self.cfg.channels.shape[3:]), 64 * self.cfg.network_width,
                              kernel_size=3, stride=1, padding=1)(radar_input)
                y = nn.ReLU()(y)
            else:
                y = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(radar_input)

            rad_block1_pool = y
            # Concatenate Block 1 Radar to image
            if 1 in self.cfg.fusion_blocks:
                x = torch.cat([x, y], dim=1)

        concat_1 = x
        # Block2
        x = self.block2_Image(x)

        if self.cfg.pooling == 'maxmin':
            x = min_max_pool2d(x)
        else:
            x = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(x)

        block2_pool = x
        # Block 2 - Radar
        if len(self.cfg.channels) > 3:
            if self.cfg.pooling == 'min':
                y = min_pool2d(y)
            elif self.cfg.pooling == 'maxmin':
                y = min_max_pool2d(y)
            elif self.cfg.pooling == 'conv':
                y = nn.Conv2d(len(self.cfg.channels.shape[3:]), 64 * self.cfg.network_width,
                              kernel_size=3, stride=1, padding=1)(y)
                y = nn.ReLU()(y)
            else:
                y = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(y)

            rad_block2_pool = y
            # Concatenate Block 1 Radar to image
            if 2 in self.cfg.fusion_blocks:
                x = torch.cat([x, y], dim=1)

        concat_2 = x
        # Block 3
        x = self.block3_Image(x)

        if self.cfg.pooling == 'maxmin':
            x = min_max_pool2d(x)
        else:
            x = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(x)

        block3_pool = x
        # Block 3 - Radar
        if len(self.cfg.channels) > 3:
            if self.cfg.pooling == 'min':
                y = min_pool2d(y)
            elif self.cfg.pooling == 'maxmin':
                y = min_max_pool2d(y)
            elif self.cfg.pooling == 'conv':
                y = nn.Conv2d(len(self.cfg.channels.shape[3:]), 64 * self.cfg.network_width,
                              kernel_size=3, stride=1, padding=1)(y)
                y = nn.ReLU()(y)
            else:
                y = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(y)

            rad_block3_pool = y
            # Concatenate Block 1 Radar to image
            if 3 in self.cfg.fusion_blocks:
                x = torch.cat([x, y], dim=1)

        concat_3 = x
        # Block 4
        x = self.block4_Image(x)

        if self.cfg.pooling == 'maxmin':
            x = min_max_pool2d(x)
        else:
            x = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(x)

        block4_pool = x
        # Block 4 - Radar
        if len(self.cfg.channels) > 3:
            if self.cfg.pooling == 'min':
                y = min_pool2d(y)
            elif self.cfg.pooling == 'maxmin':
                y = min_max_pool2d(y)
            elif self.cfg.pooling == 'conv':
                y = nn.Conv2d(len(self.cfg.channels.shape[3:]), 64 * self.cfg.network_width,
                              kernel_size=3, stride=1, padding=1)(y)
                y = nn.ReLU()(y)
            else:
                y = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(y)

            rad_block4_pool = y
            # Concatenate Block 1 Radar to image
            if 4 in self.cfg.fusion_blocks:
                x = torch.cat([x, y], dim=1)

        concat_4 = x
        # Block 5
        x = self.block5_Image(x)

        if self.cfg.pooling == 'maxmin':
            x = min_max_pool2d(x)
        else:
            x = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(x)

        block5_pool = x
        # Block 5 - Radar
        if len(self.cfg.channels) > 3:
            if self.cfg.pooling == 'min':
                y = min_pool2d(y)
            elif self.cfg.pooling == 'maxmin':
                y = min_max_pool2d(y)
            elif self.cfg.pooling == 'conv':
                y = nn.Conv2d(len(self.cfg.channels.shape[3:]), 64 * self.cfg.network_width,
                              kernel_size=3, stride=1, padding=1)(y)
                y = nn.ReLU()(y)
            else:
                y = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(y)

            rad_block5_pool = y
            # Concatenate Block 1 Radar to image
            if 5 in self.cfg.fusion_blocks:
                x = torch.cat([x, y], dim=1)

        concat_5 = x
        layer_outputs = [concat_3, concat_4, concat_5]
        radar_layers = [rad_block1_pool, rad_block2_pool, rad_block3_pool, rad_block4_pool, rad_block5_pool]
        return {'layer_outputs': layer_outputs, 'radar_layers': radar_layers}


class VggRetinaNet(nn.Module):
    def __init__(self, pretrained=True, cfg=None, num_classes=None):
        super(VggRetinaNet, self).__init__()
        self.backbone = cfg.network
        if self.backbone == 'vgg16':
            self.vgg = models.vgg16(pretrained=True, progress=True)
        elif self.backbone == 'vgg19':
            self.vgg = models.vgg19(pretrained=True, progress=True)
        elif 'vgg-max' in self.backbone:
            self.vgg = VggMax(backbone=self.backbone, cfg=cfg)
            if pretrained:
                # 加载model，model是自己定义好的模�?
                vgg16 = models.vgg16(pretrained=True)

                # 读取参数
                pretrained_dict = vgg16.state_dict()
                model_dict = self.vgg.state_dict()

                # 将pretrained_dict里不属于model_dict的键剔除�?
                pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

                # 更新现有的model_dict
                model_dict.update(pretrained_dict)

                # 加载我们真正需要的state_dict
                self.vgg.load_state_dict(model_dict)
        else:
            raise ValueError("Backbone '{}' not recognized.".format(backbone))
            
        fpn_sizes = [self.vgg.block3_Image[4].out_channels + len(cfg.channels[3:]), 
                     self.vgg.block4_Image[4].out_channels + len(cfg.channels[3:]), 
                     self.vgg.block5_Image[4].out_channels + len(cfg.channels[3:])]
        self.fpn = PyramidFeatures(fpn_sizes[0], fpn_sizes[1], fpn_sizes[2])

        self.cfg = cfg
        
        self.num_classes = num_classes
        self.regressionModel = RegressionModel(256 + len(self.cfg.channels[3:]))
        self.classificationModel = ClassificationModel(256 + len(self.cfg.channels[3:]), num_classes=self.num_classes)

        self.clipBoxes = ClipBoxes()
        self.regressBoxes = BBoxTransform()

        self.anchors = Anchors()
        self.focalLoss = loss.FocalLoss()
        
        for m in self.modules():
           if isinstance(m, nn.Conv2d):
              n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
              m.weight.data.normal_(0, math.sqrt(2. / n))
           elif isinstance(m, nn.BatchNorm2d):
              m.weight.data.fill_(1)
              m.bias.data.zero_()

        prior = 0.01

        self.classificationModel.output.weight.data.fill_(0)
        self.classificationModel.output.bias.data.fill_(-math.log((1.0 - prior) / prior))

        self.regressionModel.output.weight.data.fill_(0)
        self.regressionModel.output.bias.data.fill_(0)
        
        self.freeze_bn()

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
  
    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            if len(inputs.size()) == 3:
              img_batch = inputs.unsqueeze(0)
            else:
              img_batch = inputs
        layer_outputs = self.vgg(img_batch)['layer_outputs']
        radar_outputs = self.vgg(img_batch)['radar_layers']
        try:
            if 'fpn' in self.backbone:
                if self.cfg.pooling == 'min':
                    radar_outputs.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(radar_outputs[-1]))
                    radar_outputs.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(radar_outputs[-1]))
                elif self.cfg.pooling == 'conv':
                    radar_outputs.append(nn.Conv2d(256 * self.cfg.network_width, 64 * self.cfg.network_width,
                                                   kernel_size=3, stride=2)(radar_outputs[-1]))
                    radar_outputs.append(nn.Conv2d(64 * self.cfg.network_width, 64 * self.cfg.network_width,
                                                   kernel_size=3, stride=2)(radar_outputs[-1]))
                else:
                    radar_outputs.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(radar_outputs[-1]))
                    radar_outputs.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))(radar_outputs[-1]))
        except Exception as e:
            radar_outputs = None
            raise e
        
        features = self.fpn(layer_outputs, radar_outputs)
        regression = torch.cat([self.regressionModel(feature) for feature in features], dim=1)
        classification = torch.cat([self.classificationModel(feature) for feature in features], dim=1)
        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            scores = torch.max(classification, dim=2, keepdim=True)[0]
            scores_over_thresh = (scores > 0.05)[0, :, 0]
            if scores_over_thresh.sum() == 0:
                # no boxes to NMS, just return
                return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

            classification = classification[:, scores_over_thresh, :]
            transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
            scores = scores[:, scores_over_thresh, :]

            anchors_nms_idx = nms(transformed_anchors[0, :, :], scores[0, :, 0], 0.3)

            nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)
            return [transformed_anchors[0, anchors_nms_idx, :], nms_scores, nms_class]
