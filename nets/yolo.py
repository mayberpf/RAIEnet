import torch
import torch.nn as nn
import time
from nets.ConvNext import ConvNeXt_Small, ConvNeXt_Tiny
from nets.CSPdarknet import C3, Conv, CSPDarknet,DW_Conv
from nets.Swin_transformer import Swin_transformer_Tiny
import torch.nn.functional as F
# from ConvNext import ConvNeXt_Small, ConvNeXt_Tiny
# from CSPdarknet import C3, Conv, CSPDarknet,DW_Conv
# from Swin_transformer import Swin_transformer_Tiny
# from fastai.vision.all import *
import pdb 

#---------------------------------------------------#
#   add_model
#---------------------------------------------------#
class unetUp(nn.Module):
    def __init__(self, in_size, out_size):
        super(unetUp, self).__init__()
        self.conv1  = nn.Conv2d(in_size, out_size, kernel_size = 3, padding = 1)
        self.conv2  = nn.Conv2d(out_size, out_size, kernel_size = 3, padding = 1)
        self.up     = nn.UpsamplingBilinear2d(scale_factor = 2)
        self.relu   = nn.SiLU(inplace = True)

    def forward(self, inputs1, inputs2):
        outputs = torch.cat([inputs1, self.up(inputs2)], 1)
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)
        return outputs

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.d = dimension

    def forward(self, x):
        """ print("***********************")
        for f in x:
            print(f.shape) """
        return torch.cat(x, self.d)

class Neck(nn.Module):
    def __init__(self) :
        super().__init__()
        self.neck1  = nn.Sequential(
            Conv( 512, 256, k=1, s=1, p=None, g=1, act=True),
            C3(256, 512, n=1, shortcut=True, g=1, e=0.5)
        )
        self.neck2  = nn.Sequential(
            Conv( 768, 512, k=1, s=1, p=None, g=1, act=True),
            C3(512, 256, n=1, shortcut=True, g=1, e=0.5)
        )
        self.neck3  = nn.Sequential(
            Conv( 384, 256, k=1, s=1, p=None, g=1, act=True),
            C3(256, 128, n=1, shortcut=True, g=1, e=0.5)
        )
        self.neck4  = nn.Sequential(
            Conv( 128, 128, k=1, s=1, p=None, g=1, act=True),
            C3(128, 128, n=1, shortcut=True, g=1, e=0.5)
        )
        self.neck5  = nn.Sequential(
            Conv( 384, 256, k=1, s=1, p=None, g=1, act=True),
            C3(256, 256, n=1, shortcut=True, g=1, e=0.5)
        )
        self.neck6  = nn.Sequential(
            Conv( 768, 512, k=1, s=1, p=None, g=1, act=True),
            C3(512, 512, n=1, shortcut=True, g=1, e=0.5)
        )

        # self.conv_1 = DW_Conv(128,128,3,2,2)
        # self.conv_2 = DW_Conv(256,256,3,2,2)
        self.conv_1 = Conv(128,128,3,2)
        self.conv_2 = Conv(256,256,3,2)
        self.upsample = nn.Upsample(scale_factor=2,mode='nearest')
        self.cat = Concat()
        # self.c3 = C3()
    def forward(self,feat1,feat2,feat3):

        # pdb.set_trace()
        # print("feat3在卷积前的输出:",feat3.shape)
        feat3 = self.neck1(feat3)
        # print("feat3在卷积前的输出:",feat3.shape)
        x_2 = self.upsample(feat3)
        x_2 = self.cat([feat2,x_2])
        feat2 = self.neck2(x_2)
        x_1 = self.upsample(feat2)
        x_1 = self.cat([feat1,x_1])
        feat1 = self.neck3(x_1)

        #=============================PAN结构=================#
        # feat1 = self.neck4(feat1)
        # #下采样

        # # print("在卷积前的输出:",feat1.shape)
        # x_1 = self.conv_1(feat1)
        # # print("在卷积后的输出:",x_1.shape)
        # x_1 = self.cat([feat2,x_1])
        # feat2 = self.neck5(x_1)
        # x_2 = self.conv_2(feat2)
        # x_2 = self.cat([feat3,x_2])
        # feat3 = self.neck6(x_2)
        #=============================PAN结构=================#
        # print("在卷积后的输出:",x_1.shape)
        # print(feat1.shape)
        # print(feat2.shape)
        # print(feat3.shape)
        # pdb.set_trace()
        
        return feat1,feat2,feat3


#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#
class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, backbone='cspdarknet', pretrained=False, input_shape=[640, 640]):
        super(YoloBody, self).__init__()
        depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 32
        base_depth          = max(round(dep_mul * 3), 1)  # 1
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        #-----------------------------------------------#
        self.backbone_name  = backbone
        if backbone == "cspdarknet":
            #---------------------------------------------------#   
            #   生成CSPdarknet53的主干模型
            #   获得三个有效特征层，他们的shape分别是：
            #   80,80,256
            #   40,40,512
            #   20,20,1024
            #---------------------------------------------------#
            self.backbone   = CSPDarknet(base_channels, base_depth, phi, pretrained)
        else:
            #---------------------------------------------------#   
            #   如果输入不为cspdarknet，则调整通道数
            #   使其符合YoloV5的格式
            #---------------------------------------------------#
            self.backbone       = {
                'convnext_tiny'         : ConvNeXt_Tiny,
                'convnext_small'        : ConvNeXt_Small,
                'swin_transfomer_tiny'  : Swin_transformer_Tiny,
            }[backbone](pretrained=pretrained, input_shape=input_shape)
            in_channels         = {
                'convnext_tiny'         : [192, 384, 768],
                'convnext_small'        : [192, 384, 768],
                'swin_transfomer_tiny'  : [192, 384, 768],
            }[backbone]
            feat1_c, feat2_c, feat3_c = in_channels 
            self.conv_1x1_feat1 = Conv(feat1_c, base_channels * 4, 1, 1)
            self.conv_1x1_feat2 = Conv(feat2_c, base_channels * 8, 1, 1)
            self.conv_1x1_feat3 = Conv(feat3_c, base_channels * 16, 1, 1)

#这里就是YOLO的一些neck和head
            
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1    = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        # 80, 80, 256 => 80, 80, 3 * (5 + num_classes) => 80, 80, 3 * (4 + 1 + num_classes)
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * (5 + num_classes) => 40, 40, 3 * (4 + 1 + num_classes)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes) => 20, 20, 3 * (4 + 1 + num_classes)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)


    def forward(self, x):
        #  backbone
        feat1, feat2, feat3 = self.backbone(x)

        if self.backbone_name != "cspdarknet":
            feat1 = self.conv_1x1_feat1(feat1)
            feat2 = self.conv_1x1_feat2(feat2)
            feat3 = self.conv_1x1_feat3(feat3)

        # 20, 20, 1024 -> 20, 20, 512
        P5          = self.conv_for_feat3(feat3)
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)
        # 40, 40, 512 -> 40, 40, 1024
        P4          = torch.cat([P5_upsample, feat2], 1)
        # 40, 40, 1024 -> 40, 40, 512
        P4          = self.conv3_for_upsample1(P4)

        # 40, 40, 512 -> 40, 40, 256
        P4          = self.conv_for_feat2(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 80, 80, 512 -> 80, 80, 256
        P3          = self.conv3_for_upsample2(P3)
        
        # 80, 80, 256 -> 40, 40, 256
        P3_downsample = self.down_sample1(P3)
        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)
        # 40, 40, 512 -> 40, 40, 512
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 512 -> 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5 = torch.cat([P4_downsample, P5], 1)
        # 20, 20, 1024 -> 20, 20, 1024
        P5 = self.conv3_for_downsample2(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,80,80)
        #---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,40,40)
        #---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,20,20)
        #---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)
        return out0, out1, out2
#---------------------------------------------------#
#   yolo_body
#---------------------------------------------------#

class YoloV_body(nn.Module):
    def __init__(self, anchors_mask, num_classes, phi, backbone='cspdarknet', pretrained=False, input_shape=[640, 640]):
        super(YoloV_body, self).__init__()
        depth_dict          = {'s' : 0.33, 'm' : 0.67, 'l' : 1.00, 'x' : 1.33,}
        width_dict          = {'s' : 0.50, 'm' : 0.75, 'l' : 1.00, 'x' : 1.25,}
        dep_mul, wid_mul    = depth_dict[phi], width_dict[phi]

        base_channels       = int(wid_mul * 64)  # 32
        base_depth          = max(round(dep_mul * 3), 1)  # 1
        #-----------------------------------------------#
        #   输入图片是640, 640, 3
        #   初始的基本通道是64
        #-----------------------------------------------#
        self.backbone_name  = backbone
        if backbone == "cspdarknet":
            #---------------------------------------------------#   
            #   生成CSPdarknet53的主干模型
            #   获得三个有效特征层，他们的shape分别是：
            #   80,80,256
            #   40,40,512
            #   20,20,1024
            #---------------------------------------------------#
            self.backbone   = CSPDarknet(base_channels, base_depth, phi, pretrained)
        else:
            #---------------------------------------------------#   
            #   如果输入不为cspdarknet，则调整通道数
            #   使其符合YoloV5的格式
            #---------------------------------------------------#
            self.backbone       = {
                'convnext_tiny'         : ConvNeXt_Tiny,
                'convnext_small'        : ConvNeXt_Small,
                'swin_transfomer_tiny'  : Swin_transformer_Tiny,
            }[backbone](pretrained=pretrained, input_shape=input_shape)
            in_channels         = {
                'convnext_tiny'         : [192, 384, 768],
                'convnext_small'        : [192, 384, 768],
                'swin_transfomer_tiny'  : [192, 384, 768],
            }[backbone]
            feat1_c, feat2_c, feat3_c = in_channels 
            self.conv_1x1_feat1 = Conv(feat1_c, base_channels * 4, 1, 1)
            self.conv_1x1_feat2 = Conv(feat2_c, base_channels * 8, 1, 1)
            self.conv_1x1_feat3 = Conv(feat3_c, base_channels * 16, 1, 1)

#这里就是YOLO的一些neck和head
            
        self.upsample   = nn.Upsample(scale_factor=2, mode="nearest")

        self.conv_for_feat3         = Conv(base_channels * 16, base_channels * 8, 1, 1)
        self.conv3_for_upsample1    = C3(base_channels * 16, base_channels * 8, base_depth, shortcut=False)

        self.conv_for_feat2         = Conv(base_channels * 8, base_channels * 4, 1, 1)
        self.conv3_for_upsample2    = C3(base_channels * 8, base_channels * 4, base_depth, shortcut=False)

        self.down_sample1           = Conv(base_channels * 4, base_channels * 4, 3, 2)
        self.conv3_for_downsample1  = C3(base_channels * 8, base_channels * 8, base_depth, shortcut=False)

        self.down_sample2           = Conv(base_channels * 8, base_channels * 8, 3, 2)
        self.conv3_for_downsample2  = C3(base_channels * 16, base_channels * 16, base_depth, shortcut=False)

        # 80, 80, 256 => 80, 80, 3 * (5 + num_classes) => 80, 80, 3 * (4 + 1 + num_classes)
        self.yolo_head_P3 = nn.Conv2d(base_channels * 4, len(anchors_mask[2]) * (5 + num_classes), 1)
        # 40, 40, 512 => 40, 40, 3 * (5 + num_classes) => 40, 40, 3 * (4 + 1 + num_classes)
        self.yolo_head_P4 = nn.Conv2d(base_channels * 8, len(anchors_mask[1]) * (5 + num_classes), 1)
        # 20, 20, 1024 => 20, 20, 3 * (5 + num_classes) => 20, 20, 3 * (4 + 1 + num_classes)
        self.yolo_head_P5 = nn.Conv2d(base_channels * 16, len(anchors_mask[0]) * (5 + num_classes), 1)

        self.neck_fpn = Neck()
        # self.neck = Conv(256,512,7,1,3)
        self.neck = Conv(256,512)
        # self.obj_mask = nn.Sequential(
        #     #Conv 7
        #     nn.ConvTranspose2d(512, 384, kernel_size=1, stride=1), 
        #     nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        #     nn.ConvTranspose2d(384,256, kernel_size=1, stride=1), 
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.ConvTranspose2d(256,1 ,kernel_size=2, stride=2),
        #     # nn.Sigmoid()
        # )
        self.obj_mask = nn.Sequential(
            #Conv 7
            nn.ConvTranspose2d(512, 256, kernel_size=1, stride=1), 
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            # nn.ConvTranspose2d(256,1, kernel_size=1, stride=1), 
            # nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ConvTranspose2d(256,1 ,kernel_size=2, stride=2),
            # nn.Sigmoid()
        )
        self.Unet_1 = unetUp(768,256)
        self.Unet_2 = unetUp(384,256)
        # self.Unet_3 = nn.Sequential(
        #     nn.UpsamplingBilinear2d(scale_factor = 2),
        #     nn.Conv2d(256, 128, kernel_size = 3, padding = 1),
        #     nn.Conv2d(128, 128, kernel_size = 3, padding = 1),
        #     nn.ReLU(inplace = True),
        # )
        # self.Unet_4 = nn.Sequential(
        #     nn.ConvTranspose2d(128,128,kernel_size=1,stride=1),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(128,1,kernel_size=1,stride=1),
        #     nn.ReLU(inplace=True)
        # )
        # self.last = nn.Sigmoid()
        # self.vp = nn.Sequential(
        #     #Conv 7
        #     #nn.Conv2d(4096, 4096, kernel_size=1, stride=1, padding=0), 
        #     #nn.Dropout(),
        #     #Conv 8
        #     nn.ConvTranspose2d(512, 384, kernel_size=1, stride=1), 
        #     nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
        #     nn.ConvTranspose2d(384,256, kernel_size=1, stride=1), 
        #     nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        #     nn.ConvTranspose2d(256,5 ,kernel_size=2, stride=2),
        #     nn.Sigmoid()
        # )

    def forward(self, x):
        #  backbone
        neck_1, neck_2, neck_3 = self.backbone(x)
        up_2 = self.Unet_1(neck_2,neck_3)
        # print(up_2.shape)
        up_3 = self.Unet_2(neck_1,up_2)
        # print(up_3.shape)
        # mask_ = self.Unet_3(up_3)
        # pdb.set_trace()
        # print(up_3.shape)
        mask_ = self.neck(up_3)
        # print(mask_.shape)

        mask_out = self.obj_mask(mask_)

        # print(mask_.shape)
        # mask_out = self.Unet_4(mask_)
        # print(mask_.shape)
        # mask_ = F.interpolate(mask_,scale_factor=2,mode='bilinear')
        # print(mask_.shape)
        # mask_out = self.last(mask_)
        # mask_ = nn.Sigmoid()
        # print(feat1.shape,feat2.shape,feat3.shape)
        feat1,feat2,feat3 = self.neck_fpn(neck_1, neck_2, neck_3)
        # print(feat1.shape,feat2.shape,feat3.shape)
        # mask_out = self.neck(feat1)
        # print(f"融合网络加了一层1*1conv后的shape为{out3.shape}")
        # vp = self.vp(vp_out)  
        # print(mask_out.shape)
        # mask_pred = self.obj_mask(mask_out)
        # print(mask_pred.shape)

        if self.backbone_name != "cspdarknet":
            feat1 = self.conv_1x1_feat1(feat1)
            feat2 = self.conv_1x1_feat2(feat2)
            feat3 = self.conv_1x1_feat3(feat3)

        # 20, 20, 1024 -> 20, 20, 512
        P5          = self.conv_for_feat3(feat3)
        # 20, 20, 512 -> 40, 40, 512
        P5_upsample = self.upsample(P5)
        # 40, 40, 512 -> 40, 40, 1024
        P4          = torch.cat([P5_upsample, feat2], 1)
        # 40, 40, 1024 -> 40, 40, 512
        P4          = self.conv3_for_upsample1(P4)

        # 40, 40, 512 -> 40, 40, 256
        P4          = self.conv_for_feat2(P4)
        # 40, 40, 256 -> 80, 80, 256
        P4_upsample = self.upsample(P4)
        # 80, 80, 256 cat 80, 80, 256 -> 80, 80, 512
        P3          = torch.cat([P4_upsample, feat1], 1)
        # 80, 80, 512 -> 80, 80, 256
        P3          = self.conv3_for_upsample2(P3)
        
        # 80, 80, 256 -> 40, 40, 256
        P3_downsample = self.down_sample1(P3)
        # 40, 40, 256 cat 40, 40, 256 -> 40, 40, 512
        P4 = torch.cat([P3_downsample, P4], 1)
        # 40, 40, 512 -> 40, 40, 512
        P4 = self.conv3_for_downsample1(P4)

        # 40, 40, 512 -> 20, 20, 512
        P4_downsample = self.down_sample2(P4)
        # 20, 20, 512 cat 20, 20, 512 -> 20, 20, 1024
        P5 = torch.cat([P4_downsample, P5], 1)
        # 20, 20, 1024 -> 20, 20, 1024
        P5 = self.conv3_for_downsample2(P5)

        #---------------------------------------------------#
        #   第三个特征层
        #   y3=(batch_size,75,80,80)
        #---------------------------------------------------#
        out2 = self.yolo_head_P3(P3)
        #---------------------------------------------------#
        #   第二个特征层
        #   y2=(batch_size,75,40,40)
        #---------------------------------------------------#
        out1 = self.yolo_head_P4(P4)
        #---------------------------------------------------#
        #   第一个特征层
        #   y1=(batch_size,75,20,20)
        #---------------------------------------------------#
        out0 = self.yolo_head_P5(P5)
        # print(out0.shape)
        # print(out1.shape)
        # print(out2.shape)
        return out0, out1, out2,mask_out






# if __name__ =='__main__':


#     anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
#     num_classes = 11
#     phi = 's'
#     backbone = 'cspdarknet' 
#     pretrained=False
#     input_shape=[640,640]

#     img = torch.randn(1,3,640,640)
#     net = YoloV_body(anchors_mask,num_classes,phi,backbone,pretrained,input_shape)
#     import time

#     start = time.time()

#     output = net(img)
#     end = time.time()
#     consume_time = end - start
#     print(consume_time)
#     # print(len(output))
#     yolo_out = output[:3]
#     mask_pred = output[3]
#     # vp_pred = output[4]
#     print(f'mask_pred的shape为{mask_pred.shape}')
#     print(mask_pred[0][0].max()) 
#     # print(f'vp_pred的shape为{vp_pred.shape}')
