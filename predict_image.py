import matplotlib.pyplot as plt
import time
import glob
import cv2
import numpy as np
from PIL import Image
import colorsys
import os
import time
import pdb
import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont
from nets.yolo import YoloV_body
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox

class YOLO(object):
    _defaults = {
        "model_path"        : '/home/ktd/rpf_ws/yolov5-pytorch-main/12/last_epoch_weights.pth',
        "classes_path"      : './model_data/loadmark_classes.txt',
        "anchors_path"      : './model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        "input_shape"       : [640, 640],
        "backbone"          : 'cspdarknet',
        "phi"               : 's',
        "confidence"        : 0.2,
        "nms_iou"           : 0.3,
        "letterbox_image"   : True,
        "cuda"              : True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   初始化YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            self._defaults[name] = value 
            
        #---------------------------------------------------#
        #   获得种类和先验框的数量
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)#这里获取类别和类别的个数，通过读取txt文件，对字符串进行操作
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)#这里返回的anchors==shape(9,2)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #---------------------------------------------------#
        #   画框设置不同的颜色
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]#不同类别的框颜色不同
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()

        show_config(**self._defaults)

    #---------------------------------------------------#
    #   生成模型
    #---------------------------------------------------#
    def generate(self, onnx=False):
        #---------------------------------------------------#
        #   建立yolo模型，载入yolo模型的权重
        #---------------------------------------------------#
        self.net    = YoloV_body(self.anchors_mask, self.num_classes, self.phi, backbone = self.backbone, input_shape = self.input_shape)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, and classes loaded.'.format(self.model_path))
        if not onnx:
            if self.cuda:
                self.net = nn.DataParallel(self.net)
                self.net = self.net.cuda()

    #---------------------------------------------------#
    #   检测图片
    #---------------------------------------------------#
    def detect_image(self, image, crop = False, count = False):
        #---------------------------------------------------#
        #   计算输入图片的高和宽
        #---------------------------------------------------#
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   给图像增加灰条，实现不失真的resize
        #   也可以直接resize进行识别
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1], self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   添加上batch_size维度
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   将图像输入网络当中进行预测！
            #---------------------------------------------------------#
            outputs = self.net(images)
            mask = outputs[3]
            mask = torch.sigmoid(mask)
            mask = mask[0][0].cpu().numpy()
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   将预测框进行堆叠，然后进行非极大抑制
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
            pdb.set_trace()
            if results[0] is None: 
                print("未检测到任何路面标识！")
                return image,mask

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]#这里的置信度为什么要两个通道相乘？？这个要看nms极大值抑制
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   设置字体与边框厚度
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        #---------------------------------------------------------#
        #   图像绘制
        #---------------------------------------------------------#
        for i, c in list(enumerate(top_label)):
            pdb.set_trace()
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]
            top, left, bottom, right = box
            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image,mask,results


if __name__ == "__main__":
    # pdb.set_trace()
    yolo = YOLO()
    mode = "predict"
    simplify        = True
    onnx_save_path  = "model_data/models.onnx"

    if mode == "predict":
        detect_dir = '/home/ktd/rpf_ws/yolov5-pytorch-main/load_image/'
        l = len(detect_dir)
        detect_image = glob.glob(os.path.join(detect_dir,'*.jpg'))
        # pdb.set_trace()
        mask_save = '/home/ktd/rpf_ws/yolov5-pytorch-main/pred_mask'
        image_save = "/home/ktd/rpf_ws/yolov5-pytorch-main/pred_image"
        # while True:
        for img in detect_image:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
                w,h  = image.size
            except:
                print('Open Error! Try again!')
                continue
            else:
                # pdb.set_trace()
                
                r_image,mask ,results= yolo.detect_image(image)
                # mask[mask>0.5] = 255
                # mask[mask<=0.5] = 0
                # cv2.imwrite('160*160_mask.png',mask)
                pdb.set_trace()
                r_image = cv2.cvtColor(np.asarray(r_image),cv2.COLOR_RGB2BGR)
                row_image = cv2.imread(img)
                stride = 640.0/160.0
                scale = min(640.0/w,640/h)
                iw = w * scale
                ih = h * scale
                cut_h = (640-ih)//2//stride
                cut_w = (640-iw)//2//stride
                res_mask = mask[int(cut_h):int(cut_h+ih//stride),int(cut_w):int(cut_w+iw//stride)]
                res_mask = cv2.resize(res_mask,(w,h),cv2.INTER_CUBIC)
                res_mask[res_mask>0.5] = 255
                res_mask[res_mask<=0.5] = 0
                merge_image=np.concatenate([row_image,res_mask[:,:,np.newaxis]],axis=2)
                merge_image_1 = np.concatenate([r_image,res_mask[:,:,np.newaxis]],axis=2)
                pdb.set_trace()
                cv2.imwrite('res_mask1.png',(res_mask).astype(np.uint8))

                cv2.imwrite(os.path.join(image_save,'test_1.png'),merge_image_1)
                cv2.imwrite(os.path.join(image_save,'test.png'),merge_image)
                cv2.imwrite(os.path.join(image_save,img[l:]),r_image)
                cv2.imwrite(os.path.join(mask_save,img[l:]),(res_mask).astype(np.uint8))
                break
        print("done!")
