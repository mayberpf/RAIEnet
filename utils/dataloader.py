from random import sample, shuffle
from xml.sax.handler import property_interning_dict
import pdb
import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from utils.utils import download_weights, get_anchors, get_classes, show_config
from utils.utils import cvtColor, preprocess_input

# from utils import download_weights, get_anchors, get_classes, show_config
# from utils import cvtColor, preprocess_input




import numpy as np
from PIL import Image


class GridMask(object):
    def __init__(self, use_h, use_w, rotate=1, offset=False, ratio=0.5, mode=0, prob=1.):
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob

    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch

    def __call__(self, sample):
        img, annots = sample['img'], sample['annot']
        if np.random.rand() > self.prob:
            return sample
        h = img.shape[0]
        w = img.shape[1]
        self.d1 = 2
        self.d2 = min(h, w)
        hh = int(1.5 * h)
        ww = int(1.5 * w)
        d = np.random.randint(self.d1, self.d2)#随机产生一个在2到min(h,w)的整数
        # d = self.d
        #        self.l = int(d*self.ratio+0.5)
        if self.ratio == 1:
            self.l = np.random.randint(1, d)
        else:
            self.l = min(max(int(d * self.ratio + 0.5), 1), d - 1)
        mask = np.ones((hh, ww), np.float32)#产生一个全白的mask，宽高是原图像的1.5倍
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh // d):
                s = d * i + st_h
                t = min(s + self.l, hh)
                mask[s:t, :] *= 0
        if self.use_w:
            for i in range(ww // d):
                s = d * i + st_w
                t = min(s + self.l, ww)
                mask[:, s:t] *= 0
        #到这里，生成全白的mask，然后将按照横竖进行置0。最后会将mask和img原图相乘，即可得到最终的数据增强图片。
        r = np.random.randint(self.rotate)#产生一个随机的旋转参数--->应该是旋转角度
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        #        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
        mask = mask[(hh - h) // 2:(hh - h) // 2 + h, (ww - w) // 2:(ww - w) // 2 + w]#产生与原图大小相同的mask

        if self.mode == 1:
            mask = 1 - mask
        mask = np.expand_dims(mask.astype(np.float), axis=2)
        mask = np.tile(mask, [1, 1, 3])
        if self.offset:
            offset = np.float(2 * (np.random.rand(h, w) - 0.5))
            offset = (1 - mask) * offset
            img = img * mask + offset
        else:
            img = img * mask
        return {'img': img, 'annot': annots}



class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length, \
                        mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio = 0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_prob         = mixup_prob
        self.train              = train
        self.special_aug_ratio  = special_aug_ratio

        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)
        
        self.bbox_attrs         = 5 + num_classes
        self.threshold          = 4

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length

        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = sample(self.annotation_lines, 3)#从所有的训练图片中抽取3张图片
            lines.append(self.annotation_lines[index])#将需要提取的图片加入到列表中
            shuffle(lines)#随机排列====目前是四张图片
            image, box  = self.get_random_data_with_Mosaic(lines, self.input_shape)
            
            if self.mixup and self.rand() < self.mixup_prob:
                lines           = sample(self.annotation_lines, 1)
                image_2, box_2  = self.get_random_data(lines[0], self.input_shape, random = self.train)
                image, box      = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        else:
            image, box      = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)

        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))#这里的preprocess_input是对图片进行归一化的处理
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            #---------------------------------------------------#
            #   对真实框进行归一化，调整到0-1之间
            #---------------------------------------------------#
            #这里的box已经是对图片操作之后，相对640，640的label值了
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            #---------------------------------------------------#
            #   序号为0、1的部分，为真实框的中心
            #   序号为2、3的部分，为真实框的宽高
            #   序号为4的部分，为真实框的种类
            #---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]#这一步计算每个box的wh,这个wh相对于640的[0,1]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2#这一步计算每个box的中心点xy
        y_true = self.get_target(box)
        return image, box, y_true

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()#这里将annotation_line切分为列表形式
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])#这里是说获取box的二维数组

        if not random:#这就是不做数据增强，图片和box的转变过程
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            new_image   = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))
            image_data  = np.array(new_image, np.float32)

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.25, 2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)#就这么一个函数实现图像翻转,左右的翻转

        image_data      = np.array(image, np.uint8)#PIL这个库读取图片之后不是np，所以转一步。
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:#首先搞清楚这里的box----->(Xmin,Ymin,Xmax,Ymax)
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx#*真实框的缩放比例+灰边的随机值==得到缩放后真实框的Xmin和Xmax的坐标
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy#同理，这里是y
            if flip: box[:, [0,2]] = w - box[:, [2,0]]#因为翻转只有左右翻转
            box[:, 0:2][box[:, 0:2]<0] = 0#这几行代码写的很神器，主要目的就是希望缩放之后的真实框始终在图中。
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]#根据坐标关系，得到box的wh
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] #这一步不太理解，感觉 就是单纯看这个预测框正不正确
        
        return image_data, box#最后这里返回的是处理后的image和变换过的box==>（x,y,x,y）
    
    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = [] 
        box_datas   = []
        index       = 0
        for line in annotation_line:#在annotation_line中存放在了四张图片的信息
            #---------------------------------#
            #   每一行进行分割
            #---------------------------------#
            line_content = line.split()
            #---------------------------------#
            #   打开图片
            #---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)
            
            #---------------------------------#
            #   图片的大小
            #---------------------------------#
            iw, ih = image.size
            #---------------------------------#
            #   保存框的位置
            #---------------------------------#
            box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
            
            #---------------------------------#
            #   是否翻转图片
            #---------------------------------#
            flip = self.rand()<.5
            if flip and len(box)>0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)#图像翻转
                box[:, [0,2]] = iw - box[:, [2,0]]#预测框翻转

            #------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            #------------------------------------------#
            new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale*h)
                nw = int(nh*new_ar)
            else:
                nw = int(scale*w)
                nh = int(nw/new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            #-----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            #-----------------------------------------------#
            if index == 0:#这里还需要注意一个点，min_offset_x和min_offset_y两个值是随机的，但是在进入循环之前就确定好了
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y) - nh
            elif index == 1:
                dx = int(w*min_offset_x) - nw
                dy = int(h*min_offset_y)
            elif index == 2:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y)
            elif index == 3:
                dx = int(w*min_offset_x)
                dy = int(h*min_offset_y) - nh
            
            new_image = Image.new('RGB', (w,h), (128,128,128))
            new_image.paste(image, (dx, dy))#这里我们要考虑，dx，dy可能为负数，那么可以理解为只贴了下半部分
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            #---------------------------------#
            #   对box进行重新处理
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)#为什么总要打乱？
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0#左上角坐标小于0，就将其置为0
                box[:, 2][box[:, 2]>w] = w#右下角要和wh比
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)]
                box_data = np.zeros((len(box),5))
                box_data[:len(box)] = box
            
            image_datas.append(image_data)
            box_datas.append(box_data)
        # pdb.set_trace()

        #---------------------------------#
        #   将图片分割，放在一起
        #---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image       = np.array(new_image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype           = new_image.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对框进行进一步的处理
        #---------------------------------#
        # pdb.set_trace()
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes
    
    def get_near_points(self, x, y, i, j):
        sub_x = x - i#相对于grid序号的偏差值
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]
        #为什么会根据偏差值返回不同的数组

    def get_target(self, targets):
        #-----------------------------------------------------------#
        #   一共有三个特征层数
        #-----------------------------------------------------------#
        num_layers  = len(self.anchors_mask)
        
        input_shape = np.array(self.input_shape, dtype='int32')
        grid_shapes = [input_shape // {0:32, 1:16, 2:8, 3:4}[l] for l in range(num_layers)]#这里num_layers只有三层，不清楚为什么会有四个键值对，每个键值对实际上对应一个步长[20,20]
        y_true      = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs), dtype='float32') for l in range(num_layers)]#这里是构造了yolo模型真实输出的预测形式[3,20,20,16]
        box_best_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)]#[3,20,20]
        
        if len(targets) == 0:
            return y_true
        
        for l in range(num_layers):
            in_h, in_w      = grid_shapes[l]
            anchors         = np.array(self.anchors) / {0:32, 1:16, 2:8, 3:4}[l]#l对应层数，每一层对应一个步长。每个anchor都是相对于640图片的
            #所以这里需要除以步长，也化成网格形式
            
            batch_target = np.zeros_like(targets)#构建和targets相同形式的数组。此时在targets中，存放是的xywh都是相对于640的[0,1]的值
            #-------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            #-------------------------------------------------------#
            batch_target[:, [0,2]]  = targets[:, [0,2]] * in_w#之前已经化成了[0,1]所以要转换为网格数为多少，乘就完了。
            batch_target[:, [1,3]]  = targets[:, [1,3]] * in_h
            batch_target[:, 4]      = targets[:, 4]
            #-------------------------------------------------------#
            #   wh                          : num_true_box, 2
            #   np.expand_dims(wh, 1)       : num_true_box, 1, 2
            #   anchors                     : 9, 2
            #   np.expand_dims(anchors, 0)  : 1, 9, 2
            #   
            #   ratios_of_gt_anchors代表每一个真实框和每一个先验框的宽高的比值
            #   ratios_of_gt_anchors    : num_true_box, 9, 2
            #   ratios_of_anchors_gt代表每一个先验框和每一个真实框的宽高的比值
            #   ratios_of_anchors_gt    : num_true_box, 9, 2
            #
            #   ratios                  : num_true_box, 9, 4
            #   max_ratios代表每一个真实框和每一个先验框的宽高的比值的最大值
            #   max_ratios              : num_true_box, 9
            #-------------------------------------------------------#
            ratios_of_gt_anchors = np.expand_dims(batch_target[:, 2:4], 1) / np.expand_dims(anchors, 0)#计算过后[4,9,2]
            #这个比率为什么这么算，还要扩展维度！我建议可以看看数组运算的广播机制！真的很神奇
            ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)#[4,9,2]
            ratios               = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis = -1)#[4,9,4]
            max_ratios           = np.max(ratios, axis = -1)#[4,9]
            
            for t, ratio in enumerate(max_ratios):
                #-------------------------------------------------------#
                #   ratio : 9
                #-------------------------------------------------------#
                over_threshold = ratio < self.threshold
                over_threshold[np.argmin(ratio)] = True#np.argmin()返回的是最小值的序号
                for k, mask in enumerate(self.anchors_mask[l]):
                    if not over_threshold[mask]:
                        continue
                    #----------------------------------------#
                    #   获得真实框属于哪个网格点
                    #   x  1.25     => 1
                    #   y  3.75     => 3
                    #----------------------------------------#
                    i = int(np.floor(batch_target[t, 0]))#batch_target是所有box按照grit划分后的值,t--->哪一个box框,i---->框中心点x属于哪个网格
                    j = int(np.floor(batch_target[t, 1]))
                    
                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)#batch_target[t, 0], batch_target[t, 1]中心点xy坐标，ij是中心点所在网格的序号
                    #返回了数组[0,0],[1,0],[0,1]
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]

                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:#确保这个点在grid 网格中
                            continue

                        if box_best_ratio[l][k, local_j, local_i] != 0:#这里不太清楚，在前面新设定的数组都是0，为什么还要检测是不是0。
                            if box_best_ratio[l][k, local_j, local_i] > ratio[mask]:
                                y_true[l][k, local_j, local_i, :] = 0
                            else:
                                continue
                            
                        #----------------------------------------#
                        #   取出真实框的种类
                        #----------------------------------------#
                        c = int(batch_target[t, 4])#这里得到类别的序号。

                        #----------------------------------------#
                        #   tx、ty代表中心调整参数的真实值
                        #----------------------------------------#
                        y_true[l][k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[l][k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[l][k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[l][k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[l][k, local_j, local_i, 4] = 1
                        y_true[l][k, local_j, local_i, c + 5] = 1
                        #----------------------------------------#
                        #   获得当前先验框最好的比例
                        #----------------------------------------#
                        box_best_ratio[l][k, local_j, local_i] = ratio[mask]#这里将比例也放在了存了进来？？不理解===#看上面，这就是为什么会比较box_best_ratio
                        
        return y_true
    
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images  = []
    bboxes  = []
    y_trues = [[] for _ in batch[0][2]]
    masks = []
    for img, box, y_true,mask in batch:
        images.append(img)
        bboxes.append(box)
        masks.append(mask)
        for i, sub_y_true in enumerate(y_true):
            y_trues[i].append(sub_y_true)
            
    images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    masks  = torch.from_numpy(np.array(masks)).type(torch.FloatTensor)
    bboxes  = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    y_trues = [torch.from_numpy(np.array(ann, np.float32)).type(torch.FloatTensor) for ann in y_trues]
    return images, bboxes,y_trues,masks

#=================================YV_dataset====================#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#================================================================#
class Yolo_YV_Dataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length, \
                        mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio = 0.7):
        super(Yolo_YV_Dataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.anchors            = anchors
        self.anchors_mask       = anchors_mask
        self.epoch_length       = epoch_length
        self.mosaic             = mosaic
        self.mosaic_prob        = mosaic_prob
        self.mixup              = mixup
        self.mixup_prob         = mixup_prob
        self.train              = train
        self.special_aug_ratio  = special_aug_ratio

        self.epoch_now          = -1
        self.length             = len(self.annotation_lines)
        
        self.bbox_attrs         = 5 + num_classes
        self.threshold          = 4
        self.mask_path = '/home/ktd/rpf_ws/yolov5-pytorch-main/VOCdevkit/VOC2007/train_mask/'
        self.use_h = True
        self.use_w = True
        self.mode = 1

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length

        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        # #---------------------------------------------------#
        # if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
        #     lines = sample(self.annotation_lines, 3)
        #     lines.append(self.annotation_lines[index])
        #     shuffle(lines)
        #     image, box  = self.get_random_data_with_Mosaic(lines, self.input_shape)
            
            # if self.mixup and self.rand() < self.mixup_prob:
            #     lines           = sample(self.annotation_lines, 1)
            #     image_2, box_2  = self.get_random_data(lines[0], self.input_shape, random = self.train)
            #     image, box      = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        # else:

        image, box,mask      = self.get_random_data(self.annotation_lines[index], self.input_shape, random = self.train)
        # pdb.set_trace()
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        # mask = mask.resize()
        mask = np.array(mask,dtype=np.float32)
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            #---------------------------------------------------#
            #   对真实框进行归一化，调整到0-1之间
            #---------------------------------------------------#
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]
            #---------------------------------------------------#
            #   序号为0、1的部分，为真实框的中心
            #   序号为2、3的部分，为真实框的宽高
            #   序号为4的部分，为真实框的种类
            #---------------------------------------------------#
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        y_true = self.get_target(box)
        return image, box, y_true,mask

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    # def get_mask(annotations):

    #     pass

    def get_random_data(self, annotation_line, input_shape, jitter=.2, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        mask_name = line[0].split('/')[-1].replace('jpg','png')
        mask_path = self.mask_path+mask_name
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        mask = Image.open(mask_path)
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        # pdb.set_trace()
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])
        #==================这里加入gridmask的数据增强========================#

        if self.rand()>0.75:
            hh = int(ih*1.5)
            ww = int(iw*1.5)
            grid_mask = np.ones((hh, ww), np.float32)
            d = np.random.randint(ih//8, ih//4)#随机产生一个在2到min(h,w)的整数
            l = np.random.randint(100, 200)
            st_h = np.random.randint(d)
            st_w = np.random.randint(d)
            if self.use_h:
                for i in range(hh // d):
                    s = d * i + st_h
                    t = min(s + l, hh)
                    grid_mask[s:t, :] *= 0
            if self.use_w:
                for i in range(ww // d):
                    s = d * i + st_w
                    t = min(s + l, ww)
                    grid_mask[:, s:t] *= 0
            # r = np.random.randint(self.rotate)#产生一个随机的旋转参数--->应该是旋转角度
            # pdb.set_trace()
            grid_mask = Image.fromarray(np.uint8(grid_mask))
            # grid_mask.show()
            # mask = mask.rotate(r)
            grid_mask = np.asarray(grid_mask)
            #        mask = 1*(np.random.randint(0,3,[hh,ww])>0)
            grid_mask = grid_mask[(hh - ih) // 2:(hh - ih) // 2 + ih, (ww - iw) // 2:(ww - iw) // 2 + iw]#产生与原图大小相同的mask
        

            if self.mode == 1:
                grid_mask = 1 - grid_mask
            grid_mask = np.expand_dims(grid_mask.astype(np.float16), axis=2)
            grid_mask = np.tile(grid_mask, [1, 1, 3])
            # if self.offset:
            #     offset = np.float(2 * (np.random.rand(h, w) - 0.5))
            #     offset = (1 - mask) * offset
            #     img = img * mask + offset
            # else:
            image = image * grid_mask
            # pdb.set_trace()
            image = Image.fromarray(np.uint8(image))
            # image.show()

            # return {'img': img, 'annot': annots}
            

            

        
        
        
        # pdb.set_trace()

        if not random and self.rand()>0.3:
            scale = min(w/iw, h/ih)
            nw = int(iw*scale)
            nh = int(ih*scale)
            dx = (w-nw)//2
            dy = (h-nh)//2

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC)
            mask       = mask.resize((nw,nh), Image.BICUBIC)
            new_image_1   = Image.new('RGB', (w,h), (128,128,128))
            new_image_2   = Image.new('L', (w,h),0)
            new_image_1.paste(image, (dx, dy))
            new_image_2.paste(mask, (dx, dy))
            new_image_2 = new_image_2.resize((160,160),Image.BICUBIC)
            image_data  = np.array(new_image_1, np.float32)
            # pdb.set_trace()
            mask_max = np.array(new_image_2).max()
            if mask_max ==0:
                mask_data  = np.array(new_image_2, np.float32)
            else:
                mask_data  = np.array(new_image_2, np.float32)/mask_max

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0:
                np.random.shuffle(box)
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
                box[:, 0:2][box[:, 0:2]<0] = 0
                box[:, 2][box[:, 2]>w] = w
                box[:, 3][box[:, 3]>h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box

            return image_data, box ,mask_data
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
        scale = self.rand(.8, 1.2)
        if new_ar < 1:
            nh = int(scale*h)
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC)
        mask = mask.resize((nw,nh), Image.BICUBIC)

        #------------------------------------------#
        #   将图像多余的部分加上灰条
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image_1 = Image.new('L', (w,h) ,0)
        new_image.paste(image, (dx, dy))
        new_image_1.paste(mask, (dx, dy))
        image = new_image
        mask = new_image_1
        mask = mask.resize((160,160),Image.BICUBIC)

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: 
            image = image.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        image_data      = np.array(image, np.uint8)
        # pdb.set_trace()
        mask_max = int(np.array(mask).max())
        if mask_max ==0:
            mask_data      = np.array(mask, np.uint8)
        else:
            mask_data      = np.array(mask, np.uint8)/mask_max

        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            # if flip: box[:, [0,2]] = w - box[:, [2,0]]
            if flip: box[:, [1,3]] = h - box[:, [3,1]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box,mask_data
    
    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    # def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
    #     h, w = input_shape
    #     min_offset_x = self.rand(0.3, 0.7)
    #     min_offset_y = self.rand(0.3, 0.7)

    #     image_datas = [] 
    #     box_datas   = []
    #     index       = 0
    #     for line in annotation_line:
    #         #---------------------------------#
    #         #   每一行进行分割
    #         #---------------------------------#
    #         line_content = line.split()
    #         #---------------------------------#
    #         #   打开图片
    #         #---------------------------------#
    #         image = Image.open(line_content[0])
    #         image = cvtColor(image)
            
    #         #---------------------------------#
    #         #   图片的大小
    #         #---------------------------------#
    #         iw, ih = image.size
    #         #---------------------------------#
    #         #   保存框的位置
    #         #---------------------------------#
    #         box = np.array([np.array(list(map(int,box.split(',')))) for box in line_content[1:]])
            
    #         #---------------------------------#
    #         #   是否翻转图片
    #         #---------------------------------#
    #         flip = self.rand()<.5
    #         if flip and len(box)>0:
    #             image = image.transpose(Image.FLIP_LEFT_RIGHT)
    #             box[:, [0,2]] = iw - box[:, [2,0]]

    #         #------------------------------------------#
    #         #   对图像进行缩放并且进行长和宽的扭曲
    #         #------------------------------------------#
    #         new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter)
    #         scale = self.rand(.4, 1)
    #         if new_ar < 1:
    #             nh = int(scale*h)
    #             nw = int(nh*new_ar)
    #         else:
    #             nw = int(scale*w)
    #             nh = int(nw/new_ar)
    #         image = image.resize((nw, nh), Image.BICUBIC)

    #         #-----------------------------------------------#
    #         #   将图片进行放置，分别对应四张分割图片的位置
    #         #-----------------------------------------------#
    #         if index == 0:
    #             dx = int(w*min_offset_x) - nw
    #             dy = int(h*min_offset_y) - nh
    #         elif index == 1:
    #             dx = int(w*min_offset_x) - nw
    #             dy = int(h*min_offset_y)
    #         elif index == 2:
    #             dx = int(w*min_offset_x)
    #             dy = int(h*min_offset_y)
    #         elif index == 3:
    #             dx = int(w*min_offset_x)
    #             dy = int(h*min_offset_y) - nh
            
    #         new_image = Image.new('RGB', (w,h), (128,128,128))
    #         new_image.paste(image, (dx, dy))
    #         image_data = np.array(new_image)

    #         index = index + 1
    #         box_data = []
    #         #---------------------------------#
    #         #   对box进行重新处理
    #         #---------------------------------#
    #         if len(box)>0:
    #             np.random.shuffle(box)
    #             box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
    #             box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
    #             box[:, 0:2][box[:, 0:2]<0] = 0
    #             box[:, 2][box[:, 2]>w] = w
    #             box[:, 3][box[:, 3]>h] = h
    #             box_w = box[:, 2] - box[:, 0]
    #             box_h = box[:, 3] - box[:, 1]
    #             box = box[np.logical_and(box_w>1, box_h>1)]
    #             box_data = np.zeros((len(box),5))
    #             box_data[:len(box)] = box
            
    #         image_datas.append(image_data)
    #         box_datas.append(box_data)

    #     #---------------------------------#
    #     #   将图片分割，放在一起
    #     #---------------------------------#
    #     cutx = int(w * min_offset_x)
    #     cuty = int(h * min_offset_y)

    #     new_image = np.zeros([h, w, 3])
    #     new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
    #     new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
    #     new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
    #     new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

    #     new_image       = np.array(new_image, np.uint8)
    #     #---------------------------------#
    #     #   对图像进行色域变换
    #     #   计算色域变换的参数
    #     #---------------------------------#
    #     r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
    #     #---------------------------------#
    #     #   将图像转到HSV上
    #     #---------------------------------#
    #     hue, sat, val   = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
    #     dtype           = new_image.dtype
    #     #---------------------------------#
    #     #   应用变换
    #     #---------------------------------#
    #     x       = np.arange(0, 256, dtype=r.dtype)
    #     lut_hue = ((x * r[0]) % 180).astype(dtype)
    #     lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
    #     lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

    #     new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
    #     new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

    #     #---------------------------------#
    #     #   对框进行进一步的处理
    #     #---------------------------------#
    #     new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

    #     return new_image, new_boxes

    # def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
    #     new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
    #     if len(box_1) == 0:
    #         new_boxes = box_2
    #     elif len(box_2) == 0:
    #         new_boxes = box_1
    #     else:
    #         new_boxes = np.concatenate([box_1, box_2], axis=0)
    #     return new_image, new_boxes
    
    def get_near_points(self, x, y, i, j):
        sub_x = x - i
        sub_y = y - j
        if sub_x > 0.5 and sub_y > 0.5:
            return [[0, 0], [1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y > 0.5:
            return [[0, 0], [-1, 0], [0, 1]]
        elif sub_x < 0.5 and sub_y < 0.5:
            return [[0, 0], [-1, 0], [0, -1]]
        else:
            return [[0, 0], [1, 0], [0, -1]]

    def get_target(self, targets):
        #-----------------------------------------------------------#
        #   一共有三个特征层数
        #-----------------------------------------------------------#
        num_layers  = len(self.anchors_mask)
        
        input_shape = np.array(self.input_shape, dtype='int32')
        grid_shapes = [input_shape // {0:32, 1:16, 2:8, 3:4}[l] for l in range(num_layers)]
        y_true      = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1], self.bbox_attrs), dtype='float32') for l in range(num_layers)]
        box_best_ratio = [np.zeros((len(self.anchors_mask[l]), grid_shapes[l][0], grid_shapes[l][1]), dtype='float32') for l in range(num_layers)]
        
        if len(targets) == 0:
            return y_true
        
        for l in range(num_layers):
            in_h, in_w      = grid_shapes[l]
            anchors         = np.array(self.anchors) / {0:32, 1:16, 2:8, 3:4}[l]
            
            batch_target = np.zeros_like(targets)
            #-------------------------------------------------------#
            #   计算出正样本在特征层上的中心点
            #-------------------------------------------------------#
            batch_target[:, [0,2]]  = targets[:, [0,2]] * in_w
            batch_target[:, [1,3]]  = targets[:, [1,3]] * in_h
            batch_target[:, 4]      = targets[:, 4]
            #-------------------------------------------------------#
            #   wh                          : num_true_box, 2
            #   np.expand_dims(wh, 1)       : num_true_box, 1, 2
            #   anchors                     : 9, 2
            #   np.expand_dims(anchors, 0)  : 1, 9, 2
            #   
            #   ratios_of_gt_anchors代表每一个真实框和每一个先验框的宽高的比值
            #   ratios_of_gt_anchors    : num_true_box, 9, 2
            #   ratios_of_anchors_gt代表每一个先验框和每一个真实框的宽高的比值
            #   ratios_of_anchors_gt    : num_true_box, 9, 2
            #
            #   ratios                  : num_true_box, 9, 4
            #   max_ratios代表每一个真实框和每一个先验框的宽高的比值的最大值
            #   max_ratios              : num_true_box, 9
            #-------------------------------------------------------#
            ratios_of_gt_anchors = np.expand_dims(batch_target[:, 2:4], 1) / np.expand_dims(anchors, 0)
            ratios_of_anchors_gt = np.expand_dims(anchors, 0) / np.expand_dims(batch_target[:, 2:4], 1)
            ratios               = np.concatenate([ratios_of_gt_anchors, ratios_of_anchors_gt], axis = -1)
            max_ratios           = np.max(ratios, axis = -1)
            
            for t, ratio in enumerate(max_ratios):
                #-------------------------------------------------------#
                #   ratio : 9
                #-------------------------------------------------------#
                over_threshold = ratio < self.threshold
                over_threshold[np.argmin(ratio)] = True
                for k, mask in enumerate(self.anchors_mask[l]):
                    if not over_threshold[mask]:
                        continue
                    #----------------------------------------#
                    #   获得真实框属于哪个网格点
                    #   x  1.25     => 1
                    #   y  3.75     => 3
                    #----------------------------------------#
                    i = int(np.floor(batch_target[t, 0]))
                    j = int(np.floor(batch_target[t, 1]))
                    
                    offsets = self.get_near_points(batch_target[t, 0], batch_target[t, 1], i, j)
                    for offset in offsets:
                        local_i = i + offset[0]
                        local_j = j + offset[1]

                        if local_i >= in_w or local_i < 0 or local_j >= in_h or local_j < 0:
                            continue

                        if box_best_ratio[l][k, local_j, local_i] != 0:
                            if box_best_ratio[l][k, local_j, local_i] > ratio[mask]:
                                y_true[l][k, local_j, local_i, :] = 0
                            else:
                                continue
                            
                        #----------------------------------------#
                        #   取出真实框的种类
                        #----------------------------------------#
                        c = int(batch_target[t, 4])

                        #----------------------------------------#
                        #   tx、ty代表中心调整参数的真实值
                        #----------------------------------------#
                        y_true[l][k, local_j, local_i, 0] = batch_target[t, 0]
                        y_true[l][k, local_j, local_i, 1] = batch_target[t, 1]
                        y_true[l][k, local_j, local_i, 2] = batch_target[t, 2]
                        y_true[l][k, local_j, local_i, 3] = batch_target[t, 3]
                        y_true[l][k, local_j, local_i, 4] = 1
                        y_true[l][k, local_j, local_i, c + 5] = 1
                        #----------------------------------------#
                        #   获得当前先验框最好的比例
                        #----------------------------------------#
                        box_best_ratio[l][k, local_j, local_i] = ratio[mask]
                        
        return y_true

    # def yolo_dataset_collate(batch):
    #     images  = []
    #     bboxes  = []
    #     y_trues = [[] for _ in batch[0][2]]
    #     for img, box, y_true in batch:
    #         images.append(img)
    #         bboxes.append(box)
    #         for i, sub_y_true in enumerate(y_true):
    #             y_trues[i].append(sub_y_true)
                
    #     images  = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    #     bboxes  = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    #     y_trues = [torch.from_numpy(np.array(ann, np.float32)).type(torch.FloatTensor) for ann in y_trues]
    #     return images, bboxes,y_trues

# if __name__ =='__main__':
#     train_annotation_path = '/home/ktd/rpf_ws/yolov5-pytorch-main/2007_train.txt'
#     with open(train_annotation_path, encoding='utf-8') as f:#这里面存放的是图片的地址和bbox的标签---训练集
#         train_lines = f.readlines()
#     input_shape = [640,640]
#     classes_path = '/home/ktd/rpf_ws/yolov5-pytorch-main/model_data/loadmark_classes.txt'
#     anchors_path = '/home/ktd/rpf_ws/yolov5-pytorch-main/model_data/yolo_anchors.txt'
#     anchors_mask    = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
#     class_names, num_classes = get_classes(classes_path)
#     anchors, num_anchors     = get_anchors(anchors_path)
#     UnFreeze_Epoch      = 50
#     mosaic              = False
#     mosaic_prob         = 0.5
#     mixup               = False
#     mixup_prob          = 0.5
#     special_aug_ratio   = 0.7
#     # pdb.set_trace()
#     data_load1 = Yolo_YV_Dataset(train_lines, input_shape, num_classes, anchors, anchors_mask, epoch_length=UnFreeze_Epoch, \
#                 mosaic=mosaic, mixup=mixup, mosaic_prob=mosaic_prob, mixup_prob=mixup_prob, train=False, special_aug_ratio=special_aug_ratio)
#     # pdb.set_trace()
#     i = 0
#     while True:
#         # pdb.set_trace()
#         data = data_load1[i]
#         # print(len(data))
#         # print(data[0].shape)
#         # print(data[1].size)
#         # print(data[2][0].shape)
#         # print(data[2][1].shape)
#         # print(data[2][2].shape)
#         print(data[3].shape)
#         print(i,data[3].max())
#         print(type(data[0]))
#         print(data[0].shape)
#         import matplotlib.pyplot as plt
#         plt.subplot(121)
#         plt.imshow(data[0].transpose(1,2,0))
#         # plt.imshow(mask_pred,alpha=0.2)
#         plt.title('image')
#         plt.axis('off')
#         plt.subplot(122)
#         plt.imshow(data[3])
#         # plt.imshow(mask_pred,alpha=0.2)
#         plt.title('mask')
#         plt.axis('off')
#         # pdb.set_trace()
#         plt.show()
#         i+=1




        # break
        # i+=1
        # from PIL import Image

        # img = np.transpose(data[0],(1,2,0))

        # mask = data[3]
        # print(img.shape)
        # plt.subplot(121)
        # plt.imshow(img)
        # # plt.imshow(mask_pred,alpha=0.2)
        # plt.title('img')
        # plt.axis('off')
        # plt.subplot(122)
        # plt.imshow(mask)
        # # plt.imshow(mask_pred,alpha=0.2)
        # plt.title('mask')
        # plt.axis('off')
        # pdb.set_trace()

        # plt.show()


        # img = Image.fromarray(img.astype('uint8'))
        # img.show()

