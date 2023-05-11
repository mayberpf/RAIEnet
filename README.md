# 基于cspdarknet的路面信息检测模型
我们基于yolov5的backbone修改模型的neck和head使其能够满足对车道线和路面标识检测的要求。
基于分割获取浅层车道线信息及检测的方法获取路面标识信息。
本代码只包括预测及结果可视化部分，训练代码由于比较复杂，所以没有放进来。
# 效果图
@import "test_1.png"
@import "189.jpg"

# 依赖
```ruby
    matplotlib
    glob
    cv2
    colorsys
    os
    time
    numpy 
    torch
    PIL 
 ```
 # 数据准备
 在main函数中，需要指定检测图片的文件夹;
 同时，需在下方设置检测结果所保存的位置。
 默认数据输入格式为宽:1920,高:1080
 其余图片输入格式还没有尝试。
 ```ruby
    detect_dir = '/home/ktd/rpf_ws/yolov5-pytorch-main/VOCdevkit/VOC2007/JPEGImages/'
    mask_save = '/home/ktd/rpf_ws/yolov5-pytorch-main/pred_mask'
    image_save = "/home/ktd/rpf_ws/yolov5-pytorch-main/pred_image"
 ```
 # 模型权重
 模型权重可以通过百度网盘下载：
链接: https://pan.baidu.com/s/1NLBB8lE1VUuPV8F5DUueLg 提取码: v5tv 
 指定权重所在地址，在YOLO类的初始化中
 ```ruby
         "model_path"        : '/home/ktd/rpf_ws/yolov5-pytorch-main/12/last_epoch_weights.pth',
 ```
# Run
```ruby
pyhton predict_image.py
```
 