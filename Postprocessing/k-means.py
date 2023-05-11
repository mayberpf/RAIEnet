# 对图像用kmeans聚类
# 显示图片的函数
from sys import flags
import cv2
from matplotlib.image import imread
import numpy as np
from matplotlib import pyplot as plt
from numpy.core.defchararray import center
import pdb
def show(winname,src):
    cv2.namedWindow(winname,cv2.WINDOW_GUI_NORMAL)
    cv2.imshow(winname,src)
    cv2.waitKey()

img = cv2.imread('/home/ktd/cv_lane/189.jpg')
o = img.copy()
print(img.shape)
img[img[...]>200] =255
img[img[...]<=200] =0
# show("img",img)
# pdb.set_trace()

# 将一个像素点的rgb值作为一个单元处理，这一点很重要
data = img.reshape((-1,3))
print(data.shape)
# 转换数据类型
data = np.float32(data)
# 设置Kmeans参数
critera = (cv2.TermCriteria_EPS+cv2.TermCriteria_MAX_ITER,10,0.1)
flags = cv2.KMEANS_RANDOM_CENTERS
# 对图片进行四分类
r,best,center = cv2.kmeans(data,5,None,criteria=critera,attempts=10,flags=flags)
print(r)
print(best.shape)
print(center)
center = np.uint8(center)
# 将不同分类的数据重新赋予另外一种颜色，实现分割图片
data[best.ravel()==0] = (0,0,0)
data[best.ravel()==1] = (255,0,0) 
data[best.ravel()==2] = (0,0,255)
data[best.ravel()==4] = (0,255,0) 
data[best.ravel()==3] = (255,255,255) 
# 将结果转换为图片需要的格式
data = np.uint8(data)
oi = data.reshape((img.shape))
# 显示图片
# show('img',img)
show('res',oi)
