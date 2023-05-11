import numpy as np
import cv2 
labels = ['BL', 'CL', 'DM', 'JB', 'LA', 'PC', 'RA', 'SA', 'SL', 'SLA', 'SRA']
results = [np.array([[8.63770508e+02, 2.96160522e+02, 9.54357605e+02, 5.57585083e+02,
        9.84152377e-01, 9.99244094e-01, 4.00000000e+00],
       [8.76527344e+02, 1.34535913e+03, 9.76824524e+02, 1.65187683e+03,
        9.71647501e-01, 9.99576867e-01, 7.00000000e+00],
       [8.83139526e+02, 8.97762451e+02, 1.02303076e+03, 9.89018005e+02,
        9.57201958e-01, 9.99280751e-01, 7.00000000e+00]])]
    
print(len(results))


#读入图片
img = cv2.imread("/home/rpf/rpf_code/cv_lane/res_mask.png",0)

# import pdb;pdb.set_trace()
# # 中值滤波，去噪
# img = cv2.medianBlur(img, 3)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('original', gray)

# # 阈值分割得到二值化图片
# ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# # 膨胀操作
# kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# bin_clo = cv2.dilate(binary, kernel2, iterations=2)
# num_objects, labels = cv2.connectedComponents(img)

# 连通域分析
kernel = np.ones((5,5),np.uint8)
img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
# cv2.imshow('oginal', img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# import pdb;pdb.set_trace()


num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

# import pdb;pdb.set_trace()

# 算法流程


# 去除小面积区域、添加车道线index
# #====根据stats得到的类别情况，筛选去除面积小的区域====#
label_lists = []
for cls_index in range(1,num_labels):
    area_s = stats[cls_index][-1]
    if area_s < 400:
        labels[labels ==cls_index] = 0
    else:
        label_lists.append(cls_index)
    # pass
import pdb;pdb.set_trace()
#=========== 对应车道线index，获得外接矩形中心点=========#
centers = []
for cls_index in label_lists:
    center  = centroids[cls_index]
    centers.append(centroids[cls_index])

pdb.set_trace()
# ==========获取路面标识框中心点坐标============#

results = results[0]
print(results.shape)
y1 = results[:,0]
# x1 = results[:,1]
y2 = results[:,2]
# x2 = results[:,3]
# x_mid = x2 - x1
y_mid = y2 - y1

y_mid = np.expand_dims(y_mid,axis=1)
print(y_mid)
results = np.concatenate([results,y_mid],axis=1)
print(results.shape)
l = y_mid.shape[0]
# print(results)
results = results[np.lexsort(results.T)]
print(results)

# 计算中心点坐标的距离，比较，选两个最小值

# 确定对应车道先index

# 确定路面标识所在位置（两车道先所夹位置）




















# results = results[0]
# print(results.shape)
# y1 = results[:,0]
# # x1 = results[:,1]
# y2 = results[:,2]
# # x2 = results[:,3]
# # x_mid = x2 - x1
# y_mid = y2 - y1

# y_mid = np.expand_dims(y_mid,axis=1)
# print(y_mid)
# results = np.concatenate([results,y_mid],axis=1)
# print(results.shape)
# l = y_mid.shape[0]
# # print(results)
# results = results[np.lexsort(results.T)]
# print(results)


# for i in range(results.shape[0]):
#     # import pdb;pdb.set_trace()
#     label = labels[int(results[i,-2])]
#     print(f"第{i+1}个车道对应的标签为:",label)
#     # break