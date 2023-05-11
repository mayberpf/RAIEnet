# import cv2
import numpy as np

# 读入图片
# img = cv2.imread("/home/ktd/cv_lane/res_mask1.png",0)

# # import pdb;pdb.set_trace()
# # # 中值滤波，去噪
# # img = cv2.medianBlur(img, 3)
# # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# # cv2.namedWindow('original', cv2.WINDOW_AUTOSIZE)
# # cv2.imshow('original', gray)

# # # 阈值分割得到二值化图片
# # ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

# # # 膨胀操作
# # kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# # bin_clo = cv2.dilate(binary, kernel2, iterations=2)
# # num_objects, labels = cv2.connectedComponents(img)

# # 连通域分析
# kernel = np.ones((5,5),np.uint8)
# img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
# # cv2.imshow('oginal', img)
# # cv2.waitKey()
# # cv2.destroyAllWindows()
# # import pdb;pdb.set_trace()


# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)

# # 查看各个返回值
# # 连通域数量
# print('num_labels = ',num_labels)
# # 连通域的信息：对应各个轮廓的x、y、width、height和面积
# print('stats = ',stats)
# # 连通域的中心点
# print('centroids = ',centroids)
# # 每一个像素的标签1、2、3.。。，同一个连通域的标签是一致的
# print('labels = ',labels)
# # import pdb;pdb.set_trace()
# label_lists = []
# lane_points = []
# #====根据stats得到的类别情况，筛选去除面积小的区域====#
# for cls_index in range(1,num_labels):
#     area_s = stats[cls_index][-1]
#     if area_s < 400:
#         labels[labels ==cls_index] = 0
#     else:
#         label_lists.append(cls_index)
#     pass
# # import pdb;pdb.set_trace()
# #====根据label_list取出每条车道线，按照逻辑取出点坐标！====#
# for index , label in enumerate (label_lists):
#     y,x = np.where(labels ==label)
#     y_min = y.min()
#     y_max = y.max()
#     dis_y  =y_max - y_min
#     if dis_y  <100:
#         point_num = 3
#     elif 100<=dis_y<400:
#         point_num = 6
#     else:
#         point_num = 8
#     y_pos = [i for i in range(y_min,y_max,int(dis_y/(point_num-1)))]
#     if len(y_pos) != point_num:
#         y_pos.append(y_max)
# #============取出y坐标=============#
#     x_pos = []
#     for y_po in y_pos:
#         index = np.where(y==y_po)
#         x_po = int(x[index].mean())
#         x_pos.append(x_po)

#     lane = []
#     for i in range(len(y_pos)):
#         lane.append([y_pos[i],x_pos[i]])
#     lane_points.append(lane)
#     # pdb.set_trace()
# print(lane_points)



#=============车道线像素点====list=============#

lane_points = [[[728, 925], [771, 896], [814, 855], [857, 799], [900, 743], [943, 697]], [[743, 772], [792, 662], [841, 503], [890, 335], [939, 161], [988, 5]], [[742, 1023], [784, 1067], [826, 1101], [868, 1134], [910, 1168], [952, 1196]], [[744, 1128], [789, 1242], [834, 1379], [879, 1479], [924, 1603], [969, 1696]], [[781, 1423], [811, 1537], [841, 1648], [871, 1806], [901, 1920], [931, 2023]], [[973, 1727], [991, 1759], [1009, 1797]]]
# import pdb;pdb.set_trace()
#=================下面是验证提取出的点的位置是否正确（点的可视化，点==半径很小的圆）====================#
# mask = np.zeros((img.shape[0], img.shape[1]), np.uint8)

#====相机内参=====#
camera_Intrinsics = np.array([[1201.83, 0, 1022.68],
[0, 1196.92, 753.325],
[0, 0, 1]])
print(camera_Intrinsics.shape)
#====求相机内参矩阵的逆矩阵，然后该逆矩阵与像素坐标相乘(像素坐标(2,1)---->(3,1))========#
camera_Intrinsics_n = np.linalg.inv(camera_Intrinsics)
print(camera_Intrinsics_n.shape)
print(camera_Intrinsics_n)
camera_coordinate_lanes = []
#==================取出像素点坐标===============#
#====================坐标系转换================#
#=============像素坐标到--->相机坐标系===========#
for one_lane in lane_points:
    camera_coordinate_lane = []
    for px in one_lane:
        # print(px)
        v,u = px
        px_coordinate = np.array([u,v,1])
        px_coordinate = np.transpose(px_coordinate)
        # print(px_coordinate.shape)
        camera_coordinate = np.dot(camera_Intrinsics_n,px_coordinate)
        camera_coordinate_lane.append(camera_coordinate)
        # print(camera_coordinate)
    camera_coordinate_lanes.append(camera_coordinate_lane)

print(len(camera_coordinate_lanes))


#===========激光雷达--->相机==旋转平移矩阵=======#
Lidar_to_right_camera = np.array([[0.0237966, -0.999415, 0.0245615, -0.612344],
[-0.0699549, -0.026173, -0.997207, -0.106558],
[0.997267, 0.0220119, -0.0705369, -0.0426379],
[0,0,0,1]])
print(Lidar_to_right_camera.shape)
#====求j激光雷达到相机的旋转平移矩阵的逆矩阵，然后该逆矩阵与相机坐标相乘(相机坐标(3,1)---->(4,1))========#
Lidar_to_right_camera_n = np.linalg.inv(Lidar_to_right_camera)
print(Lidar_to_right_camera_n.shape)
print(Lidar_to_right_camera_n)

#==================取出相机坐标系下坐标===============#
#====================坐标系转换================#
#=============相机坐标到--->激光雷达坐标系===========#
Lidar_coordinate_lanes = []
for one_lane in camera_coordinate_lanes:
    Lidar_coordinate_lane = []
    for camer_co in one_lane:
        # print(px)
        x,y,z = camer_co
        camer_co = np.array([x,y,z,1])
        print(camer_co)
        camer_co = np.transpose(camer_co)
        # print(px_coordinate.shape)
        Lidar_coordinate = np.dot(Lidar_to_right_camera_n,camer_co)
        Lidar_coordinate_lane.append(Lidar_coordinate)
        # print(camera_coordinate)
    Lidar_coordinate_lanes.append(Lidar_coordinate_lane)

print(len(Lidar_coordinate_lanes))
print(Lidar_coordinate_lanes[0])



import numpy as np  # 用来处理数据
import matplotlib.pyplot as plt
x = np.array([])
y = np.array([])
z = np.array([])
for i in range(len(Lidar_coordinate_lanes)):
    if i >0:
        break
    for lidar_lane in Lidar_coordinate_lanes[i]:
        # import pdb;pdb.set_trace()
        x = np.append(x,lidar_lane[0])
        y = np.append(y,lidar_lane[1])
        z = np.append(z,lidar_lane[2])
    # print(x.shape,y.shape,z.shape)
 
ax = plt.subplot(projection = '3d')  # 创建一个三维的绘图工程
ax.set_title('3d_image_show')  # 设置本图名称
ax.scatter(x, y, z, c = 'r')   # 绘制数据点 c: 'r'红色，'y'黄色，等颜色
 
ax.set_xlabel('X')  # 设置x坐标轴
ax.set_ylabel('Y')  # 设置y坐标轴
ax.set_zlabel('Z')  # 设置z坐标轴
 
plt.show()



#激光雷达坐标系---->地图坐标系








# image = cv2.imread('/home/ktd/rpf_ws/yolov5-pytorch-main/load_image/189.jpg')
# for one_lane in lane_points:
#     for i in range(len(one_lane)):
#         y,x = one_lane[i]
#         cv2.circle(image,(x,y),3,255,4)
#         # mask[y,x] = 255
# pdb.set_trace()
# cv2.imwrite('points_test2.png',image)
# cv2.imshow('points_mask', image)
# cv2.waitKey()
# cv2.destroyAllWindows()
# # 不同的连通域赋予不同的颜色
# output = np.zeros((img.shape[0], img.shape[1], 3), np.uint8)
# for i in range(1, num_labels):
#     mask = labels == i
#     output[:, :, 0][mask] = np.random.randint(0, 255)
#     output[:, :, 1][mask] = np.random.randint(0, 255)
#     output[:, :, 2][mask] = np.random.randint(0, 255)
# cv2.imwrite('own_test.png',output)
# # cv2.imshow('oginal', output)
# cv2.waitKey()
# cv2.destroyAllWindows()

