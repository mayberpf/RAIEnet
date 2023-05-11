import open3d as o3d
import numpy as np
# import cv2
# import imageio
import pdb



# def farthest_point_sample(point, npoint):
#     """
#     Input:
#         xyz: pointcloud data, [N, D]
#         npoint: number of samples
#     Return:
#         centroids: sampled pointcloud index, [npoint, D]
#     """
#     N, D = point.shape
#     xyz = point[:,:3]
#     centroids = np.zeros((npoint,))
#     distance = np.ones((N,)) * 1e10
#     farthest = np.random.randint(0, N)
#     for i in range(npoint):
#         centroids[i] = farthest
#         centroid = xyz[farthest, :]
#         dist = np.sum((xyz - centroid) ** 2, -1)
#         mask = dist < distance
#         distance[mask] = dist[mask]
#         farthest = np.argmax(distance, -1)
#     point = point[centroids.astype(np.int32)]
#     return point

file_name = '/media/rpf/Elements/0322/0322_map/color_map.pcd'
pcd = o3d.io.read_point_cloud(file_name)
print(pcd)
o3d.visualization.draw_geometries([pcd])
# points = np.fromfile(file_name, dtype=np.float32).reshape(-1, 4)
# print(pcd)
# # o3d.visualization.draw_geometries([pcd])
# ########点云转array#######
# points = np.array(pcd.points)
# #======因为存在点为nan，这里做滤除=========#
# mask = ~np.isnan(points)
# points = points[mask.any(axis=1)]





# #=======读取图片===========#
# img_path = '/home/ktd/rpf_ws/HD_map/toPengfei/image/73.jpg'
# image = cv2.imread(img_path)
# col,row ,ch= image.shape
# # img = imageio.imread(img_path)
# #======读取标定参数=======#
# camera_Intrinsics = np.array([[1471.9977675539926, 0, 1470.8174108801904,0],
# [0, 1023.2380376854941, 762.8721428607233,0],
# [0, 0, 1,0]])
# Lidar_to_right_camera = np.array([[7.46762e-06,-0.999961,-0.00864178,-0.0106],
# [0.019212,0.00864028,-0.99977,-0.0556481],
# [0.999815,-0.000158478,0.0192114,-0.0814182],
# [0,0,0,1]])
# #======维度变换=========#
# points = points.transpose()
# #添加一个维度使其能与4*4外参矩阵相乘#
# reflect = np.ones((1,points.shape[1]))
# points = np.concatenate((points,reflect),axis=0)

# #======坐标系转换=======#
# camera_points = Lidar_to_right_camera.dot(points)
# #======滤除相机坐标系下z小于0的部分=======#
# mask = (camera_points[2,:]>0)
# camera_points = camera_points[:,mask]
# points = points[:,mask]
# #======转到像素坐标系========#
# pxis_points = camera_Intrinsics.dot(camera_points)
# pxis_points = pxis_points/pxis_points[2,:]





# #======滤除图像范围以外的点=======#
# res_points = []
# for  i in range(pxis_points.shape[1]):
#     c = int(np.round(pxis_points[0,i]))
#     r = int(np.round(pxis_points[1,i]))
#     if c < row and r < col and r > 0 and c > 0:
#         # pdb.set_trace()
#         point = [ points[0,i], points[1,i], points[2,i], int(pxis_points[0,i]), int(pxis_points[1,i]) ]
#         res_points.append(point)
# #======res_points存放的是最终滤除xyz的坐标以及其对应像素坐标=======#

# res_points = np.array(res_points)




# pxi = res_points[:,3:]


# # pdb.set_trace()

# output_cloud = res_points[:,:3]
# pcd = o3d.geometry.PointCloud()#创建一个PointCloud对象
# pcd.points = o3d.utility.Vector3dVector(output_cloud)#将矩阵变为open3d里面的数据格式
# print(pcd)

# # o3d.visualization.draw_geometries([pcd])
# #==============到这个地方是没有问题的===========#
# #==============到这个地方是没有问题的===========#
# #==============到这个地方是没有问题的===========#
# #==============到这个地方是没有问题的===========#





# #=============读取车道线检测结果=============#
# image = cv2.imread('/home/ktd/rpf_ws/HD_map/raw_image/001.jpg')
# img = cv2.imread("/home/ktd/rpf_ws/HD_map/res_mask1.png",0)
# kernel = np.ones((5,5),np.uint8)
# img = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)
# num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
# lane_index = []

# vis_mask = np.zeros((image.shape[0],image.shape[1]))
# for p in pxi:
#     vis_mask[int(p[1]),int(p[0])] = 255
#     # pdb.set_trace()
#     # pass
# cv2.imwrite('vis_mask.png',vis_mask)
# # cv2.imshow('vis_mask',vis_mask)
# # cv2.waitKey()
# # cv2.destroyAllWindows()

# #=============到这里可视化均没有问题==============#

# pdb.set_trace()

# #==============点云向图像投影的事情================#

# for p in pxi:
#     coordinate = (int(p[0]),int(p[1]))
#     # vis_mask[int(p[1]),int(p[0])] = 255
#     cv2.circle(img=image,center = coordinate,radius = 2,color=(0,0,255))

# # cv2.imshow('test',image)
# cv2.imwrite('refl.png',image)
# # cv2.waitKey()
# # cv2.destroyAllWindows()
# pdb.set_trace()


# #============到这里点云向图像投影是没有问题的============#

# for onepoint in res_points:
#     # pdb.set_trace()
#     val = labels[int(onepoint[4]),int(onepoint[3])]
#     lane_index.append(val)

# lane_index = np.array(lane_index)
# lane_index = np.expand_dims(lane_index,axis=1)

# result = np.concatenate((res_points,lane_index),axis=1)
# mask = (result[:,5]>0)
# result = result[mask,:]
# output_lanes = []
# for i in range(int(result[:,5].max())):
#     lane = []
# # lane = []
#     for lane_ in result:

#         if int(lane_[5]) == i+1:
#             lane.append(lane_)
#     lane = np.array(lane)
#     # pdb.set_trace()
#     if lane[:,0].max()-lane[:,0].min() > 20:
#         lane = farthest_point_sample(lane,10)
#     elif lane[:,0].max()-lane[:,0].min() > 10:
#         lane = farthest_point_sample(lane,5)
#     elif lane[:,0].max()-lane[:,0].min() > 5:
#         lane = farthest_point_sample(lane,3)
#     else:
#         lane = farthest_point_sample(lane,2)
#     # if lane.shape[0]>50:
#     #     lane = farthest_point_sample(lane,5)
#     # elif lane.shape[0]>20:
#     #     lane = farthest_point_sample(lane,4)
#     # elif lane.shape[0]>10:
#     #     lane = farthest_point_sample(lane,3)
#     # else:
#     #     lane = farthest_point_sample(lane,2)

#         # lane = lane[:10,:]
#     output_lanes.append(lane)
# pdb.set_trace()

# # output_lanes_array = np.array(output_lanes)
# #=======output_lanes为输出结果============#
# #=======output_lanes为列表
# #==车道线点：xyz三维坐标、xy像素坐标、车道线index===#

# #=======将点云三维坐标xyz以及车道线index存放在pcd文件======#
# vis_lanes = output_lanes[0]
# # point_color = output_lanes[0]
# for i  in range(1,len( output_lanes)):
#     # pdb.set_trace()
#     vis_lanes = np.concatenate((vis_lanes,output_lanes[i]),axis=0)    

#     # for i in range(int(result[:,5].max())):

# # color_value = max(min(255 - point_intensity[point_id], 255), 0) / 255.0 * 240.0
# vis_cloud = vis_lanes[:,:3]
# #============将每个点的xyz坐标提取出来==============#
# pdb.set_trace()

# vis_cloud = vis_lanes[:,:3]
# pdb.set_trace()
# point_color = np.ones((vis_lanes.shape[0],3))
# color = vis_lanes[:,5] * 240/2550
# point_color[:,0] = color
# point_color[:,1] = color
# point_color[:,2] = color

# point_index = np.ones((vis_lanes.shape[0],3))
# index = vis_lanes[:,5] 
# point_index[:,0] = index
# point_index[:,1] = index
# point_index[:,2] = index
# # point_color = np.ones((vis_lanes.shape[0],1))
# # point_index = vis_lanes[:,5]
# # point_color = vis_lanes[:,5]
# # point_color = np.expand_dims(point_color,axis=1)
# # point_index = np.expand_dims(point_index,axis=2)

# # point_color = np.expand_dims(point_color,axis=1)
# # point_color = np.expand_dims(point_color,axis=2)
# # pdb.set_trace()
# pcd = o3d.geometry.PointCloud()#创建一个PointCloud对象
# pcd.points = o3d.utility.Vector3dVector(vis_cloud)#将矩阵变为open3d里面的数据格式
# pcd.normals = o3d.utility.Vector3dVector(point_index)
# # pcd.colors = o3d.utility.Vector3dVector(point_color)
# # pcd.colors = o3d.utility.IntVector(point_index)
# #=======点数在前面！==========#
# print(pcd)
# path = 'lane.pcd'
# o3d.io.write_point_cloud(path, pcd, write_ascii=True)
# o3d.visualization.draw_geometries([pcd])
# #=======将点云三维坐标xyz以及车道线index存放在pcd文件======#

# #=======可视化======#
# vis_lanes = output_lanes[0]
# # point_color = output_lanes[0]
# for i  in range(1,len( output_lanes)):
#     # pdb.set_trace()
#     vis_lanes = np.concatenate((vis_lanes,output_lanes[i]),axis=0)    

#     # for i in range(int(result[:,5].max())):

# # color_value = max(min(255 - point_intensity[point_id], 255), 0) / 255.0 * 240.0
# vis_cloud = vis_lanes[:,:3]
# pdb.set_trace()
# point_color = np.ones((vis_lanes.shape[0],3))

# color = vis_lanes[:,5] * 240/2550
# point_color[:,0] = color
# point_color[:,1] = color
# point_color[:,2] = color
# # point_color = np.expand_dims(point_color,axis=1)
# # point_color = np.expand_dims(point_color,axis=2)
# # pdb.set_trace()
# pcd = o3d.geometry.PointCloud()#创建一个PointCloud对象
# pcd.points = o3d.utility.Vector3dVector(vis_cloud)#将矩阵变为open3d里面的数据格式
# pcd.colors = o3d.utility.Vector3dVector(point_color)
# #=======点数在前面！==========#
# print(pcd)
# path = 'lane.pcd'
# o3d.io.write_point_cloud(path, pcd, write_ascii=True)
# o3d.visualization.draw_geometries([pcd])


# pdb.set_trace()




# # CAM = 2
# # pdb.set_trace()
# # def load_velodyne_points(filename):
# #     points = np.fromfile(filename, dtype=np.float32).reshape(-1, 4)
# #     #points = points[:, :3]  # exclude luminance
# #     return points

# # def load_calib(calib_dir):
# #     # P2 * R0_rect * Tr_velo_to_cam * y
# #     lines = open(calib_dir).readlines()
# #     lines = [ line.split()[1:] for line in lines ][:-1]
# #     #
# #     P = np.array(lines[CAM]).reshape(3,4)
# #     #
# #     Tr_velo_to_cam = np.array(lines[5]).reshape(3,4)
# #     Tr_velo_to_cam = np.concatenate(  [ Tr_velo_to_cam, np.array([0,0,0,1]).reshape(1,4)  ]  , 0     )
# #     #
# #     R_cam_to_rect = np.eye(4)
# #     R_cam_to_rect[:3,:3] = np.array(lines[4][:9]).reshape(3,3)
# #     #
# #     P = P.astype('float32')
# #     Tr_velo_to_cam = Tr_velo_to_cam.astype('float32')
# #     R_cam_to_rect = R_cam_to_rect.astype('float32')
# #     return P, Tr_velo_to_cam, R_cam_to_rect

# # def prepare_velo_points(pts3d_raw):
# #     '''Replaces the reflectance value by 1, and tranposes the array, so
# #         points can be directly multiplied by the camera projection matrix'''
# #     pdb.set_trace()
# #     pts3d = pts3d_raw
# #     # Reflectance > 0
# #     indices = pts3d[:, 3] > 0
# #     pts3d = pts3d[indices ,:]
# #     pts3d[:,3] = 1
# #     return pts3d.transpose(), indices

# # def project_velo_points_in_img(pts3d, T_cam_velo, Rrect, Prect):
# #     '''Project 3D points into 2D image. Expects pts3d as a 4xN
# #         numpy array. Returns the 2D projection of the points that
# #         are in front of the camera only an the corresponding 3D points.'''
# #     # 3D points in camera reference frame.
# #     pdb.set_trace()
# #     pts3d_cam = Rrect.dot(T_cam_velo.dot(pts3d))
# #     # Before projecting, keep only points with z>0
# #     # (points that are in fronto of the camera).
# #     idx = (pts3d_cam[2,:]>=0)
# #     pts2d_cam = Prect.dot(pts3d_cam[:,idx])
# #     return pts3d[:, idx], pts2d_cam/pts2d_cam[2,:], idx


# # def align_img_and_pc(img_dir, pc_dir, calib_dir):
    
# #     img = imageio.imread(img_dir)
# #     pts = load_velodyne_points( pc_dir )
# #     P, Tr_velo_to_cam, R_cam_to_rect = load_calib(calib_dir)
# #     pdb.set_trace()
# #     pts3d, indices = prepare_velo_points(pts)
# #     pts3d_ori = pts3d.copy()
# #     reflectances = pts[indices, 3]
# #     pts3d, pts2d_normed, idx = project_velo_points_in_img( pts3d, Tr_velo_to_cam, R_cam_to_rect, P  )
# #     #print reflectances.shape, idx.shape
# #     reflectances = reflectances[idx]
# #     #print reflectances.shape, pts3d.shape, pts2d_normed.shape
# #     assert reflectances.shape[0] == pts3d.shape[1] == pts2d_normed.shape[1]

# #     rows, cols = img.shape[:2]

# #     points = []
# #     for i in range(pts2d_normed.shape[1]):
# #         c = int(np.round(pts2d_normed[0,i]))
# #         r = int(np.round(pts2d_normed[1,i]))
# #         if c < cols and r < rows and r > 0 and c > 0:
# #             color = img[r, c, :]
# #             point = [ pts3d[0,i], pts3d[1,i], pts3d[2,i], reflectances[i], color[0], color[1], color[2], pts2d_normed[0,i], pts2d_normed[1,i]  ]
# #             points.append(point)

# #     pdb.set_trace()
# #     points = np.array(points)

# #     return points

# # # update the following directories
# # IMG_ROOT = '/home/ktd/rpf_ws/VoxelNet-pytorch/data/training/image_2/'
# # PC_ROOT = '/home/ktd/rpf_ws/VoxelNet-pytorch/data/training/velodyne/'
# # CALIB_ROOT = '/home/ktd/rpf_ws/VoxelNet-pytorch/data/training/calib/'
# # PC_CROP_ROOT = '/home/ktd/rpf_ws/VoxelNet-pytorch/data/training/out/'


# # for frame in range(0, 20):
# #     img_dir = IMG_ROOT + '%06d.png' % frame
# #     pc_dir = PC_ROOT + '%06d.bin' % frame
# #     calib_dir = CALIB_ROOT + '%06d.txt' % frame
# #     # pdb.set_trace()
# #     points = align_img_and_pc(img_dir, pc_dir, calib_dir)
    
# #     output_name = PC_CROP_ROOT + '%06d.bin' % frame
# #     print('Save to %s' % output_name)
# #     points[:,:4].astype('float32').tofile(output_name)