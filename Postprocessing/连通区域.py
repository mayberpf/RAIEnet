import cv2
import numpy as np
import pdb
# 4邻域的连通域和 8邻域的连通域
# [row, col]
NEIGHBOR_HOODS_4 = True
OFFSETS_4 = [[0, -1], [-1, 0], [0, 0], [1, 0], [0, 1]]

NEIGHBOR_HOODS_8 = False
OFFSETS_8 = [[-1, -1], [0, -1], [1, -1],
             [-1,  0], [0,  0], [1,  0],
             [-1,  1], [0,  1], [1,  1]]
#第二遍扫描
def reorganize(binary_img: np.array):
    index_map = []
    points = []
    index = -1
    rows, cols = binary_img.shape
    for row in range(rows):
        for col in range(cols):
            var = binary_img[row][col]
            if var < 0.5:
                continue
            if var in index_map:
                index = index_map.index(var)
                num = index + 1
            else:
                index = len(index_map)
                num = index + 1
                index_map.append(var)
                points.append([])
            binary_img[row][col] = num
            points[index].append([row, col])
    #print(binary_img)
    #print(points)
    return binary_img, points

#四领域或八领域判断
def neighbor_value(binary_img: np.array, offsets, reverse=False):
    rows, cols = binary_img.shape
    label_idx = 0
    rows_ = [0, rows, 1] if reverse == False else [rows-1, -1, -1]
    cols_ = [0, cols, 1] if reverse == False else [cols-1, -1, -1]
    for row in range(rows_[0], rows_[1], rows_[2]):
        for col in range(cols_[0], cols_[1], cols_[2]):
            # pdb.set_trace()
            label = 256
            if binary_img[row][col] < 0.5:
                continue
            for offset in offsets:
                neighbor_row = min(max(0, row+offset[0]), rows-1)
                neighbor_col = min(max(0, col+offset[1]), cols-1)
                neighbor_val = binary_img[neighbor_row, neighbor_col]
                if neighbor_val < 0.5:
                    continue
                label = neighbor_val if neighbor_val < label else label
            # pdb.set_trace()
            if label == 255:
                label_idx += 1
                label = label_idx
            binary_img[row][col] = label
    print('第一遍扫描：',binary_img)
    print('开始第二遍...')
    return binary_img

# binary_img: bg-0, object-255; int
#第一遍扫描
def Two_Pass(binary_img: np.array, neighbor_hoods):
    if neighbor_hoods == NEIGHBOR_HOODS_4:
        offsets = OFFSETS_4
    elif neighbor_hoods == NEIGHBOR_HOODS_8:
        offsets = OFFSETS_8
    else:
        raise ValueError

    binary_img = neighbor_value(binary_img, offsets, False)

    return binary_img

if __name__ == "__main__":
    #创建四行七列的矩阵
    # pdb.set_trace()
    binary_img = np.zeros((4, 7), dtype=np.int16)
    #指定点设置为255
    index = [[0, 2], [0, 5],
            [1, 0], [1, 1], [1, 2], [1, 4], [1, 5], [1, 6],
            [2, 2], [2, 5],
            [3, 1], [3, 2], [3, 4],[3,5], [3, 6]]
    for i in index:
        binary_img[i[0], i[1]] = np.int16(255)

    print("原始二值图像")
    print(binary_img)
    # pdb.set_trace()
    #print("Two_Pass")
    #调用Two Pass算法，计算第一遍扫面的结果
    binary_img = Two_Pass(binary_img, NEIGHBOR_HOODS_4)
    #print(binary_img)
    #计算第一遍扫面的结果
    # pdb.set_trace()
    binary_img, points = reorganize(binary_img)
    print(binary_img)
    print(points)