# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 22:15:57 2018

@author: Administrator
"""

import cv2
import numpy as np


def process(img) :
    static_size = 1000
    scale = static_size / np.max(img.shape)
    real_scale = np.max(img.shape) / static_size
    h, w = img.shape
    smimg = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    smimg = cv2.cvtColor(smimg, cv2.COLOR_GRAY2RGB)
    smimg = cv2.edgePreservingFilter(smimg, flags=1, sigma_s = 10, sigma_r=0.53)
    
    smimg= cv2.cvtColor(smimg, cv2.COLOR_RGB2GRAY)
    cv2.imwrite("smimg.jpg", smimg)
    
    
    

class _static_ :
    img_p = "./target_images/(313,901)(2750,1016)(2730,3026)(244,3014).jpg"
    
def get_max_width_height(tr, tl, br, bl) :
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
     
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
     
    max_width = max(int(width_a), int(width_b))
    max_height = max(int(height_a), int(height_b))
    return max_width, max_height    


tl = [313, 901]
tr = [2750, 1016]
br = [2730, 3026]
bl = [244, 3014]

img = cv2.imread(_static_.img_p, 0)
img_ori = img.copy()

max_width, max_height = get_max_width_height(tr, tl, br, bl)
src_p = np.array([tl, tr, br, bl], dtype="float32")
dst_p = np.array([[0, 0], [max_width - 1, 0], 	[max_width - 1, max_height - 1], [0, max_height - 1]], dtype = "float32") 

# warp transformation
matrix = cv2.getPerspectiveTransform(src_p, dst_p)
warped = cv2.warpPerspective(img, matrix, (max_width, max_height), borderMode=cv2.BORDER_TRANSPARENT)

process(warped)

h_gap = int(warped.shape[0] / 10)
x_sp = 0
x_ep = warped.shape[1]
y_pos = 0
while y_pos < warped.shape[0] :
    cv2.line(warped, (x_sp, y_pos), (x_ep, y_pos), (125, 125, 125), thickness=5)
    y_pos += 250

# reverse transformation
img = cv2.warpPerspective(warped, matrix, dsize=(img.shape[1], img.shape[0]), flags=cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_TRANSPARENT )


ind = np.where(img[:, :] == 125)
img_ori[ind] = img[ind]

cv2.imwrite("./test.jpg", img_ori)






