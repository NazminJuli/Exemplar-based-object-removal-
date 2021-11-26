import pywt
import skimage.io
from cr.sparse import lop
from PIL import Image
from skimage.color import rgb2gray, rgba2rgb
from skimage.util import img_as_ubyte
import numpy as np
import matplotlib.pyplot as plt
import cv2

img = Image.open("/1/raw_man.jpg")
# img = Image.open("/home/kow/CutOutWiz/Projects/pythonProject/exemplar_inpainting/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/resources/beauty-portrait-young-brunettete-woman.png")

print(img.size, img.mode)
img = np.array(img).astype('uint8')
h,w,ch = img.shape
if (ch==4):
    img = rgba2rgb(img)
    print('rgba converted to rgb')
img = rgb2gray(img)

coefficients = pywt.dwt2(img, 'haar')
cA, (cH, cV, cD) = coefficients

coefficients = pywt.dwt2(cA, 'haar')
cA1, (cH1, cV1, cD1) = coefficients
coefficients = pywt.dwt2(cA1, 'haar')
cA2, (cH2, cV2, cD2) = coefficients
coefficients = pywt.dwt2(cA2, 'haar')
cA3, (cH3, cV3, cD3) = coefficients
# img = np.array(img).astype('float32')
# coefficients = lop.dwt2D(img.shape, wavelet='haar', level=4)
# coefficients = lop.jit(coefficients)
# coefficients = coefficients.times(img)
# print(len(coefficients))
# print(len(cA), len(cA1), len(cA2), len(cA3))
print("nd array",isinstance(cA, (np.ndarray)))
#
# s = np.sum(cA)
# s1 = np.sum(cA1)
# s2 = np.sum(cA2)
# s3 = np.sum(cA3)
# xx = s+s1+s2+s3
# E_low = np.sum([s, s1, s2, s3])
# print("energy low:", E_low, xx)
# print(s,s1,s2,s3)
#
# s = np.sum(cH)
# s1 = np.sum(cH1)
# s2 = np.sum(cH2)
# s3 = np.sum(cH3)
# H_high = np.sum([s, s1, s2, s3])
# print("energy high H:", H_high)
# print(s,s1,s2,s3)
#
# s = np.sum(cV)
# s1 = np.sum(cV1)
# s2 = np.sum(cV2)
# s3 = np.sum(cV3)
# V_high = np.sum([s, s1, s2, s3])
# print("energy high V:", V_high)
# print(s,s1,s2,s3)
#
# s = np.sum(cD)
# s1 = np.sum(cD1)
# s2 = np.sum(cD2)
# s3 = np.sum(cD3)
# D_high = np.sum([s, s1, s2, s3])
# print("energy high D:", D_high)
# print(s,s1,s2,s3)
# # print(sum(sum(cA)))

# freq = E_low / (E_low + H_high + V_high + D_high)
# print("frequency :", freq)
# freq_percentage =  (freq * 100) / (E_low + H_high + V_high + D_high)
# print("frequency % :", freq_percentage)

print("ndarray", np.ndarray.sum(cA))
print("ndarray", np.ndarray.sum(cA1))
print("ndarray", np.ndarray.sum(cA2))
print("ndarray", np.ndarray.sum(cA3))
# print("ndarray", np.ndarray.sum(cA))
s_approximation = np.ndarray.sum(cA) + np.ndarray.sum(cA1) + np.ndarray.sum(cA2) + np.ndarray.sum(cA3)
h_approximation = np.ndarray.sum(cH) + np.ndarray.sum(cH1) + np.ndarray.sum(cH2) + np.ndarray.sum(cH3)
v_approximation = np.ndarray.sum(cV) + np.ndarray.sum(cV1) + np.ndarray.sum(cV2) + np.ndarray.sum(cV3)
d_approximation = np.ndarray.sum(cD) + np.ndarray.sum(cD1) + np.ndarray.sum(cD2) + np.ndarray.sum(cD3)

print("total low:", s_approximation, h_approximation, v_approximation, d_approximation)
# print("total higher", s_approximation + h_approximation + v_approximation + d_approximation)
sum_higher_frequency = s_approximation + h_approximation + v_approximation + d_approximation
frequency = s_approximation / sum_higher_frequency
per = (frequency * 100) /  sum_higher_frequency
print("freq: per", frequency, per)
# sH = np.sum(cH, dtype='uint8')
# sH1 = np.sum(cH1, dtype='uint8')
# sH2 = np.sum(cH2, dtype='uint8')
# sH3 = np.sum(cH3, dtype='uint8')
# H_low = np.sum([sH, sH1, sH2, sH3], dtype='uint8')
# print("Horizontal low:", H_low)

fig = plt.figure(figsize=(25, 15))
titles = ['Approximation', ' Horizontal detail',
          'Vertical detail', 'Diagonal detail']
# cv2.imshow('b', cA3)
# cv2.imshow('h', cH3)
# cv2.imshow('v', cV3)
# cv2.imshow('d', cD3)
# cv2.waitKey(0)

for i, a in enumerate([cA3, cH3, cV3, cD3]):
    ax = fig.add_subplot(1, 4, i + 1)
    ax.imshow(a, interpolation="nearest", cmap=plt.cm.gray)

    ax.set_title(titles[i], fontsize=12)
    ax.set_xticks([])
    ax.set_yticks([])
    name = str(i) + '_.png'
    plt.savefig(name)

plt.show()
