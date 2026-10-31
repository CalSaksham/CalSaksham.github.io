import cv2
import numpy as np
import scipy


img_array = cv2.imread('Pizza.jpeg', 0).astype(np.float64) / 255
img_array_2 = cv2.imread('sun.jpg', 0).astype(np.float64) / 255

scale = 400 / max(img_array.shape)
if scale < 1:
    img_array = cv2.resize(img_array, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
print(f"Image size: {img_array.shape}")
scale = 400 / max(img_array_2.shape)
if scale < 1:
    img_array_2 = cv2.resize(img_array_2, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
print(f"Image size: {img_array_2.shape}")

def load_img(img_name):
    img_array = cv2.imread(img_name, 0).astype(np.float64) / 255
    scale = 400 / max(img_array.shape)
    if scale < 1:
        img_array = cv2.resize(img_array, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return img_array


def convolute(img, filter):

    out = np.zeros((img.shape[0]+(filter.shape[0]-1), img.shape[1] + (filter.shape[1]-1)))
    for i in range(len(img)):
        for j in range(len(img[0])):
            for x in range(len(filter)-1,-1,-1):
                for y in range(len(filter[0])-1,-1,-1):
                    out[i+x][j+y] += img[i][j]*filter[x][y]
        
    return out

def convolute2for(img, filter):

    out = np.zeros((img.shape[0]+(filter.shape[0]-1), img.shape[1] + (filter.shape[1]-1)))
    pad = np.zeros((img.shape[0]+ 2*(filter.shape[0]-1), img.shape[1] + 2* (filter.shape[1]-1)))
    pad[filter.shape[0]-1 : filter.shape[0]-1+img.shape[0], filter.shape[1]-1 : filter.shape[1]-1+img.shape[1]] = img

    filter = np.flip(filter)

    for i in range(len(out)):
        for j in range(len(out[0])):
            out[i][j] += np.sum(filter * pad[i:i+len(filter), j:j+len(filter[0])])
        
    return out

def binarize(img):
    for i in range(len(img)):
        for j in range(len(img[0])):
            if img[i][j] <0.1:
                img[i][j] = 0
            else:
                img[i][j] = 1
        

def edge_detector(img):
    dx = scipy.signal.convolve2d(img, Dx, mode='same')
    dy = scipy.signal.convolve2d(img, Dy, mode='same')
    effective = np.sqrt(dx**2 + dy**2)
    return effective


def sharp_it(img, x):
    blur = scipy.signal.convolve2d(img, gaussian_kernel, mode='same')
    high = img - blur
    identity = np.zeros_like(gaussian_kernel)
    identity[gaussian_kernel.shape[0] // 2, gaussian_kernel.shape[1] // 2] = 1
    unsharp = (identity * (1 + x)) - (x * gaussian_kernel)
    sharp = np.clip(scipy.signal.convolve2d(img, unsharp, mode='same'), 0, 1)
    return blur, high, sharp

def gaussian_stack(img, dim):
    gaussian_kernel = np.dot(cv2.getGaussianKernel(49,8), cv2.getGaussianKernel(49,8).T)
    out = np.zeros((img.shape[0], img.shape[1], dim))
    curr = img
    for i in range(dim):
        out[:,:,i] = curr
        curr = scipy.signal.convolve2d(curr, gaussian_kernel, mode='same')
    return out

def laplace_stack(img, dim):
    gauss_stack = gaussian_stack(img, dim)
    out = np.zeros((img.shape[0], img.shape[1], dim))
    for i in range(dim-1):
        out[:,:,i] += gauss_stack[:,:,i] - gauss_stack[:,:,i+1]
    out[:,:,-1] = gauss_stack[:,:,-1]
    return out

def match_size(img1, img2):
    return cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

def merged_img(img1, img2, mask,dim):
    img2 = match_size(img1, img2)
    if mask == 'vert':
        mask = np.zeros_like(img1)
        t = mask[:, :mask.shape[1]//2]
        mask[:, :mask.shape[1]//2] += np.ones_like(t)

    elif mask == 'circ':
        mask = np.zeros_like(img1)
        cv2.circle(mask, (img1.shape[1]//2, img1.shape[0]//2), 115, 1, -1)


    mask_gauss = gaussian_stack(mask, dim+2)
    im1_place = laplace_stack(img1, dim)
    im2_place = laplace_stack(img2, dim)

    # start = np.zeros_like(img1)
    # start[:start.shape[0]//2, :start.shape[1]//2] += img1[:start.shape[0]//2, :start.shape[1]//2]
    # start[start.shape[0]//2:, start.shape[1]//2 :] += img2[start.shape[0]//2 :, start.shape[1]//2 :]

    out = np.zeros((img1.shape[0], img1.shape[1], dim))
    a = np.zeros((img1.shape[0], img1.shape[1], dim))
    b = np.zeros((img1.shape[0], img1.shape[1], dim))
    for i in range(dim):
        a[:,:,i] = im1_place[:,:,i]* mask_gauss[:,:,i+2]
        b[:,:,i] = im2_place[:,:,i]*(1-mask_gauss[:,:,i+2])
        out[:,:,i]  = im1_place[:,:,i]*mask_gauss[:,:,i+2] + im2_place[:,:,i]*(1-mask_gauss[:,:,i+2])
    return a, b, out




box = np.ones((9, 9)) / 81
Dx = np.array([[1, 0, -1]])
Dy = Dx.T
gaussian_kernel = np.dot(cv2.getGaussianKernel(9,1), cv2.getGaussianKernel(9,1).T)

img_array = load_img('DSC00868_Original 2.jpeg')

res_dx = convolute(img_array, Dx)
res_dy = convolute(img_array, Dy)
res_box = convolute(img_array, box)

scipy_dx = scipy.signal.convolve2d(img_array, Dx, mode = 'full')
scipy_dy = scipy.signal.convolve2d(img_array, Dy, mode = 'full')
scipy_box = scipy.signal.convolve2d(img_array, box, mode = 'full')

img_array = load_img('DSC00868_Original 2.jpeg')

res_dx = convolute(img_array, Dx)
res_dy = convolute(img_array, Dy)
res_box = convolute(img_array, box)

scipy_dx = scipy.signal.convolve2d(img_array, Dx, mode = 'full')
scipy_dy = scipy.signal.convolve2d(img_array, Dy, mode = 'full')
scipy_box = scipy.signal.convolve2d(img_array, box, mode = 'full')

cv2.imshow('result_dx', res_dx+0.5)
cv2.imshow('scipy_dx', scipy_dx+0.5)
cv2.imshow('result_dy', res_dy+0.5)
cv2.imshow('scipy_dy', scipy_dy+0.5)
cv2.imshow('result_box', res_box)
cv2.imshow('scipy_box', scipy_box)

img_array = load_img('Cameraman Image.png')

res_dx = convolute(img_array, Dx)
res_dy = convolute(img_array, Dy)
cv2.imshow('original_camera', img_array)
cv2.imshow('result_dx_camera', res_dx+0.5)
cv2.imshow('result_dy_camera', res_dy+0.5)

effective = edge_detector(img_array)
cv2.imshow('effective_camera', effective)
binarize(effective)
cv2.imshow('binarized_camera', effective)



gauss_x = convolute2for(convolute2for(img_array, gaussian_kernel), Dx)
gauss_y = convolute2for(convolute2for(img_array, gaussian_kernel), Dy)
gauss = edge_detector(convolute2for(img_array, gaussian_kernel))
binarize(gauss)

cv2.imshow('gaussx', gauss_x+0.5)
cv2.imshow('gaussy', gauss_y+0.5)
gauss = edge_detector(convolute2for(img_array, gaussian_kernel))
cv2.imshow('blur magnitude', gauss)
binarize(gauss)
cv2.imshow('blur binarized', gauss)


DoGx = convolute2for(gaussian_kernel, Dx)
DoGy = convolute2for(gaussian_kernel, Dy)

gauss_Gx = convolute2for(img_array, convolute2for(gaussian_kernel, Dx))
gauss_Gy = convolute2for(img_array, convolute2for(gaussian_kernel, Dy))

import matplotlib.pyplot as plt

plt.imshow(DoGx, cmap='gray')
plt.title('DoGx')
plt.show()

plt.imshow(DoGy, cmap='gray')
plt.title('DoGy')
plt.show()

cv2.imshow('gaussGx', gauss_Gx+0.5)
cv2.imshow('gaussGy', gauss_Gy+0.5)


img_array = load_img('Taj (1).jpg')
for x in [0.5, 1, 2, 5]:
      blur, high, sharp = sharp_it(img_array, x)
      cv2.imshow(f'taj sharp {x}', sharp)
cv2.imshow('taj original', img_array)
cv2.imshow('taj blur', blur)
cv2.imshow('taj high', high + 0.5)


img_array = load_img('UC Berkeley DSC 2783.jpg')
blur, high, sharp = sharp_it(img_array, 5)
_, _, resharp = sharp_it(blur, 5)
cv2.imshow('blur1', blur)
cv2.imshow('high1', high+0.5)
cv2.imshow('sharp1', sharp)
cv2.imshow('original1', img_array)
cv2.imshow('resharp1', resharp)

img_array = load_img('virat.jpg')
blur, high, sharp = sharp_it(img_array, 5)
cv2.imshow('blur2', blur)
cv2.imshow('high2', high+0.5)
cv2.imshow('sharp2', sharp)
cv2.imshow('original2', img_array)



img_array = load_img('apple.jpeg')
img_array_2 = load_img('orange.jpeg')

a, b, out_stack = merged_img(img_array, img_array_2, 'vert', 15)
g = gaussian_stack(img_array, 15)
l = laplace_stack(img_array, 15)
for i in range(5):
    cv2.imshow(f'apple gauss {i}', g[:,:,i])
    cv2.imshow(f'apple laplace {i}', l[:,:,i] + 0.5)
g = gaussian_stack(img_array_2, 15)
l = laplace_stack(img_array_2, 15)
for i in range(5):
    cv2.imshow(f'orange gauss {i}', g[:,:,i])
    cv2.imshow(f'orange laplace {i}', l[:,:,i] + 0.5)


cv2.imshow('apple 0', a[:,:,0] + 0.5)
cv2.imshow('orange 0', b[:,:,0] + 0.5)
cv2.imshow('blend 0', out_stack[:,:,0] + 0.5)

cv2.imshow('apple 2', a[:,:,2] + 0.5)
cv2.imshow('orange 2', b[:,:,2] + 0.5)
cv2.imshow('blend 2', out_stack[:,:,2] + 0.5)

cv2.imshow('apple 4', a[:,:,4] + 0.5)
cv2.imshow('orange 4', b[:,:,4] + 0.5)
cv2.imshow('blend 4', out_stack[:,:,4] + 0.5)

cv2.imshow('apple sum', a.sum(axis=2))
cv2.imshow('orange sum', b.sum(axis=2))
cv2.imshow('blend sum', out_stack.sum(axis=2))


img_array = load_img('saksham.jpg')
img_array_2 = load_img('rhea.jpg')

_, _, out_stack = merged_img(img_array, img_array_2, 'vert', 15)
cv2.imshow('rhesham', out_stack.sum(axis=2))


img_array_2 = load_img('sun.jpg')
img_array = load_img('Pizzea.jpg')

_, _, out_stack  = merged_img(img_array, img_array_2, 'circ', 15)

cv2.imshow('sizza', out_stack.sum(axis=2))


cv2.waitKey(0)