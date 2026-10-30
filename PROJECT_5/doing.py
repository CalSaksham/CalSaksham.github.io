import json
import numpy as np
import matplotlib.pyplot as plt

def computeH(im1_pts, im2_pts):
    inp = []
    out = []
 
    for i in range(len(im1_pts)):
        x = im1_pts[i][0]
        y = im1_pts[i][1]
        xout = im2_pts[i][0]
        yout = im2_pts[i][1]
        inp.append([x, y, 1, 0, 0, 0, -x*xout, -y*xout])
        out.append(xout)
        inp.append([0, 0, 0, x, y, 1, -x*yout, -y*yout])
        out.append(yout)    
    print(inp)
    print(out)
    h = np.linalg.lstsq(inp, out)[0]
    arr = [[h[0], h[1], h[2]],[h[3], h[4], h[5]],[h[6], h[7], 1]]
    H = np.array(arr)
    return H

for x in range(2):
    if x == 0:
        with open("Correspondence Tool copy.json") as f:
            js = json.load(f)
        im1_pts = js["im1Points"]
        im2_pts = js["im2Points"]
    if x == 1:
        with open("Correspondence Tool Data copy.json") as f:
            js = json.load(f)
        im1_pts = js["im1Points"][-8:]
        im2_pts = js["im2Points"][-8:]

    H = computeH(im1_pts, im2_pts)
    print(H)

    for name, points in [(js["im1_name"], im1_pts),
                         (js["im2_name"], im2_pts)]:
        plt.figure()
        image = plt.imread(name + ".jpeg")

        if x == 0:
            image = np.rot90(image, k=-1)
        plt.imshow(image)

        for i, (px, py) in enumerate(points):
            plt.text(px, py, str(i + 1), color="orange")
        plt.title(name)
        plt.show()



def warpImageNearestNeighbor(im, H):
    h, w = im.shape[0], im.shape[1]
    corners = np.array([
        [0, 0, 1],
        [w - 1, 0, 1],
        [w - 1, h - 1, 1],
        [0, h - 1, 1]
    ])

    corners = np.dot(H, corners.T)
    for i in range(4):
        scale = corners[2][i]
        corners[0][i] = corners[0][i] / scale
        corners[1][i] = corners[1][i] / scale
        corners[2][i] = 1

    xmin = np.floor(min(corners[0]))
    xmax = np.ceil(max(corners[0]))
    ymin = np.floor(min(corners[1]))
    ymax = np.ceil(max(corners[1]))

    warped = np.zeros(
        (int(ymax - ymin + 1), int(xmax - xmin + 1), 3), dtype = im.dtype)

    Hinv = np.linalg.inv(H)

    for y in range(warped.shape[0]):
        for x in range(warped.shape[1]):
            p = np.dot(Hinv, np.array([x + xmin, y + ymin, 1]))

            old_x = int(round(p[0] / p[2]))
            old_y = int(round(p[1] / p[2]))
            if 0 <= old_x and old_x < w and 0 <= old_y and old_y < h:
                warped[y, x] = im[old_y, old_x]

    return warped

def warpImageBilinear(im, H):
    h, w = im.shape[0], im.shape[1]
    corners = np.array([
        [0, 0, 1],
        [w - 1, 0, 1],
        [w - 1, h - 1, 1],
        [0, h - 1, 1]
    ])

    corners = np.dot(H, corners.T)
    for i in range(4):
        scale = corners[2][i]
        corners[0][i] = corners[0][i] / scale
        corners[1][i] = corners[1][i] / scale
        corners[2][i] = 1

    xmin = np.floor(min(corners[0]))
    xmax = np.ceil(max(corners[0]))
    ymin = np.floor(min(corners[1]))
    ymax = np.ceil(max(corners[1]))

    warped = np.zeros(
        (int(ymax - ymin + 1), int(xmax - xmin + 1), 3),
        dtype=im.dtype
    )

    Hinv = np.linalg.inv(H)

    for y in range(warped.shape[0]):
        for x in range(warped.shape[1]):
            p = np.dot(Hinv, np.array([x + xmin, y + ymin, 1]))

            old_x = p[0] / p[2]
            old_y = p[1] / p[2]

            if 0 <= old_x <= w - 1 and 0 <= old_y <= h - 1:
                x0 = int(old_x)
                y0 = int(old_y)
                x1 = min(x0 + 1, w - 1)
                y1 = min(y0 + 1, h - 1)

                dx = old_x - x0
                dy = old_y - y0

                top = (1 - dx) * im[y0, x0] + dx * im[y0, x1]
                bottom = (1 - dx) * im[y1, x0] + dx * im[y1, x1]

                warped[y, x] = (1 - dy) * top + dy * bottom

    return warped

# im = plt.imread("IMG_8089.jpeg")

# nn = warpImageNearestNeighbor(im, H)
# bil = warpImageBilinear(im, H)

# plt.figure()
# plt.imshow(nn)
# plt.title("Nearest neighbor")

# plt.figure()
# plt.imshow(bil)
# plt.title("Bilinear")
# plt.show()

# width = 1920
# height = 1080

# im2_pts = np.array([
#     [0, 0],
#     [width - 1, 0],
#     [width - 1, height - 1],
#     [0, height - 1]
# ])

# for name in ["IMG_8096", "IMG_8097", "IMG_8098"]:
#     im = plt.imread(name + ".jpeg")

#     if name == "IMG_8098":
#         im = np.rot90(im, k=1)
#     else:
#         im = np.rot90(im, k=-1)

#     plt.figure()
#     plt.imshow(im)
#     plt.title(name + "corners")
#     im1_pts = np.array(plt.ginput(4, timeout=0))
#     plt.close()

#     H = computeH(im1_pts, im2_pts)

#     nn = warpImageNearestNeighbor(im, H)
#     bil = warpImageBilinear(im, H)

#     plt.figure()
#     plt.imshow(nn)
#     plt.title("Nearest")

#     plt.figure()
#     plt.imshow(bil)
#     plt.title("Bilinear")

#     plt.show()



def weight(im):
    h, w = im.shape[0], im.shape[1]
    weights = np.zeros(im.shape, dtype=np.float32)

    for y in range(h):
        for x in range(w):
            left = x + 1
            right = w - x
            top = y + 1
            bottom = h - y
            weights[y, x] = min(left, right, top, bottom)
    return weights / weights.max()


def makeMosaic(im1, im2, H):
    h1, w1 = im1.shape[0], im1.shape[1]
    h2, w2 = im2.shape[0], im2.shape[1]

    corners = np.array([
        [0, 0, 1],
        [w1 - 1, 0, 1],
        [w1 - 1, h1 - 1, 1],
        [0, h1 - 1, 1]
    ])

    corners = np.dot(H, corners.T)

    for i in range(4):
        scale = corners[2][i]
        corners[0][i] = corners[0][i] / scale
        corners[1][i] = corners[1][i] / scale
        corners[2][i] = 1

    left = int(np.floor(min(corners[0])))
    top = int(np.floor(min(corners[1])))

    xmin = min(left, 0)
    ymin = min(top, 0)
    right = int(np.ceil(max(corners[0])))
    bottom = int(np.ceil(max(corners[1])))

    xmax = max(right, w2 - 1)
    ymax = max(bottom, h2 - 1)

    weight1 = weight(im1)
    weight2 = weight(im2)

    first = warpImageBilinear(im1 * weight1, H)
    weight1 = warpImageBilinear(weight1, H)

    mosaic = np.zeros(
        (ymax - ymin + 1, xmax - xmin + 1, 3)
    )
    total = np.zeros(mosaic.shape)
    x = left - xmin
    y = top - ymin
    h, w = first.shape[0], first.shape[1]

    mosaic[y:y+h, x:x+w] = first
    total[y:y+h, x:x+w] = weight1
    x = -xmin
    y = -ymin

    mosaic[y:y+h2, x:x+w2] += im2 * weight2
    total[y:y+h2, x:x+w2] += weight2

    for y in range(mosaic.shape[0]):
        for x in range(mosaic.shape[1]):
            if total[y, x][0] > 0:
                mosaic[y, x] = mosaic[y, x] / total[y, x]

    return np.clip(np.round(mosaic), 0, 255).astype(im1.dtype)

with open("Correspondence Tool copy.json") as f:
    js = json.load(f)

im1 = plt.imread("IMG_8081.jpeg")
im2 = plt.imread("IMG_8082.jpeg")
im1 = im1[::4, ::4]
im2 = im2[::4, ::4]
pts1 = np.array(js["im1Points"]) / 4
pts2 = np.array(js["im2Points"]) / 4
H = computeH(pts1, pts2)
im1 = np.rot90(im1, k=-1)
im2 = np.rot90(im2, k=-1)

H = computeH(pts1, pts2)
mosaic1 = makeMosaic(im1, im2, H)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(im1)
plt.subplot(1, 3, 2)
plt.imshow(im2)
plt.subplot(1, 3, 3)
plt.imshow(mosaic1)
plt.title("Mosaic 1")
plt.show()

with open("Correspondence Tool Data copy.json") as f:
    js = json.load(f)

im1 = plt.imread("IMG_8089.jpeg")
im2 = plt.imread("IMG_8090.jpeg")
im1 = im1[::4, ::4]
im2 = im2[::4, ::4]
pts1 = np.array(js["im1Points"][-8:]) / 4
pts2 = np.array(js["im2Points"][-8:]) / 4

H = computeH(pts1, pts2)
mosaic2 = makeMosaic(im1, im2, H)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(im1)
plt.subplot(1, 3, 2)
plt.imshow(im2)
plt.subplot(1, 3, 3)
plt.imshow(mosaic2)
plt.title("Mosaic 2")
plt.show()

with open("Correspondence Tool Data (1).json") as f:
    js = json.load(f)
im1 = plt.imread("IMG_0846.jpeg")
im2 = plt.imread("IMG_0847.jpeg")
im1 = im1[::4, ::4]
im2 = im2[::4, ::4]
pts1 = np.array(js["im1Points"]) / 4
pts2 = np.array(js["im2Points"]) / 4

H = computeH(pts1, pts2)
mosaic3 = makeMosaic(im1, im2, H)

plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(im1)
plt.subplot(1, 3, 2)
plt.imshow(im2)
plt.subplot(1, 3, 3)
plt.imshow(mosaic3)
plt.title("Mosaic 3")
plt.show()