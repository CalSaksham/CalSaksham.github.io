import json
import numpy as np
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from scipy.ndimage import gaussian_filter
from skimage.feature import corner_harris, peak_local_max

def get_harris_corners(im, edge_discard=20):
    """
    This function takes a b&w image and an optional amount to discard
    on the edge (default is 5 pixels), and finds all harris corners
    in the image. Harris corners near the edge are discarded and the
    coordinates of the remaining corners are returned. A 2d array (h)
    containing the h value of every pixel is also returned.

    h is the same shape as the original image, im.
    coords is 2 x n (ys, xs).
    """

    assert edge_discard >= 20

    # find harris corners
    h = corner_harris(im, method='eps', sigma=1)
    coords = peak_local_max(h, min_distance=1)

    # discard points on edge
    edge = edge_discard  # pixels
    mask = (coords[:, 0] > edge) & \
           (coords[:, 0] < im.shape[0] - edge) & \
           (coords[:, 1] > edge) & \
           (coords[:, 1] < im.shape[1] - edge)
    coords = coords[mask].T
    return h, coords


def dist2(x, c):
    """
    dist2  Calculates squared distance between two sets of points.

    Description
    D = DIST2(X, C) takes two matrices of vectors and calculates the
    squared Euclidean distance between them.  Both matrices must be of
    the same column dimension.  If X has M rows and N columns, and C has
    L rows and N columns, then the result has M rows and L columns.  The
    I, Jth entry is the  squared distance from the Ith row of X to the
    Jth row of C.

    Adapted from code by Christopher M Bishop and Ian T Nabney.
    """

    ndata, dimx = x.shape
    ncenters, dimc = c.shape
    assert dimx == dimc, 'Data dimension does not match dimension of centers'

    return (np.ones((ncenters, 1)) * np.sum((x**2).T, axis=0)).T + \
            np.ones((   ndata, 1)) * np.sum((c**2).T, axis=0)    - \
            2 * np.inner(x, c)


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
    h = np.linalg.lstsq(inp, out)[0]
    arr = [[h[0], h[1], h[2]],[h[3], h[4], h[5]],[h[6], h[7], 1]]
    H = np.array(arr)
    return H


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


def anms(h, coords, n=500):
    corners = []
    for k in range(len(coords[0])):
        y = coords[0][k]
        x = coords[1][k]
        corners.append([h[y, x], x, y])

    corners.sort(reverse=True)
    s = np.array([c[0] for c in corners])
    xs = np.array([c[1] for c in corners])
    ys = np.array([c[2] for c in corners])

    radii = []
    for i in range(len(corners)):
        d = (xs[:i] - xs[i])**2 + (ys[:i] - ys[i])**2
        d = d[s[i] < 0.9 * s[:i]]
        if len(d) > 0:
            r = d.min()
        else:
            r = float("inf")
        radii.append([r, xs[i], ys[i]])

    radii.sort(reverse=True)
    pts = []
    for r, x, y in radii[:n]:
        pts.append([x, y])
    return np.array(pts)


def descriptors(gray, pts):
    blur = gaussian_filter(gray, 2)
    out = []
    for x, y in pts:
        patch = blur[y-20:y+20, x-20:x+20]
        small = patch[::5, ::5]
        small = (small - small.mean()) / small.std()
        out.append(small.flatten())
    return np.array(out)


def match(desc1, desc2, ratio=0.7):
    matches = []
    for i in range(len(desc1)):
        first = float("inf")
        second = float("inf")
        best = -1
        for j in range(len(desc2)):
            dist = np.sqrt(np.sum((desc1[i] - desc2[j])**2))
            if dist < first:
                second = first
                first = dist
                best = j
            elif dist < second:
                second = dist
        if first / second < ratio:
            matches.append([i, best])
    return matches


def ransac(pts1, pts2, iters=500, eps=3):
    best = []
    for k in range(iters):
        pick = np.random.choice(len(pts1), 4, replace=False)
        H = computeH(pts1[pick], pts2[pick])

        inliers = []
        for i in range(len(pts1)):
            p = np.dot(H, [pts1[i][0], pts1[i][1], 1])
            x = p[0] / p[2]
            y = p[1] / p[2]
            if np.sqrt((x - pts2[i][0])**2 + (y - pts2[i][1])**2) < eps:
                inliers.append(i)

        if len(inliers) > len(best):
            best = inliers

    H = computeH(pts1[best], pts2[best])
    return H, best


sets = [
    ("IMG_8081", "IMG_8082", "Correspondence Tool copy.json", 0, True),
    ("IMG_8089", "IMG_8090", "Correspondence Tool Data copy.json", -8, False),
    ("IMG_0846", "IMG_0847", "Correspondence Tool Data (1).json", 0, False),
]
 
for n, (name1, name2, json_file, start, rotate) in enumerate(sets):
    im1 = plt.imread(name1 + ".jpeg")[::4, ::4]
    im2 = plt.imread(name2 + ".jpeg")[::4, ::4]
    if rotate:
        im1 = np.rot90(im1, k=-1)
        im2 = np.rot90(im2, k=-1)
 
    gray1 = rgb2gray(im1)
    gray2 = rgb2gray(im2)
    h1, coords1 = get_harris_corners(gray1)
    h2, coords2 = get_harris_corners(gray2)
    pts1 = anms(h1, coords1)
    pts2 = anms(h2, coords2)
 
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(im1)
    plt.plot(coords1[1], coords1[0], "r.", markersize=2)
    plt.title("Harris corners (" + str(coords1.shape[1]) + ")")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(im1)
    plt.plot(pts1[:, 0], pts1[:, 1], "r.", markersize=4)
    plt.title("After ANMS (" + str(len(pts1)) + ")")
    plt.axis("off")
    plt.savefig("corners" + str(n + 1) + ".png", bbox_inches="tight")
    plt.show()
 
    desc1 = descriptors(gray1, pts1)
    desc2 = descriptors(gray2, pts2)
 
    plt.figure(figsize=(12, 3))
    for i in range(6):
        plt.subplot(1, 6, i + 1)
        plt.imshow(desc1[i].reshape(8, 8), cmap="gray")
        plt.axis("off")
    plt.suptitle("Some 8x8 descriptors from " + name1)
    plt.savefig("descriptors" + str(n + 1) + ".png", bbox_inches="tight")
    plt.show()
 
    matches = match(desc1, desc2)
    m1 = []
    m2 = []
    for i, j in matches:
        m1.append(pts1[i])
        m2.append(pts2[j])
    m1 = np.array(m1, dtype=float)
    m2 = np.array(m2, dtype=float)
 
    w = im1.shape[1]
    plt.figure(figsize=(12, 6))
    plt.imshow(np.hstack([im1, im2]))
    for i in range(len(m1)):
        plt.plot([m1[i][0], m2[i][0] + w], [m1[i][1], m2[i][1]], linewidth=0.8)
        plt.plot(m1[i][0], m1[i][1], "r.", markersize=4)
        plt.plot(m2[i][0] + w, m2[i][1], "r.", markersize=4)
    plt.title(str(len(matches)) + " matches")
    plt.axis("off")
    plt.savefig("matches" + str(n + 1) + ".png", bbox_inches="tight")
    plt.show()
 
    H_auto, inliers = ransac(m1, m2)
 
    plt.figure(figsize=(12, 6))
    plt.imshow(np.hstack([im1, im2]))
    for i in inliers:
        plt.plot([m1[i][0], m2[i][0] + w], [m1[i][1], m2[i][1]], linewidth=0.8)
        plt.plot(m1[i][0], m1[i][1], "r.", markersize=4)
        plt.plot(m2[i][0] + w, m2[i][1], "r.", markersize=4)
    plt.title(str(len(inliers)) + " RANSAC inliers")
    plt.axis("off")
    plt.savefig("inliers" + str(n + 1) + ".png", bbox_inches="tight")
    plt.show()
 
    with open(json_file) as f:
        js = json.load(f)
    man1 = np.array(js["im1Points"][start:]) / 4
    man2 = np.array(js["im2Points"][start:]) / 4
    H_manual = computeH(man1, man2)
 
    manual = makeMosaic(im1, im2, H_manual)
    auto = makeMosaic(im1, im2, H_auto)
 
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(manual)
    plt.title("Manual")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(auto)
    plt.title("Automatic")
    plt.axis("off")
    plt.savefig("mosaic" + str(n + 1) + ".png", bbox_inches="tight")
    plt.show()
 