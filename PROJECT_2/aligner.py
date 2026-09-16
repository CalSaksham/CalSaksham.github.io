# CS180 (CS280A): Project 1 starter Python code

# these are just some suggested libraries
# instead of scikit-image you could use matplotlib and opencv to read, write, and display images

import numpy as np
import skimage as sk
import skimage.io as skio

images = [
    "Master PNP Prok 01700 01754a.tif"
]
for image in images:
    # name of the input file
    imname = image

    # read in the image
    im = skio.imread(imname)

    # convert to double (might want to do this later on to save memory)    
    im = sk.img_as_float(im)
        
    # compute the height of each part (just 1/3 of total)
    height = np.floor(im.shape[0] / 3.0).astype(int)

    # separate color channels
    b = im[:height]
    g = im[height: 2*height]
    r = im[2*height: 3*height]

    # align the images
    # functions that might be useful for aligning the images include:
    # np.roll, np.sum, sk.transform.rescale (for multiscale)

    print(len(b))

    # create a color image

    def border_remover(channel):

        
        row_normalized_var_threshhold = np.mean(np.var(channel, axis = 1))/10
        col_normalized_var_threshhold = np.mean(np.var(channel, axis = 0))/10
        new_channel = channel.copy()
        for row in range(len(new_channel)):
            if np.var(new_channel[row]) > row_normalized_var_threshhold:
                break
        for neg_row in range(len(new_channel)-1, -1,-1):
            if np.var(new_channel[neg_row]) > row_normalized_var_threshhold:
                break
        new_channel = new_channel.T
        for col in range(len(new_channel)):
            if np.var(new_channel[col]) > col_normalized_var_threshhold:
                break
        for neg_col in range(len(new_channel)-1, -1,-1):
            if np.var(new_channel[neg_col]) > col_normalized_var_threshhold:
                break    
        new_channel = new_channel.T
        return (row, neg_row+1, col, neg_col+1)

    def pyramid(channel_1, channel_2, loss, size = 5):
        if min(channel_1.shape) < 300:
            return helper(channel_1, channel_2, 0, 0, loss, 15)
        disp = pyramid(sk.transform.rescale(channel_1, 0.5, anti_aliasing = True), sk.transform.rescale(channel_2, 0.5, anti_aliasing = True), loss, min(size*2, 15))
        return helper(channel_1, channel_2, disp[0]*2, disp[1]*2, loss, size)


    def helper(channel_1, channel_2,center_x,center_y, loss, size):
        minim = -float('inf')
        out = np.array([])
        disp = (0,0)
        borderless_channel_2 = channel_2
        borderless_channel_1 = channel_1

        

        height, width = channel_1.shape
        r0, r1 = height// 5, height - height// 5
        c0, c1 = width // 5, width - width // 5

        borderless_channel_2 = borderless_channel_2[r0:r1, c0:c1]
        channel_2_normalized = (borderless_channel_2 - np.mean(borderless_channel_2))
        channel_2_normalized /= np.linalg.norm(channel_2_normalized)

        for i in range(center_x-size, center_x+size+1):
            for j in range(center_y-size, center_y+size+1):
                rolled_image = borderless_channel_1.copy()
                rolled_image = np.roll(rolled_image, (i, j), axis=(0, 1))[r0:r1, c0:c1]
                curr = 0
                if loss:
                    channel_1_normalized = (rolled_image - np.mean(rolled_image))
                    channel_1_normalized /= np.linalg.norm(channel_1_normalized)
                    curr = np.sum(channel_1_normalized*channel_2_normalized)
                else:
                    for norm_row in range(len(rolled_image)):
                        curr -= np.linalg.norm(rolled_image[norm_row]-borderless_channel_2[norm_row])

                if curr > minim:
                    minim = curr
                    disp = (i, j)
        return disp

    def align(channel_1, channel_2, loss, string):
        disp = pyramid(channel_1, channel_2, loss)
        print(f"{string}:", f"({disp[1]}, {disp[0]}")
        return np.roll(channel_1, disp, axis = (0,1))

    temp_b = border_remover(b)
    temp_g = border_remover(g)
    temp_r = border_remover(r)

    top = max(temp_b[0], temp_g[0], temp_r[0])
    bottom = min(temp_b[1], temp_g[1], temp_r[1])
    left = max(temp_b[2], temp_g[2], temp_r[2])
    right = min(temp_b[3], temp_g[3], temp_r[3])

    b = b[top:bottom, left:right]
    g = g[top:bottom, left:right]
    r = r[top:bottom, left:right]

    loss = 1
    # if imname == 'cathedral.jpg':
    #     loss = 0
    ag = align(g, b, loss, 'g->b')
    ar = align(r, b, loss, 'r->b')
    im_out = np.dstack([ar, ag, b])

    # save the image
    fname = f"./{imname}_out.jpg"

    im_out = sk.img_as_ubyte(im_out)
    skio.imsave(fname, im_out)

# display the image
# skio.imshow(im_out)
# skio.show()



        # temp_1 = border_remover(channel_1)
        # temp_2 = border_remover(channel_2)

        # final_temp = (max(temp_1[0], temp_2[0]), min(temp_1[1], temp_2[1]), max(temp_1[2], temp_2[2]), min(temp_1[3], temp_2[3]))
        # borderless_channel_1 = channel_1[final_temp[0]:final_temp[1], final_temp[2]:final_temp[3]]
        # borderless_channel_2 = channel_2[final_temp[0]:final_temp[1], final_temp[2]:final_temp[3]]