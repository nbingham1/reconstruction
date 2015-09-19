# -*- coding: utf-8 -*-
import skimage.data
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import selectivesearch

from cStringIO import StringIO
import numpy as np
import scipy.ndimage as nd
import PIL.Image
from IPython.display import clear_output, Image, display
from google.protobuf import text_format

from wand.image import Image


def showarray(a, fmt='jpeg'):
    a = np.uint8(np.clip(a, 0, 255))
    f = StringIO()
    PIL.Image.fromarray(a).show() #.save(f, fmt)


def main():

    # loading lena image
    #img = skimage.data.lena()
    imagefilename = 'TextImage.jpg'
    img = np.float32(PIL.Image.open(imagefilename))/256.0
    #showarray(img)


    # perform selective search
    img_lbl, regions = selectivesearch.selective_search(
        img, scale=500, sigma=0.1, min_size=2)

    candidates = set()
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in candidates:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 2000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        candidates.add(r['rect'])


    # draw rectangles on the original image
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for x, y, w, h in candidates:
        print x, y, w, h
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)


    # create separate cropped images
    i = 0
    for x, y, w, h in candidates:
	with Image(filename=imagefilename) as img:
	     print x, y, w, h
	     i = i + 1
	     oldimg = img
	     print img.size
	     img.crop(x,y,width=w,height=h)
	     img.format = 'jpeg'
	     img.save(filename='frame_'+str(i)+'.jpg')
	     img = oldimg

    plt.show()

    

if __name__ == "__main__":
    main()







