import ssearch
import numpy
import caffe
import haar
from PIL import Image

def load(path):
    return Image.open(path).convert('RGB')

def display(img):
    img.show()

# select
#
# img
#    a PIL.Image in 'RGB' values
#    load using: Image.open(filename).convert('RGB')
#
# return
#    a dictionary of {rect => image}
#    where rect is (x,y,w,h)
#    and image is a PIL.Image in 'RGB' values
def select(img):
    img_lbl, regions = ssearch.selective_search(img, scale=300.0, sigma=0.95, min_size=20)

    results = {}
    for r in regions:
        # excluding same rectangle (with different segments)
        if r['rect'] in regions:
            continue
        # excluding regions smaller than 2000 pixels
        if r['size'] < 1000:
            continue
        # distorted rects
        x, y, w, h = r['rect']
        if w / h > 1.2 or h / w > 1.2:
            continue
        results[r['rect']] = img.copy().crop((x, y, x+w, y+h))

    return results

# classify
#
# img
#    a PIL.Image in 'RGB' values
#    load using: Image.open(filename).convert('RGB')
#
# return
#    an array of [ label, confidence ]
#    where label is a string label describing the object
#    and confidence is a float describing the confidence of that description
def classify(img):    
    caffe_root = '../caffe-master/'
    net = caffe.Net(caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt',
                    caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel',
                    caffe.TEST)

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    
    # mean pixel
    transformer.set_mean('data', numpy.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy').mean(1).mean(1))

    # the reference model operates on images in [0,255] range instead of [0,1]
    transformer.set_raw_scale('data', 255)
   
    # the reference model has channels in BGR order instead of RGB
    transformer.set_channel_swap('data', (2,1,0))

    # set net to batch size of 50
    net.blobs['data'].reshape(50, 3, 227, 227)
    net.blobs['data'].data[...] = transformer.preprocess('data', numpy.float32(numpy.asarray(img))/255.0)

    out = net.forward()

    # load the labels
    labels = numpy.loadtxt(caffe_root + 'data/ilsvrc12/synset_words.txt', str, delimiter='\t')

    confidence = net.blobs['prob'].data[0].flatten()
    indices = confidence.argsort()

    return numpy.dstack((labels[indices], confidence[indices]))[0]

# compress
#
# img
#    a PIL.Image in 'RGB' values
#    load using: Image.open(filename).convert('RGB')
#
# ratio
#    a float representing the amount to compress the image
#    the lower the value, the more compressed the resulting image
#
# return
#    a PIL.Image in 'RGB' values
def compress(img, ratio):
    component = img.split()

    rcoeffs = haar.haar_2d(haar.to_float(component[0]))
    gcoeffs = haar.haar_2d(haar.to_float(component[1]))
    bcoeffs = haar.haar_2d(haar.to_float(component[2]))

    rstrong_coeffs = haar.keep_ratio(rcoeffs, ratio)
    gstrong_coeffs = haar.keep_ratio(gcoeffs, ratio)
    bstrong_coeffs = haar.keep_ratio(bcoeffs, ratio)

    rlossy = haar.ihaar_2d(rstrong_coeffs)
    glossy = haar.ihaar_2d(gstrong_coeffs)
    blossy = haar.ihaar_2d(bstrong_coeffs)

    r = Image.fromarray(haar.from_float(rlossy))
    g = Image.fromarray(haar.from_float(glossy))
    b = Image.fromarray(haar.from_float(blossy))
 
    return Image.merge('RGB', (r, g, b))
    
# test for select
#img = load('data/cat.jpg')
#display(img)
#
#regions = select(img)
#
#for region in regions.keys():
#    display(regions[region])

# test for classify
#img = load('data/cat.jpg')
#display(img)
#
#print classify(img)[-1:-2:-1]

# test for select and classify
#img = load('data/cat.jpg')
#display(img)
#
#regions = select(img)
#
#display(regions[regions.keys()[0]])
#print classify(regions[regions.keys()[0]])[-1:-2:-1]

# test for compress
img = load('data/five-cute-kittens.jpg')
display(img)

display(compress(img, 0.025))
