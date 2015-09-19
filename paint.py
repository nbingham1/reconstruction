
import numpy
import random
from PIL import Image
from PIL import ImageFilter

def load(path):
    return Image.open(path).convert('RGB')

def display(img):
    img.show()

def random_three_vector():
    """
    Generates a random 3D unit vector (direction) with a uniform spherical distribution
    Algo from http://stackoverflow.com/questions/5408276/python-uniform-spherical-distribution
    :return:
    """
    phi = numpy.random.uniform(0.0,numpy.pi*2.0)
    costheta = numpy.random.uniform(-1.0,1.0)

    theta = numpy.arccos( costheta )
    x = numpy.sin( theta) * numpy.cos( phi )
    y = numpy.sin( theta) * numpy.sin( phi )
    z = numpy.cos( theta )
    return numpy.asarray((x,y,z))

def brush_stroke(img, width, strength):
    (w,h,d) = img.shape
    # Generate random position
    x = random.randint(0,w)
    y = random.randint(0,h)
    # Generate random sphere of color
    color = random_three_vector()*strength
    # Apply Gaussian brush at x,y
    result = img.copy()
    for i in xrange(width):
        i0 = i-width/2
        ywidth = int(2.0*numpy.sqrt(width*width/4 - i0*i0))
        for j in xrange(ywidth):
            j0 = j-ywidth/2
            if i0+x >= 0 and i0+x < w and j0+y >= 0 and j0+y < h:
                result[i0+x][j0+y] = numpy.clip(result[i0+x][j0+y] + color*(numpy.cos(numpy.pi*numpy.sqrt(i0*i0+j0*j0)*2.0/width) + 1.0)/2.0, -128.0, 127.0)
    return result

