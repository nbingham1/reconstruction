from harr import *
import Image
import numpy as np

img = load('sky1024px.jpg')
rcoeffs = haar_2d(to_float(img[0]))
gcoeffs = haar_2d(to_float(img[1]))
bcoeffs = haar_2d(to_float(img[2]))
rstrong_coeffs = keep_ratio(rcoeffs, .0025)
gstrong_coeffs = keep_ratio(gcoeffs, .0025)
bstrong_coeffs = keep_ratio(bcoeffs, .0025)
rlossy = ihaar_2d(rstrong_coeffs)
glossy = ihaar_2d(gstrong_coeffs)
blossy = ihaar_2d(bstrong_coeffs)

save('cat-output.png', from_float(rlossy), from_float(glossy), from_float(blossy))

