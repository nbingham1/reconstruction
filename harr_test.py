import harr
import Image
import numpy as np

img = to_float(load('cat-original.png'))
coeffs = haar_2d(img)
strong_coeffs = keep_ratio(coeffs, .05)
lossy = ihaar_2d(strong_coeffs)

save('cat-coeff.png', bipolar(coeffs))
save('cat-coeff-5pct.png', bipolar(strong_coeffs))
save('cat-output.png', from_float(lossy))

