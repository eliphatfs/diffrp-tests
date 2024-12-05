import os
import sys
import numpy
import unittest
from PIL import Image
from diffrp import to_pil


def is_verbose():
    return ('-v' in sys.argv) or ('--verbose' in sys.argv)


def compute_psnr(img1, img2):
    img1_array = numpy.array(img1, dtype=numpy.float32)
    img2_array = numpy.array(img2, dtype=numpy.float32)
    
    mse = numpy.mean((img1_array - img2_array) ** 2)
    
    if mse == 0:
        return 120
    
    max_pixel = 255.0
    psnr = 20 * numpy.log10(max_pixel / numpy.sqrt(mse))
    
    return psnr


class ScreenshotTestingCase(unittest.TestCase):
    
    def setUp(self):
        self.verbose = is_verbose()
        if self.verbose:
            print()
    
    def compare(self, name, result, psnr_threshold=40):
        orig_name = name
        name = name + '.png'
        if not os.path.exists(os.path.join('reference', name)):
            os.makedirs(os.path.dirname(os.path.join('reference', name)), exist_ok=True)
            to_pil(result).save(os.path.join('reference', name), optimize=True)
        reference = Image.open(os.path.join('reference', name))
        compared = to_pil(result)
        psnr = compute_psnr(reference, compared)
        if self.verbose:
            print("STest %s PSNR %.2f dB" % (orig_name, psnr))
        if psnr < psnr_threshold:
            if not os.path.exists(os.path.join('diff', name)):
                os.makedirs(os.path.dirname(os.path.join('diff', name)), exist_ok=True)
                compared.save(os.path.join('diff', name))
            self.fail("%s is different to reference, PSNR %.2f dB" % (orig_name, psnr))
