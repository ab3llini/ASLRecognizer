from PIL import Image, ImageFilter, ImageEnhance
import numpy as np
import random as rnd




class DataAugmentation:




 def __init__(self, data):
     self.images = np.array(data)

 def __filter_image(self, im, n):
     size = 3

     if 1 <= n <= 10:
         radius = rnd.randrange(1, 20)
         augmented_image = im.filter(ImageFilter.GaussianBlur(radius))  # max radius = 20
     elif 11 <= n <= 20:
         radius = rnd.randrange(1, 40)
         augmented_image = im.filter(ImageFilter.BoxBlur(radius))  # max radius = 40
     elif 21 <= n <= 30:
         radius = rnd.randrange(1, 1000)
         percent = rnd.randrange(1, 1000)
         threshold = rnd.randrange(1, 1000)
         augmented_image = im.filter(ImageFilter.UnsharpMask(radius, percent, threshold))
     elif 31 <= n <= 40:
         a = rnd.randrange(1, 1000)
         b = rnd.randrange(1, 1000)
         c = rnd.randrange(1, 1000)
         d = rnd.randrange(1, 1000)
         e = rnd.randrange(1, 1000)
         f = rnd.randrange(1, 1000)
         g = rnd.randrange(1, 1000)
         h = rnd.randrange(1, 1000)
         i = rnd.randrange(1, 1000)
         kernel = [a, b, c, d, e, f, g, h, i]
         scale = rnd.randrange(1000, 10000)
         offset = rnd.randrange(1, 100)
         augmented_image = im.filter(ImageFilter.Kernel((3, 3), kernel, scale, offset)).show()
     elif 41 <= n <= 50:
         augmented_image = im.filter(ImageFilter.MinFilter(size))
     elif 51 <= n <= 60:
         augmented_image = im.filter(ImageFilter.MaxFilter(size))
     elif 61 <= n <= 70:
         augmented_image = im.filter(ImageFilter.MedianFilter(size))
     elif 71 <= n <= 80:
         augmented_image = im.filter(ImageFilter.ModeFilter(size))
     elif 81 <= n <= 90:
         augmented_image = im.filter(ImageFilter.CONTOUR)
     elif 91 <= n <= 100:
         augmented_image = im.filter(ImageFilter.SMOOTH)
     elif 101 <= n <= 110:
         augmented_image = im.filter(ImageFilter.EDGE_ENHANCE)
     elif 111 <= n <= 120:
         augmented_image = im.filter(ImageFilter.BLUR)
     elif 121 <= n <= 130:
         augmented_image = im.filter(ImageFilter.DETAIL)
     elif 131 <= n <= 140:
         augmented_image = im.filter(ImageFilter.EDGE_ENHANCE_MORE)
     elif 141 <= n <= 150:
         augmented_image = im.filter(ImageFilter.EMBOSS)
     elif 151 <= n <= 160:
         augmented_image = im.filter(ImageFilter.FIND_EDGES)
     elif 161 <= n <= 170:
         augmented_image = im.filter(ImageFilter.SMOOTH_MORE)
     elif 171 <= n <= 180:
         augmented_image = im.filter(ImageFilter.SHARPEN)

     return augmented_image

 def __enhance_image(self, im, n):
     factor = rnd.randrange(1, 10)

     if 181 <= n <= 190:
         factor = factor * 2
         enhancer = ImageEnhance.Sharpness(im)
     if 191 <= n <= 200:
         enhancer = ImageEnhance.Color(im)
     if 201 <= n <= 210:
         enhancer = ImageEnhance.Contrast(im)
     if 221 <= n <= 220:
         enhancer = ImageEnhance.Brightness(im)
     enhanced_image = enhancer.enhance(factor / 10)

     return enhanced_image

 def __zoom_image(self, im, n):
     im_crop = im.crop((0, 0, 500, 500))
     im_crop = im_crop.resize(im.size)

     return im_crop

 def augment_data(self, images_count):
     rnd.seed(1811)
     result_images = []

     for im_pixels in range(len(self.images)):
         image = Image.fromarray(im_pixels.astype('uint8'), 'RGB')
         for j in range(images_count):
             n = rnd.randrange(1, 300)
             if 1 <= n <= 180:
                 result_images.append(self.filter_image(image, n))
             elif 181 <= n <= 220:
                 result_images.append(self.enhance_image(image, n))
             elif 221 <= n <= 300:
                 result_images.append(self.zoom_image(image, n))

     return result_images

