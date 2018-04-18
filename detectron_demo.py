from pdf2image import convert_from_path, convert_from_bytes 
%matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2

PATH_TO_TEST_PDF_DIR = '/tmp/detectron-visualizations/'
PATH_TO_OP_IMAGES_DIR = '/home/cdsw/op_images'
from os import listdir
from os.path import isfile, join
onlyfiles = listdir(PATH_TO_TEST_PDF_DIR)
for ip in onlyfiles:
  image_path=os.path.join(PATH_TO_TEST_PDF_DIR,ip)
  convert_from_path(image_path,output_folder=PATH_TO_OP_IMAGES_DIR)
  
  
import time  
from os import listdir
from os.path import isfile, join
onlyfiles = listdir(PATH_TO_OP_IMAGES_DIR)
for imagename in onlyfiles:
  print os.path.join(PATH_TO_OP_IMAGES_DIR,imagename)
  img = cv2.imread(os.path.join(PATH_TO_OP_IMAGES_DIR,imagename))
  #plt.pause(.5)
  plt.draw()
  plt.show()

import matplotlib.pyplot as plt 
import numpy as np 

x = np.linspace(0,10)
y = np.sin(x)

for i in range(8):
    plt.plot(x,y)
    plt.figure(i+1)
img=cv2.imread('/home/cdsw/op_images/616f87ea-b449-4e00-aac5-b9bbe706baf6-1.ppm')
plt.imshow(img)

img1=cv2.imread('/home/cdsw/op_images/59d5fa1f-4e4b-4a58-87b0-12807881a95e-1.ppm')
plt.imshow(img1)

img2=cv2.imread('/home/cdsw/op_images/f3256aad-806f-4e55-8f9b-fcbe8eb99119-1.ppm')
plt.imshow(img2)

img3=cv2.imread('/home/cdsw/op_images/f3256aad-806f-4e55-8f9b-fcbe8eb99119-1.ppm')
plt.imshow(img3)

img4=cv2.imread('/home/cdsw/op_images/c5d7070e-593c-43b5-80ab-9c65685bfb11-1.ppm')
plt.imshow(img4)

img5=cv2.imread('/home/cdsw/op_images/c8cb9a87-51d6-4e04-b342-9d375ce8ec10-1.ppm')
plt.imshow(img5)

img6=cv2.imread('/home/cdsw/op_images/85eb378a-679f-4b95-84f3-54aae2e03287-1.ppm')
plt.show(img6)

img7=cv2.imread('/home/cdsw/op_images/ce08f1f0-845d-4a13-b6d4-bb109b83b401-1.ppm')
plt.imshow(img7)

img8=cv2.imread('/home/cdsw/op_images/1653237c-e5d7-4ab7-b0e7-636a0d0c2c97-1.ppm')
plt.imshow(img8)
