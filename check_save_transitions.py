import pickle
import numpy as np
import cv2
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

temp = pickle.load(open('./save_all_transitions/transition_num200.p', mode='r'))

observation = temp[0]

first_image = observation[:, :, 1]
first_image = np.flip(first_image, axis=1)
first_image = ndimage.rotate(first_image, 90)

plt.imshow(first_image)

plt.show()

print 'testing'
