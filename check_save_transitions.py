import pickle
import numpy as np
import cv2
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt


temp = pickle.load(open('./save_all_transitions/transition_num000/game.p', mode='rb'))

observation = temp[10][0]
print(observation[0])
first_image = observation
first_image = ndimage.rotate(first_image, 270)[:][10:55]
plt.imshow(first_image)
plt.show()

print('testing')
