import pickle
import numpy as np
import cv2
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

ZERO_COLOR = [0.77, 1, 1]
FULL_COLOR = [1, 1, 0.77]
PIXEL_COLOR = [1, 0, 0]

# temp = pickle.load(open('./save_all_transitions/transition_num001/game.p', mode='rb'))
temp = pickle.load(open('./save_q/fig.p', mode='rb'))

observation = temp
# print(observation[0])
for k in range(4):
  first_image = np.flip(observation[k], axis=1)
  first_image_plot = np.zeros((80, 80, 3))
  for i in range(80):
    for j in range(80):
      if first_image[i][j] == 0:
        first_image_plot[i][j] = ZERO_COLOR
      elif first_image[i][j] == 255:
        first_image_plot[i][j] = FULL_COLOR
      else:
        first_image_plot[i][j] = PIXEL_COLOR
        
  fig = plt.imshow(first_image_plot)
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)
  plt.savefig('save_q/fig'+str(k))
  plt.show()
# plt.imshow(second_image)
# plt.savefig('save_q/fig2')
# plt.show()
# plt.imshow(third_image)
# plt.savefig('save_q/fig3')
# plt.show()
# plt.imshow(fourth_image)
# plt.savefig('save_q/fig4')
# plt.show()

print('testing')
