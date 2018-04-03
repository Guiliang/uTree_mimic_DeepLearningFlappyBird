import pickle
import numpy as np
import cv2
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import seaborn as sns

ZERO_COLOR = [0.67, 1, 1]
FULL_COLOR = [0.7, 0.7, 0.7]
PIXEL_COLOR = [1, 0, 0]
Q_PATH = './save_q/'
# temp = pickle.load(open('./save_all_transitions/transition_num001/game.p', mode='rb'))
# temp = pickle.load(open(Q_PATH + 'fig.p', mode='rb'))
temp = pickle.load(open(Q_PATH + 'influence.p', mode='rb'))

temp = [(0 if i < 0 else i) for i in temp]
observation = []
for i in range(4):
  observation.append(np.reshape(temp[i * 3600:(i + 1) * 3600], (45, 80)))

# print(observation[0])
for k in range(4):
  first_image = np.flip(observation[k], axis=1)
  first_image_plot = np.zeros((80, 80))
  first_image_plot[:][10:55] = first_image
  # x = []
  # y = []
  # for i in range(80):
  #   for j in range(80):
  #     if first_image[i][j] == 0:
  #       first_image_plot[i][j] = ZERO_COLOR
  #     elif first_image[i][j] == 255:
  #       first_image_plot[i][j] = FULL_COLOR
  #     else:
  #       x.append(j)
  #       y.append(i)
  #       first_image_plot[i][j] = PIXEL_COLOR
  ax = sns.heatmap(first_image_plot, cmap='Blues') #, cmap = "YlGnBu")
  # fig = plt.imshow(first_image_plot)
  # plt.plot(x, y, 'r*', markersize = 10)
  # fig.axes.get_xaxis().set_visible(False)
  # fig.axes.get_yaxis().set_visible(False)
  plt.savefig(Q_PATH + 'another_blue_inf' + str(k))
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
