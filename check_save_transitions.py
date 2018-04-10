import pickle
import numpy as np
import random
# import cv2
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt
import plotly.tools as tls
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

ZERO_COLOR = [0.67, 1, 1]
FULL_COLOR = [0.7, 0.7, 0.7]
PIXEL_COLOR = [1, 0, 0]
Q_PATH = './save_q/fig-star/8/'
# temp = pickle.load(open('./save_all_transitions/transition_num001/game.p', mode='rb'))
temp = pickle.load(open(Q_PATH + 'fig.p', mode='rb'))
observation = pickle.load(open('./save_q/influence.p', mode='rb'))

temps = [(0 if i < 0 else i) for i in observation]
observation = []
for i in range(4):
  observation.append(np.reshape(temps[i * 3600:(i + 1) * 3600], (45, 80)))
#

# print(observation[0])
# first_image_plot = np.zeros((81, 81))
first_image_plot = np.zeros((80, 80, 3))
medium = np.zeros((45, 80))
for k in range(4):
  medium += 2.5 * np.flip(observation[k], axis=1)
print(sorted(np.reshape(medium, 3600)))
for k in range(4):
  # first_image = np.zeros((81, 45))
  first_image = np.flip(temp[k], axis=1)
  # medium += 2.5*np.flip(observation[k], axis=1)
  # first_image[:80] = np.transpose(np.flip(observation[k], axis=1))
  # first_image_plot[:][10:55] += np.transpose(first_image)
  x = []
  y = []
  for i in range(80):
    for j in range(80):
      if first_image[i][j] == 0:
        first_image_plot[i][j] = ZERO_COLOR
      elif first_image[i][j] == 255:
        first_image_plot[i][j] = FULL_COLOR
      elif j <= 7:
        first_image_plot[i][j] = ZERO_COLOR if first_image[i][j] == 40 else FULL_COLOR
      else:
        x.append(j)
        y.append(i)
        first_image_plot[i][j] = PIXEL_COLOR
# medium = np.reshape(medium, 3600)
# medium_zero = sum([(True if i != 0 else False) for i in medium])
# medium_value = sorted(medium)[3600-int(medium_zero/2)]
# mean_value = sum(medium)/medium_zero

# print mean
# print(medium_zero)
# print(sorted(medium))
# print(medium_value)
# print(mean_value)
# print(sum(medium)/3600)

# # add random
# for i in range(50, 75):
#   for j in range(80):
#     if random.randint(0, 1000) % 100 == 0:
#       first_image_plot[i][j] += random.random() / 100
# # enlarge
# for i in range(80):
#   for j in range(80):
#     first_image_plot[i][j] *= 2.5
# remove below mean
# for i in range(80):
#   for j in range(80):
#     if (first_image_plot[i][j] < mean_value):
#       first_image_plot[i][j] = 0

# plot
# sns.set(font_scale=2.5)
# fig = sns.heatmap(first_image_plot, xticklabels=20, yticklabels=20,
#                   vmin = 0, vmax = 0.05, cmap='Blues')
# fig.axis([0, 81, 81, 0])
# major_ticks = np.arange(0, 81, 20)
# print(major_ticks)
# fig.set_xticks(major_ticks)
# fig.set_yticks(major_ticks)
# mpl_fig = plt.figure()
# ax = mpl_fig.add_subplot(111)
# x=[0, 1, 2, 3, 4, 5, 6, 7, 8]
# y=[0, 4, 5, 1, 8, 5, 3, 2, 9]
#
# line, = ax.plot(x, y, lw=2)
#
# plotly_fig = tls.mpl_to_plotly( mpl_fig )
#
# fig['layout']['xaxis1'].update({'ticktext': ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight'],
#                                        'tickvals': [0, 10, 20, 30, 40, 50, 60, 70, 80],
#                                        'tickfont': {'size': 14, 'family':'Courier New, monospace'},
#                                        'tickangle': 60
#                                       })
# ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
# ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
# fig.xaxis.set_ticks(np.arange(0, 81, 20))
# fig.yaxis.set_ticks(np.arange(0, 81, 20))
  fig = plt.imshow(first_image_plot)
  plt.plot(x, y, 'r*', markersize = 10)
  fig.axes.get_xaxis().set_visible(False)
  fig.axes.get_yaxis().set_visible(False)
  plt.savefig(Q_PATH + 'fig_star_clean' + str(k))
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
