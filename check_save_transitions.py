import pickle
import numpy as np
import cv2
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt


# temp = pickle.load(open('./save_all_transitions/transition_num001/game.p', mode='rb'))
temp = pickle.load(open('./save_q/fig.p', mode='rb'))

observation = temp
# print(observation[0])
first_image = np.reshape(observation[0:3600], (45, 80))
second_image = np.reshape(observation[3600:7200], (45,80))
third_image = np.reshape(observation[7200:10800], (45,80))
fourth_image = np.reshape(observation[10800:], (45, 80))
# first_image[5] = [220 for i in range(80)]
# first_image[6] = [40 for i in range(80)]
# first_image = ndimage.rotate(first_image, 270)
plt.imshow(first_image)
plt.savefig('save_q/fig1')
plt.show()
plt.imshow(second_image)
plt.savefig('save_q/fig2')
plt.show()
plt.imshow(third_image)
plt.savefig('save_q/fig3')
plt.show()
plt.imshow(fourth_image)
plt.savefig('save_q/fig4')
plt.show()

print('testing')
