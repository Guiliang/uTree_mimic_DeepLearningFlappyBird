import cv2
import sys

from scipy import ndimage
# from utree_training import Problem_flappyBird, Agent_CUT as Agent
from utree_training import Problem_flappyBird, Agent_oracle as Agent
from utree_training.test import opts
# from utree_training import Problem_flappyBird, Agent_regression as Agent
import game.wrapped_flappy_bird as game
import numpy as np
import pickle
import csv
import random
import traceback
import weka.core.jvm as jvm
from weka.core.converters import Loader
from weka.classifiers import Classifier
from weka.classifiers import Evaluation
import weka.core.serialization as serialization

ACTION_LIST = [0, 1]
ACTIONS = 2  # number of valid actions
FRAME_PER_ACTION = 1
TREE_PATH = "save_utree_7_g/"
FILEPATH = "save_q/CUT_performance.csv"
Q_PATH = './save_q/'
# TREE_PATH = "save_regression_utree/"
name_list = [i for i in range(14403)]
name_list.append("action")
name_list.append("value")
name_list = np.asarray(name_list)

def playGame(agent=None, clf=None, classifier=None, loader=None):
  # open up a game state to communicate with emulator
  game_state = game.GameState()
  
  # get the first state by doing nothing and preprocess the image to 80x80x4
  do_nothing = np.zeros(ACTIONS)
  do_nothing[0] = 1
  x_t, r_0, terminal = game_state.frame_step(do_nothing)
  x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
  ret, x_t = cv2.threshold(x_t, 1, 255, cv2.THRESH_BINARY)
  s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)
  a_list = np.stack(np.zeros(4))
  
  timeline = [["timestep", "action", "reward"]]
  t = 0
  while "flappy bird" != "angry bird":
    # choose an action epsilon greedily
    temp = np.zeros((4, 45, 80))
    for i in range(4):
      temp[i] = ndimage.rotate(s_t[:, :, i], 270)[:][10:55]
    # currentObs = np.insert(np.reshape(list(s_t[:, :, 0]), 6400), 6400, a_list[:3])
    # currentObs = np.reshape(temp, 14400)
    currentObs = np.insert(np.reshape(temp, 14400), 14400, a_list[:3])
    a_t = agent.utree.getBestAction(currentObs)
    # a_t = getBestActionM5(currentObs, classifier, loader)
    # a_t = getBestActionCART(currentObs, clf)
    a_list = np.append(a_t[1], a_list[:3])
    # a_t = agent.getQ(currentObs)
    # if (a_t[0] > a_t[1]):
    #   a_t = [1, 0]
    # else:
    #   a_t = [0, 1]
    
    # run the selected action and observe next state and reward
    x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
    x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
    x_t1 = np.reshape(x_t1, (80, 80, 1))
    s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)
    
    # update the old values
    s_t = s_t1
    t += 1

    # temp = np.zeros((4, 45, 80))
    # for i in range(4):
    #   temp[i] = ndimage.rotate(s_t1[:, :, i], 270)[:][10:55]
    # if random.randint(0, 100) % 10 == 0:
    #   currentObs = np.insert(np.reshape(temp, 14400), 14400, a_list[:3])
    #   action = (a_t[1] == 1)
    #   point_fig = agent.getFig(currentObs, action)
    #   point_fig = np.reshape(point_fig, (4, 45, 80))
    #   full_fig = np.zeros((4, 80, 80))
    #   for i in range(4):
    #     full_fig[i][:10] = ndimage.rotate(s_t1[:, :, i], 270)[:][:10]
    #     full_fig[i][10:55] = point_fig[i]
    #     full_fig[i][55:] = ndimage.rotate(s_t1[:, :, i], 270)[:][55:]
    #   pickle.dump(full_fig, open(Q_PATH + "fig.p", 'wb'))
    #   exit(0)
      
    print("TIMESTEP", t, "/ Action", a_t, "/ REWARD", r_t)
    timeline.append([t, a_t, r_t])
    if len(timeline)==10001:
      with open(FILEPATH, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for item in timeline:
          writer.writerow(item)
      exit(0)
      
    
    # print info
    state = "test"

def getBestActionCART(currentObs, clf):
  temp_data_list = []
  for action in ACTION_LIST:
    temp_data = currentObs.tolist() + [action]
    temp_data_list.append(temp_data)
  output = clf.predict(np.asarray(temp_data_list))
  if output[0] > output[1]:
    return [1, 0]
  else:
    return [0, 1]

def getBestActionM5(currentObs, classifier, loader):
  temp_data_dir = './temp_running_data/fb_observation.csv'
  temp_data_list = [name_list]
  for action in ACTION_LIST:
    temp_data = currentObs.tolist() + [action] + [100]
    temp_data_list.append(temp_data)
  save_csv_temp_obervations(temp_data_list, temp_data_dir)
  return get_action(temp_data_dir, classifier, loader)

def get_action(temp_data_dir, classifier, loader):
  data = loader.load_file(temp_data_dir)
  data.class_is_last()

  evaluation = Evaluation(data)
  eval = evaluation.test_model(classifier, data)
  Q_list = eval.tolist()
  act = ACTION_LIST[Q_list.index(max(Q_list))]
  if act == 0:
    return [1, 0]
  else:
    return [0, 1]

def save_csv_temp_obervations(datas, csv_name):
  with open(csv_name, 'w') as output:
    writer = csv.writer(output, lineterminator='\n')
    for data in datas:
      writer.writerow(data)

if __name__ == "__main__":
  ice_hockey_problem = Problem_flappyBird.flappyBird(games_directory=opts.GAME_DIRECTORY)
  CUTreeAgent = Agent.CUTreeAgent(problem=ice_hockey_problem, max_hist=opts.MAX_NODE_HIST,
                                  check_fringe_freq=opts.CHECK_FRINGE_FREQ, is_episodic=0)
  # CUTreeAgent.utree.fromcsvFile(TREE_PATH + "Game_File_" + sys.argv[1] + ".csv")
  CUTreeAgent.episode(int(sys.argv[1]))
  # CUTreeAgent.utree.fromcsvFile(TREE_PATH + "Game_File_" + sys.argv[1])
  # CUTreeAgent.utree = pickle.load(open(TREE_PATH + "Game_File_" + sys.argv[1] + '.p', mode='rb'))
  # try:
  #   jvm.start()
  # clf = pickle.load(open("../comparison-tree-regression/save_classifier/flappy-fb-record-10fold9.model", 'rb'))
    # classifier = Classifier(jobject=serialization.read("../comparison-tree-regression/save_classifier/"
    #                                                    "flappybird-m5p-weka-record-10fold9.model"))
    # loader = Loader(classname="weka.core.converters.CSVLoader")
  playGame(agent=CUTreeAgent)
  # except Exception as e:
  #   print(traceback.format_exec())
  # finally:
  #   jvm.stop()
