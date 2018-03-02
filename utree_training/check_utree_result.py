import cv2
import sys

from utree_training import Problem_flappyBird, Agent_oracle as Agent
import game.wrapped_flappy_bird as game
import numpy as np
import pickle

from utree_training.test import opts

ACTIONS = 2  # number of valid actions
FRAME_PER_ACTION = 1
TREE_PATH = "save_utree/"


def playGame(agent):
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
  
  t = 0
  while "flappy bird" != "angry bird":
    # choose an action epsilon greedily
    currentObs = np.insert(np.reshape(list(s_t[:, :, 0]), 6400), 6400, a_list[:3])
    a_t = agent.utree.getBestAction(currentObs)
    a_list = np.append(a_t, a_list[:3])
    print("Action:", a_t)
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
    
    # print info
    state = "test"


if __name__ == "__main__":
  ice_hockey_problem = Problem_flappyBird.flappyBird(games_directory=opts.GAME_DIRECTORY)
  CUTreeAgent = Agent.CUTreeAgent(problem=ice_hockey_problem, max_hist=opts.MAX_NODE_HIST,
                                  check_fringe_freq=opts.CHECK_FRINGE_FREQ, is_episodic=0)
  # CUTreeAgent.utree.fromcsvFile(TREE_PATH + "Game_File_" + sys.argv[1] + ".csv")
  CUTreeAgent.utree = pickle.load(open(TREE_PATH + "Game_File_" + sys.argv[1] + '.p', mode='rb'))
  playGame(CUTreeAgent)
