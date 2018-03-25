# Using oracle coaching (initialize by neural network)

import numpy as np
import scipy.io as sio
import os
import pickle
import inspect
import random
import csv
# from utree_training import C_UTree_regression as C_UTree
import C_UTree_regression as C_UTree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

TREE_PATH = "save_regression_utree/"
HOME_PATH = "/local-scratch/csv_oracle/"
Q_PATH = "save_q/"


def read_actions(game_directory, game_dir):
  actions = sio.loadmat(game_directory + game_dir + "/action_{0}.mat".format(game_dir))
  return actions["action"]


def read_states(game_directory, game_dir):
  temp = pickle.load(open(game_directory + game_dir + '/game.p', mode='rb'))
  return temp


def read_rewards(game_directory, game_dir):
  rewards = sio.loadmat(game_directory + game_dir + "/reward_{0}.mat".format(game_dir))
  return rewards["reward"]


def read_train_info(game_directory, game_dir):
  train_info = sio.loadmat(game_directory + game_dir + "/training_data_dict_all_name.mat")
  return train_info['training_data_dict_all_name']


def read_qValue(game_directory, game_dir):
  qValues = sio.loadmat(game_directory + game_dir + "/qvalue_{0}.mat".format(game_dir))
  return qValues["qValues"]


class CUTreeAgent:
  """
  Agent that implements McCallum's Sport-Analytic-U-Tree algorithm
  """
  
  def __init__(self, problem, max_hist, check_fringe_freq, is_episodic=0):
    
    self.utree = C_UTree.CUTree(gamma=problem.gamma,
                                n_actions=len(problem.actions),
                                dim_sizes=problem.dimSizes,
                                dim_names=problem.dimNames,
                                max_hist=max_hist,
                                is_episodic=is_episodic,
                                )
    self.cff = check_fringe_freq
    self.valiter = 1
    self.problem = problem
  
  def update(self, currentObs, nextObs, action, qValue, check_linear=0, check_fringe=0,
             home_identifier=None, beginflag=False):
    """
    update the tree
    :param currentObs: current observation
    :param nextObs: next observation
    :param action: action taken
    :param reward: reward get
    :param terminal: if end
    :param check_fringe: whether to check fringe
    :return:
    """
    t = self.utree.getTime()  # return the length of history
    i = C_UTree.Instance(t, currentObs, action, nextObs, None, home_identifier, qValue)
    
    self.utree.updateCurrentNode(i, beginflag)
    
    # if value_iter:
    #   self.utree.sweepLeaves()  # value iteration is performed here
    if check_linear:
      self.utree.testLinear()  # Linear regression model is performed here
      print("Finish LR in leaves")
    
    if check_fringe:
      self.utree.testFringe(check_linear)  # ks test is performed here
  
  def getQ(self, currentObs, action):
    """
    only insert instance to history
    :param currentObs:
    :param nextObs:
    :param action:
    :param reward:
    :return:
    """
    t = self.utree.getTime()
    i = C_UTree.Instance(t, currentObs, action, currentObs, None, None, None)
    q_h, q_a, q_oh, q_oa = self.utree.getInstanceQvalues(i, None)
    return [q_h, q_a], [q_oh, q_oa]
  
  def executePolicy(self, epsilon=1e-1):
    return None
    # """
    # epsilon-greedy policy basing on Q
    # :param epsilon: epsilon
    # :return: action to take
    # """
    # test = random.random()
    # if test < epsilon:
    #     return random.choice(range(len(self.problem.actions)))
    # return self.utree.getBestAction()
  
  def episode(self, checkpoint, timeout=int(1e5)):
    """
    start to build the tree within an episode
    :param timeout: no use here
    :return:
    """
    
    game_directory = self.problem.games_directory
    
    game_dir_all = os.listdir(game_directory)
    
    count = 0
    inscount = 0
    Qlist = []
    MAElist = [str(checkpoint), "MAE"]
    Corlist = [str(checkpoint), "Cor"]
    MSElist = [str(checkpoint), "MSE"]
    
    # checkpoint = 24
    if checkpoint > 0:
      # self.utree = pickle.load(open(TREE_PATH + "Game_File_" + str(checkpoint) + '.p', 'rb'))
      self.utree.fromcsvFile(TREE_PATH + "Game_File_" + str(checkpoint))
    
    for game_dir in game_dir_all:
      
      game = read_states(game_directory, game_dir)
      # actions = read_actions(game_directory, game_dir)
      # rewards = (read_rewards(game_directory, game_dir))[0]
      # training_information = read_train_info(game_directory, game_dir)
      # qValues = read_qValue(game_directory, game_dir)
      # assert states.shape[0] == actions.shape[0] and actions.shape[0] == rewards.shape[0]
      
      event_number = len(game)
      # initialAction = (int)(game[0][1][1] == 1)
      # actionlist = np.stack((initialAction, initialAction, initialAction, initialAction))
      beginflag = False
      count += 1
      
      for index in range(0, event_number):
        # action = self.problem.actions[
        # unicodedata.normalize('NFKD', actions[index]).encode('ascii', 'ignore').strip()]
        # nextaction = self.problem.actions[
        #   unicodedata.normalize('NFKD', actions[(index + 1) % event_number]).encode('ascii', 'ignore').strip()]
        
        # calibrate_name_str = unicodedata.normalize('NFKD', training_information[index]).encode('ascii',
        #                                                                                        'ignore')
        # calibrate_name_dict = ast.literal_eval(calibrate_name_str)
        # home_identifier = int(calibrate_name_dict.get('home'))
        
        # Episodic means we are training, not episodic means we are extracting Q-values
        game_info = game[index]
        states = game_info[0]
        action = game_info[1][1]
        qValue = game_info[2]
        currentObs = np.reshape(states, 14400)
        nextObs = currentObs
        
        if self.problem.isEpisodic:
          # actionlist = np.append(game[(index + 1) % event_number][1][1] == 1, actionlist[:3])
          
          if index == event_number - 1:
            nextObs = np.array([-1 for i in range(len(currentObs))])  # one game end
          # elif action == 5:
          #   # reward = 1
          #   print "reward:" + str(reward)  # goal
          
          # This should only apply once to ensure no duplicate instances
          if count <= checkpoint:
            self.update(currentObs, nextObs, action, qValue, beginflag=beginflag)
          elif index % self.cff == self.cff - 1:  # check fringe, check fringe after cff iterations
            self.update(currentObs, nextObs, action, qValue, check_linear=1, check_fringe=1, beginflag=beginflag)
          else:
            self.update(currentObs, nextObs, action, qValue, beginflag=beginflag)
          
          # reset begin flag
          # if action == 5:
          #   beginflag = True
          # else:
          #   beginflag = False
        
        else:
          if random.randint(0, 100) % 5 == 0 and count > 60:
            q_reg, q_tree = self.getQ(currentObs, action)
            if abs((max(q_reg) - max(qValue))) < abs((max(q_tree) - max(qValue))):
              Qlist.append([max(q_reg), max(qValue)])
            else:
              Qlist.append([max(q_tree), max(qValue)])
            inscount += 1
            if inscount % 100 == 0:
              print("Count:", inscount)
              Q_trans = np.transpose(Qlist)
              MAE = mean_absolute_error(Q_trans[0], Q_trans[1])
              MSE = mean_squared_error(Q_trans[0], Q_trans[1])
              Cor = np.corrcoef(Q_trans[0],Q_trans[1])
              MAElist.append(MAE)
              MSElist.append(MSE)
              Corlist.append(Cor[0][1])
              Qlist = []
              if inscount == 500:
                with open(Q_PATH + "regression" + ".csv", 'a', newline='') as csvfile:
                  writer = csv.writer(csvfile)
                  writer.writerow(MAElist)
                  writer.writerow(MSElist)
                  writer.writerow(Corlist)
                exit(0)
          else:
            continue
      
      if self.problem.isEpisodic:
        if checkpoint < count:
          self.utree.print_tree()
          # pickle.dump(self.utree, open(TREE_PATH + "Game_File_" + str(count) + '.p', 'wb'))
          # exit(0)
          self.utree.tocsvFile(TREE_PATH + "Game_File_" + str(count))
          exit(0)
          # self.utree.tocsvFile(HOME_PATH + "Game_File_" + str(count) + ".csv")
        # print out tree info
        print("Game File " + str(count))
        print("")
