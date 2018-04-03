# Using oracle coaching (initialize by neural network)

import numpy as np
import scipy.io as sio
import os
import pickle
import inspect
import random
import csv
import seaborn as sns
from utree_training import C_UTree_oracle as C_UTree
# import C_UTree_oracle as C_UTree
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error

TREE_PATH = "save_utree_7_g/"
HOME_PATH = "/local-scratch/csv_oracle/"
Q_PATH = "save_q/"
RDEC = 1.1

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
  
  def update(self, currentObs, nextObs, action, qValue, value_iter=0, check_fringe=0,
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
    
    if check_fringe:
      self.utree.testFringe()  # ks test is performed here
  
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
    q = self.utree.getInstanceQvalues(i, None)
    return q
  
  def getFig(self, currentObs, action):
    t = self.utree.getTime()
    i = C_UTree.Instance(t, currentObs, action, currentObs, None, None, None)
    return self.utree.getInstanceFigure(i)
    
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
    RMSElist = [str(checkpoint), "RMSE"]
    RAElist = [str(checkpoint), "RAE"]
    RRSElist = [str(checkpoint), "RRSE"]
    # checkpoint = 26
    if checkpoint > 0:
      self.utree.fromcsvFile(TREE_PATH + "Game_File_" + str(checkpoint) + ".csv")
      # return
    
    for game_dir in game_dir_all:
      
      game = read_states(game_directory, game_dir)
      # actions = read_actions(game_directory, game_dir)
      # rewards = (read_rewards(game_directory, game_dir))[0]
      # training_information = read_train_info(game_directory, game_dir)
      # qValues = read_qValue(game_directory, game_dir)
      # assert states.shape[0] == actions.shape[0] and actions.shape[0] == rewards.shape[0]
      
      event_number = len(game)
      initialAction = (int)(game[0][1][1] == 1)
      actionlist = np.stack((initialAction, initialAction, initialAction, initialAction))
      beginflag = False
      count += 1
      
      # if count < 30:
      #   continue
        
      for index in range(0, event_number):
        # action = self.problem.actions[
        # unicodedata.normalize('NFKD', actions[index]).encode('ascii', 'ignore').strip()]
        # nextaction = self.problem.actions[
        #   unicodedata.normalize('NFKD', actions[(index + 1) % event_number]).encode('ascii', 'ignore').strip()]
        
        # calibrate_name_str = unicodedata.normalize('NFKD', training_information[index]).encode('ascii',
        #                                                                                        'ignore')
        # calibrate_name_dict = ast.literal_eval(calibrate_name_str)
        # home_identifier = int(calibrate_name_dict.get('home'))
        
        game_info = game[index]
        states = game_info[0]
        action = actionlist[0]
        qValue = game_info[2]
        if (isinstance(qValue, list)):
          qValue = max(qValue)
        currentObs = np.insert(np.reshape(states, 14400), 14400, actionlist[1:])
        nextObs = currentObs
        actionlist = np.append(game[(index + 1) % event_number][1][1] == 1, actionlist[:3])
        
        # Episodic means we are training, not episodic means we are extracting Q-values
        if self.problem.isEpisodic:
          if index == event_number - 1:
            nextObs = np.array([-1 for i in range(len(currentObs))])  # one game end
          # elif action == 5:
          #   # reward = 1
          #   print "reward:" + str(reward)  # goal
          
          # This should only apply once to ensure no duplicate instances
          if count <= checkpoint:
            self.update(currentObs, nextObs, action, qValue, beginflag=beginflag)
          elif index % self.cff == 0:  # check fringe, check fringe after cff iterations
            self.update(currentObs, nextObs, action, qValue, value_iter=1, check_fringe=1, beginflag=beginflag)
          else:
            self.update(currentObs, nextObs, action, qValue, beginflag=beginflag)
          
          # reset begin flag
          # if action == 5:
          #   beginflag = True
          # else:
          #   beginflag = False
        
        else:
          if count > 0 and random.randint(0, 100) % 50 == 0: # and actionlist[3] == 1:
            # q_tree = self.getQ(currentObs, action)
            # Qlist.append([q_tree, qValue])
            # inscount += 1
            # if inscount % 100 == 0:
            #   print("Count:", inscount)
            #   Q_trans = np.transpose(Qlist)
            #   MAE = mean_absolute_error(Q_trans[0], Q_trans[1])
            #   MSE = mean_squared_error(Q_trans[0], Q_trans[1])
            #   RMSE = MSE ** 0.5
            #   Cor = np.corrcoef(Q_trans[0],Q_trans[1])
            #
            #   target_origin_list = np.asarray(Q_trans[1])
            #   ymean = sum(target_origin_list) / len(target_origin_list)
            #   ymae = sum(abs(target_origin_list - ymean)) / len(target_origin_list)
            #   ymse = sum((target_origin_list - ymean) ** 2) / len(target_origin_list)
            #
            #   RAE = MAE / ymae
            #   RRSE = MSE / ymse
            #
            #   MAElist.append(MAE)
            #   RMSElist.append(RMSE)
            #   Corlist.append(Cor[0][1])
            #   RAElist.append(RAE)
            #   RRSElist.append(RRSE)
            #   Qlist = []
            #   if inscount == 500:
            #     print('average coe:{0}, mae:{1}, rmse:{2}, rae:{3}, rrse:{4}, leaves:{5}'.
            #           format(float(sum(Corlist[2:])) / len(Corlist[2:]),
            #                  float(sum(MAElist[2:])) / len(MAElist[2:]),
            #                  float(sum(RMSElist[2:])) / len(RMSElist[2:]),
            #                  float(sum(RAElist[2:])) / len(RAElist[2:]),
            #                  float(sum(RRSElist[2:])) / len(RRSElist[2:]),
            #                  (self.utree.node_id_count - 1) / 2))
            #     with open(Q_PATH + "mimic" + ".csv", 'a', newline='') as csvfile:
            #       writer = csv.writer(csvfile)
            #       writer.writerow(MAElist)
            #       writer.writerow(RMSElist)
            #       writer.writerow(Corlist)
            #       writer.writerow(RAElist)
            #       writer.writerow(RRSElist)
            #     exit(0)
            point_fig = self.getFig(currentObs, action)
            pickle.dump(point_fig, open(Q_PATH + "fig.p", 'wb'))
            exit(0)
          else:
            continue

      self.utree.significanceLevel /= RDEC
      
      if self.problem.isEpisodic:
        if checkpoint <= count:
          influence = self.utree.getFeatureInfluence()
          pickle.dump(influence, open(Q_PATH + "influence.p", 'wb'))
          return
          self.utree.print_tree()
          # pickle.dump(self.utree, open(TREE_PATH + "Game_File_" + str(count) + '.p', 'wb'))
          # exit(0)
          self.utree.tocsvFile(TREE_PATH + "Game_File_" + str(count) + ".csv")
          exit(0)
          # self.utree.tocsvFile(HOME_PATH + "Game_File_" + str(count) + ".csv")
        # print out tree info
        print("Game File " + str(count))
        print("")
