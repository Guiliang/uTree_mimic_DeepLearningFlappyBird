from datetime import datetime


class flappyBird:
  """
  An MDP. Contains methods for initialisation, state transition.
  Can be aggregated or unaggregated.
  """
  
  def __init__(self, games_directory, gamma=1):
    assert games_directory is not None
    self.games_directory = games_directory
    
    self.actions = {'act0': 0,
                    'act1': 1}
    # self.actions = {'block': 0,
    #                 'carry': 1,
    #                 'check': 2,
    #                 'dumpin': 3,
    #                 'dumpout': 4,
    #                 'goal': 5,
    #                 'lpr': 6,
    #                 'offside': 7,
    #                 'pass': 8,
    #                 'puckprotection': 9,
    #                 'reception': 10,
    #                 'shot': 11,
    #                 'shotagainst': 12}
    #
    # self.stateFeatures = {'velocity_x': 'continuous',
    #                       'velocity_y': 'continuous',
    #                       'xAdjCoord': 'continuous',
    #                       'yAdjCoord': 'continuous',
    #                       'time remained': 'continuous',
    #                       'scoreDifferential': 'continuous',
    #                       'Penalty': 3,
    #                       'duration': 'continuous',
    #                       'event_outcome': 2,
    #                       'home': 2,
    #                       'away': 2,
    #                       'angel2gate': 'continuous'}  # {feature_name:continuous/feature_dimension}
    
    self.gamma = gamma
    
    self.reset = None
    self.isEpisodic = True
    # self.isEpisodic = False
    self.nStates = 14400  # 3 previous actions
    self.dimNames = ['point' + str(i) for i in range(self.nStates)]
    self.dimSizes = [2 for i in range(self.nStates)]
    
    d = datetime.today().strftime('%d-%m-%Y--%H:%M:%S')
    
    # 'Action States' indicates stateFeatures also contains action, 'Feature States' indicates stateFeatures contains only features
    self.probName = "{0}_gamma={1}_mode={2}".format(d, gamma,
                                                    "Action Feature States" if self.actions else "Feature States")
    
    # self.games_directory = '/mnt/d/Solution code/data/'
    self.games_directory = 'D:\\Solution code\\python\\uTree_mimic_DeepLearningFlappyBird\\save_all_transitions\\'  # test on my windows computer
    # self.games_directory = '/home/bill/data/'  # test on my ubuntu VM
    # self.games_directory = '/cs/oschulte/Galen/Hockey-data-entire/State-Hockey-Training-All-feature5-scale-neg_reward_v_correct_/'
