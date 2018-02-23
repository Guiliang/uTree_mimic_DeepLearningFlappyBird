# Using oracle coaching (initialize by neural network)

import random

import numpy as np
import optparse
import sys
import csv

from scipy.stats import ks_2samp

NodeSplit = 0
NodeLeaf = 1
NodeFringe = 2
ActionDimension = -1
AbsorbAction = 5
HOME = 0
AWAY = 1
MAX_DEPTH = 20


class CUTree:
  def __init__(self, gamma, n_actions, dim_sizes, dim_names, max_hist, max_back_depth=1, minSplitInstances=10,
               significance_level=0.0005, is_episodic=0):
    
    self.node_id_count = 0
    self.root = UNode(self.genId(), NodeLeaf, None, n_actions, 1)
    self.n_actions = n_actions
    self.max_hist = max_hist
    self.max_back_depth = max_back_depth
    self.gamma = gamma
    self.history = []
    self.n_dim = len(dim_sizes)
    self.dim_sizes = dim_sizes
    self.dim_names = dim_names
    self.minSplitInstances = minSplitInstances
    self.significanceLevel = significance_level
    
    self.nodes = {self.root.idx: self.root}  # root_id:root_node
    
    self.term = UNode(self.genId(), NodeLeaf, None, 1, 1)  # dummy terminal node with 0 value
    self.start = UNode(self.genId(), NodeLeaf, None, 1, 1)
    self.nodes[self.term.idx] = self.term  # term_id:term_node
    self.nodes[self.start.idx] = self.start
  
  def tocsvFile(self, filename):
    '''
    Store a record of U-Tree in file, make it easier to rebuild tree
    :param filename: the path of file to store the record
    :return:
    '''
    with open(filename, 'w', newline='') as csvfile:
      fieldname = ['idx', 'dis', 'dis_value', 'par', 'q_home', 'q_away']
      writer = csv.writer(csvfile)
      writer.writerow(fieldname)
      for i, node in self.nodes.items():
        if node.nodeType == NodeSplit:
          writer.writerow([node.idx,
                           node.distinction.dimension,
                           node.distinction.continuous_divide_value if node.distinction.continuous_divide_value else None,
                           node.parent.idx if node.parent else None,
                           None,
                           None])
        else:
          writer.writerow([node.idx,
                           None,
                           None,
                           node.parent.idx if node.parent else None,
                           node.qValues_home,
                           node.qValues_away])
  
  def tocsvFileComplete(self, filename):
    '''
    Store a record of U-Tree in file including the instances in each leaf node,
    make it easier to rebuild tree
    :param filename: the path of file to store record
    :return:
    '''
    with open(filename, 'w', newline='') as csvfile:
      fieldnamecomplete = ['idx', 'dis', 'dis_value', 'par', 'q_home', 'q_away', 'instances']
      writer = csv.writer(csvfile)
      writer.writerow(fieldnamecomplete)
      for i, node in self.nodes.items():
        if node.nodeType == NodeSplit:
          writer.writerow([node.idx,
                           node.distinction.dimension,
                           node.distinction.continuous_divide_value if node.distinction.continuous_divide_value else None,
                           node.parent.idx if node.parent else None,
                           None,
                           None,
                           None])
        else:
          writer.writerow([node.idx,
                           None,
                           None,
                           node.parent.idx if node.parent else None,
                           node.qValues_home,
                           node.qValues_away,
                           [inst.timestep for inst in node.instances]])
  
  def fromcsvFile(self, filename):
    '''
    Load U-Tree structure from csv file
    :param filename: the path of file to load record
    :return:
    '''
    with open(filename, 'r') as csvfile:
      fieldname = ['idx', 'dis', 'dis_value', 'par', 'q_home', 'q_away']
      reader = csv.reader(csvfile)
      self.node_id_count = 0
      for record in reader:
        if not record:
          continue
        if record[0] == fieldname[0]:  # idx determines header or not
          continue
        if not record[4]:  # qValues determines NodeSplit or NodeLeaf
          node = UNode(int(record[0]), NodeSplit, self.nodes[int(record[3])] if record[3] else None,
                       self.n_actions, self.nodes[int(record[3])].depth + 1 if record[3] else 1)
          node.distinction = Distinction(dimension=int(record[1]),
                                         back_idx=0,
                                         dimension_name=self.dim_names[int(record[1])],
                                         iscontinuous=True if record[2] else False,
                                         continuous_divide_value=float(record[2]) if record[
                                           2] else None)  # default back_idx is 0
        else:
          node = UNode(int(record[0]), NodeLeaf, self.nodes[int(record[3])] if record[3] else None,
                       self.n_actions, self.nodes[int(record[3])].depth + 1 if record[3] else 1)
          node.qValues_home = float(record[4])
          node.qValues_away = float(record[5])
        if node.parent:
          self.nodes[int(record[3])].children.append(node)
        if node.idx == 1:
          self.root = node
        elif node.idx == 2:
          self.term = node
        self.nodes[int(node.idx)] = node
        self.node_id_count += 1
  
  def print_tree(self):
    """
    print U tree
    :return:
    """
    self.print_tree_recursive("", self.root)
  
  def print_tree_recursive(self, blank, node):
    '''
    recursively print tree from root to leaves
    :param node: the node to be expand
    :return:
    '''
    if node.nodeType == NodeSplit:
      print(blank + "idx={}, dis={}, par={}".format(node.idx,
                                                    node.distinction.dimension,
                                                    node.parent.idx if node.parent else None))
      for child in node.children:
        self.print_tree_recursive(blank + " ", child)
    else:
      print(blank + "idx={}, q_h={}, q_a={}, par={}".
            format(node.idx,
                   # node.transitions_home_home,
                   # node.transitions_home_away,
                   # node.transitions_away_home,
                   # node.transitions_away_away,
                   node.qValues_home,
                   node.qValues_away,
                   node.parent.idx if node.parent else None))
  
  def getInstanceQvalues(self, instance, reward):
    """
    get the Q-value from instance, q(I,a)
    :return: state's maximum Q
    """
    self.insertInstance(instance)
    # set goal's q_value as equal to previous shot
    next_state = self.getInstanceLeaf(instance)
    return next_state.utility(home_identifier=True), \
           next_state.utility(home_identifier=False)
  
  def getTime(self):
    """
    :return: length of history
    """
    return len(self.history)
  
  def updateCurrentNode(self, instance, beginflag):
    """
    add the new instance ot LeafNode
    :param instance: instance to add
    :return:
    """
    old_state = self.getLeaf(previous=1)  # get the leaf
    # if old_state == self.term:  # if leaf is the dummy terminal node
    #     return
    self.insertInstance(instance)  # add the new instance to U-Tree history
    new_state = self.getLeaf()  # get the leaf of next state
    new_state.addInstance(instance, self.max_hist)  # add the instance to leaf node
    if not beginflag:  # last instance is not goal and not the beginning of the game
      old_state.updateModel(None, None, None, None, self.history[-2].qValue)
    if instance.nextObs[0] == -1:  # this instance lead to goal
      new_state.updateModel(None, None, None, None, instance.qValue)
  
  def sweepLeaves(self):
    '''
    Serve as a public function calls sweepRecursive
    :return:
    '''
    return self.sweepRecursive(self.root, self.gamma)
  
  def sweepRecursive(self, node, gamma):
    """
    Apply single step of value iteration to leaf node
    or recursively to children if it is a split node
    :param node: target node
    :param gamma: gamma in dynamic programming
    :return:
    """
    if node.nodeType == NodeLeaf:
      # home team
      for action, reward in enumerate(node.rewards_home):
        c = float(node.count_home[action])  # action count
        if c == 0:
          continue
        exp = 0
        for node_to, t_h in node.transitions_home_home[action].items():
          t_a = node.transitions_home_away[action][node_to]
          if reward[node_to] > 0:
            exp += reward[node_to] / c
          if node.idx != node_to:
            exp += gamma * (self.nodes[node_to].utility(True) * t_h + self.nodes[node_to].utility(False) * t_a) / c
        node.qValues_home[action] = exp
      
      # away team
      for action, reward in enumerate(node.rewards_away):
        c = float(node.count_away[action])  # action count
        if c == 0:
          continue
        exp = 0
        for node_to, t_h in node.transitions_away_home[action].items():
          t_a = node.transitions_away_away[action][node_to]
          if reward[node_to] > 0:
            exp += reward[node_to] / c
          if node.idx != node_to:
            exp += gamma * (self.nodes[node_to].utility(True) * t_h + self.nodes[node_to].utility(False) * t_a) / c
        node.qValues_away[action] = exp
    
    # # assert is just for debugging, replace all assert to comment
    # assert node.nodeType == NodeSplit
    for c in node.children:
      self.sweepRecursive(c, gamma)
  
  def insertInstance(self, instance):
    """
    append new instance to history
    :param instance: current instance
    :return:
    """
    self.history.append(instance)
    # if len(self.history)>self.max_hist:
    #    self.history = self.history[1:]
  
  def nextInstance(self, instance):
    """
    get the next instance
    :param instance: current instance
    :return: the next instance
    """
    # assert instance.timestep + 1 < len(self.history)
    return self.history[instance.timestep + 1]
  
  def transFromInstances(self, node, n_id, action):
    """
    compute transition probability from current node to n_id node when perform action
    Formula (7) in U tree paper
    :param node: current node
    :param n_id: target node
    :param action: action to perform
    :return: transition probability
    """
    count = 0
    total = 0
    
    for inst in node.instances:
      if inst.action == action:
        leaf_to = self.getInstanceLeaf(inst, previous=1)
        if leaf_to.idx == n_id:
          count += 1
        total += 1
    
    if total:
      return count / total
    else:
      return 0
  
  def rewardFromInstances(self, node, action):
    """
    compute reward of perform action on current node
    Formula (6) in U tree paper
    :param node: current node
    :param action: action to perform
    :return: reward computed
    """
    rtotal = 0
    total = 0
    
    for inst in node.instances:
      if inst.action == action:
        rtotal += inst.reward
        total += 1
    if total:
      return rtotal / total
    else:
      return 0
  
  def modelFromInstances(self, node):
    """
    rebuild model for leaf node, with newly added instance
    :param node:
    :return:
    """
    # assert node.nodeType == NodeLeaf
    node.count = 0  # re-initialize count
    # node.transitions = {}  # re-initialize transition

    for inst in node.instances:
      # leaf_to = self.getInstanceLeaf(inst, previous=1)  # get the to node
      # # update the node, add action reward, action count and transition states
      # if leaf_to != self.term:
      #   node.updateModel(leaf_to.idx, inst.action, inst.home_identifier,
      #                    self.history[inst.timestep + 1].home_identifier, inst.qValue)
      # else:
      #   node.updateModel(leaf_to.idx, inst.action, inst.home_identifier,
      #                    inst.home_identifier, inst.qValue)
      node.updateModel(None, None, None, None, inst.qValue)
  
  def getLeaf(self, previous=0):
    '''
    Get leaf corresponding to current history
    :param previous: 0 is not check goal, 1 is check it
    :return:
    '''
    idx = len(self.history) - 1
    node = self.root
    
    if previous == 1:
      if idx == -1 or self.history[idx].nextObs[0] == -1:
        return self.start
    
    while node.nodeType != NodeLeaf:
      # assert node.nodeType == NodeSplit
      child = node.applyDistinction(self.history, idx)
      node = node.children[child]  # go the children node
    return node
  
  def getInstanceLeaf(self, inst, ntype=NodeLeaf, previous=0):
    """
    Get leaf that inst records a transition from
    previous=0 indicates transition_from, previous=1 indicates transition_to
    :param inst: target instance
    :param ntype: target node type
    :param previous: previous=0 indicates present inst, previous=1 indicates next inst
    :return:
    """
    idx = inst.timestep + previous
    
    if previous == 1:
      if idx >= len(self.history):
        return self.term
      elif inst.nextObs[0] == -1:
        return self.start
    
    node = self.root
    while node.nodeType != ntype:  # iteratively find children
      # keep applying node's distinction until we find ntype node, where the instance should belong
      child = node.applyDistinction(self.history, idx)
      node = node.children[child]
    return node
  
  def genId(self):
    """
    :return: a new ID for node
    """
    self.node_id_count += 1
    return self.node_id_count
  
  def reduceId(self, count):
    '''
    After splitFringe(maybe something else), reduce to normal
    :param count: the reduce number
    :return:
    '''
    self.node_id_count -= count
  
  def split(self, node, distinction):
    """
    split decision tree on nodes
    :param node: node to split
    :param distinction: distinction to split
    :return:
    """
    # assert node.nodeType == NodeLeaf
    # assert distinction.back_idx >= 0
    node.nodeType = NodeSplit
    node.distinction = distinction
    
    # Add children
    if distinction.dimension == ActionDimension:
      for i in range(self.n_actions):
        idx = self.genId()
        n = UNode(idx, NodeLeaf, node, self.n_actions, node.depth + 1)
        # n.qValues_home = np.copy(node.qValues_home)
        # n.qValues_away = np.copy(node.qValues_away)
        self.nodes[idx] = n
        node.children.append(n)
    elif distinction.iscontinuous == False:
      for i in range(self.dim_sizes[distinction.dimension]):
        idx = self.genId()
        n = UNode(idx, NodeLeaf, node, self.n_actions, node.depth + 1)
        # n.qValues_home = np.copy(node.qValues_home)
        # n.qValues_away = np.copy(node.qValues_away)
        self.nodes[idx] = n
        node.children.append(n)
    else:
      for i in range(2):
        idx = self.genId()
        n = UNode(idx, NodeLeaf, node, self.n_actions, node.depth + 1)
        # n.qValues_home = np.copy(node.qValues_home)
        # n.qValues_away = np.copy(node.qValues_away)
        self.nodes[idx] = n
        node.children.append(n)
    
    # Add instances to children
    for inst in node.instances:
      n = self.getInstanceLeaf(inst, previous=0)
      # assert n.parent.idx == node.idx, "node={}, par={}, n={}".format(node.idx, n.parent.idx, n.idx)
      n.addInstance(inst, self.max_hist)
    
    # Rebuild is essential, yes, since all the transitions will change.
    for i, n in self.nodes.items():
      if n.nodeType == NodeLeaf:
        self.modelFromInstances(n)

    node.instances=[]
    # update Q-values for children
    # for n in node.children:
    #   self.sweepRecursive(n, self.gamma)
  
  def splitToFringe(self, node, distinction):
    """
    Create fringe nodes instead of leaf nodes after splitting; these nodes
    aren't used in the agent's model
    :param node: node to split
    :param distinction: distinction used for splitting
    :return:
    """
    # assert distinction.back_idx >= 0
    node.distinction = distinction
    
    # Add children
    if distinction.dimension == ActionDimension:  # ActionDimension = -1, means use action to split
      for i in range(self.n_actions):
        idx = self.genId()  # generate new id for new node
        fringe_node = UNode(idx, NodeFringe, node, self.n_actions, node.depth + 1)
        node.children.append(fringe_node)  # append new children to node
    elif distinction.iscontinuous == False:
      for i in range(self.dim_sizes[distinction.dimension]):
        idx = self.genId()
        fringe_node = UNode(idx, NodeFringe, node, self.n_actions, node.depth + 1)
        node.children.append(fringe_node)
    else:
      for i in range(2):
        idx = self.genId()
        fringe_node = UNode(idx, NodeFringe, node, self.n_actions, node.depth + 1)
        node.children.append(fringe_node)
    
    # Add instances to children
    for inst in node.instances:
      n = self.getInstanceLeaf(inst, ntype=NodeFringe, previous=0)
      # assert n.parent.idx == node.idx, "idx={}".format(n.idx)
      n.addInstance(inst, self.max_hist)  # add instance to children
  
  def unsplit(self, node):
    """
    Unsplit node
    :param node: the node to unsplit
    :return:
    """
    node.distinction = None
    self.reduceId(len(node.children))
    if node.nodeType == NodeSplit:
      # assert len(node.children) > 0
      node.nodeType = NodeLeaf
      for c in node.children:
        del self.nodes[c.idx]
      for i, n in self.nodes.items():
        if n.nodeType == NodeLeaf:
          self.modelFromInstances(n)
    
    # clear children
    node.children = []
  
  def testFringe(self):
    """
    Tests fringe nodes for viable splits, splits nodes if they're found
    :return: how many real splits it takes
    """
    return self.testFringeRecursive(self.root)  # starting from root
  
  def testFringeRecursive(self, node):
    """
    recursively perform test in fringe, until return total number of split
    :param node: node to test
    :return: number of splits
    """
    if node.depth >= MAX_DEPTH:
      return 0
    if node.nodeType == NodeLeaf:  # NodeSplit = 0 NodeLeaf = 1 NodeFringe = 2
      d = self.getUtileDistinction(node)  # test is performed here
      if d:  # if find distinction, use distinction to split
        self.split(node, d)  # please use break point to see how to split here
        return 1 + self.testFringeRecursive(node)
      return 0
    
    # assert node.nodeType == NodeSplit
    total = 0
    for c in node.children:
      total += self.testFringeRecursive(c)
    return total
  
  def getUtileDistinction(self, node):
    """
    Different kinds of tests are performed here
    1. find all the possible distinction
    2. try to split node according to distinction and get expected future discounted returns
    3. perform test until find the proper distinction, otherwise, return None
    """
    # assert node.nodeType == NodeLeaf
    if len(node.instances) < self.minSplitInstances:
      return None
    cds = self.getCandidateDistinctions(node)  # Get all the candidate distinctions
    return self.ksTestonQ(node, cds)
  
  def mseTest(self, node, cds):
    """
    Mean squared-error test is performed here.
    It is less theoretically based. So it's free to choose threshold.
    :param node: the node to test
    :param cds: a list of candidate distinction to choose
    :return: the best distinction or None is not found
    """
    # Get all expected future discounted returns for all instances in a node
    root_utils = [self.getEFDRs(node, index) for index in [HOME, AWAY]]
    root_len = len(root_utils[0])
    # calculate MSE of root
    root_predict = [sum(r) ** 2 / root_len for r in root_utils]
    root_mse = [(sum(root_val ** 2 for root_val in root_utils[index]) - root_predict[index] / (root_len - 1))
                for index in [HOME, AWAY]]
    # get the best distinction
    dist_min = self.significanceLevel / root_len
    cd_min = None
    
    for cd in cds:  # test all possible distinctions until find the one satisfy the test
      self.splitToFringe(node, cd)  # split to fringe node with split candidate
      # record
      stop = 0
      child_mse_home = []
      child_mse_away = []
      for c in node.children:
        # give action a chance to split first
        if len(c.instances) < self.minSplitInstances and cd.dimension != ActionDimension:
          stop = 1
          break
        if len(c.instances) <= 1:  # goal state
          continue
        # Get all expected future discounted returns for all instances in a children
        child_util_home = self.getEFDRs(c, HOME)
        child_util_away = self.getEFDRs(c, AWAY)
        # calculate MSE(weighted) of child
        child_len = len(child_util_home)
        child_predict_home = sum(child_util_home) ** 2 / child_len
        child_predict_away = sum(child_util_away) ** 2 / child_len
        child_weight = float(child_len) / root_len
        child_mse_home.append(sum(child_val ** 2 for child_val in child_util_home) - child_predict_home /
                              (child_len - 1) * child_weight)
        child_mse_away.append(sum(child_val ** 2 for child_val in child_util_away) - child_predict_away /
                              (child_len - 1) * child_weight)
      self.unsplit(node)  # delete split fringe node
      child_mse = [child_mse_home, child_mse_away]
      # if not enough instance in a node, stop split
      if stop == 1:
        continue
      
      # calculate difference between parents and students
      p = max(abs(sum(child_mse[index]) - root_mse[index]) for index in [HOME, AWAY])
      if p > dist_min:
        print("MSE passed, p={}, d={}, back={}".format(p, cd.dimension, cd.back_idx))
        dist_min = p
        cd_min = cd
    
    # print the best
    if cd_min:
      print("Will be split, p={}, d={}, back={}".format(dist_min, cd_min.dimension, cd_min.back_idx))
    return cd_min
  
  def ksTest(self, node, cds):
    """
    KS test is performed here
    :param node: the node to test
    :param cds: a list of candidate distinction to choose
    :return: the best distinction or None is not found
    """
    root_utils = [self.getEFDRs(node, index) for index in
                  [HOME, AWAY]]  # Get all expected future discounted returns for all instances in a node
    # get the best distinction
    dist_min = self.significanceLevel
    cd_min = None
    
    for cd in cds:  # test all possible distinctions until find the one satisfy KS test
      self.splitToFringe(node, cd)  # split to fringe node with split candidate
      # record
      stop = 0
      child_utils_home = []
      child_utils_away = []
      child_utils = [child_utils_home, child_utils_away]
      for c in node.children:
        if len(c.instances) < self.minSplitInstances and cd.dimension != ActionDimension:
          stop = 1
          break
        # Get all expected future discounted returns for all instances in a children
        child_utils_home.append(self.getEFDRs(c, HOME))
        child_utils_away.append(self.getEFDRs(c, AWAY))
      self.unsplit(node)  # delete split fringe node
      # if not enough instance in a node, stop split
      if stop == 1:
        continue
      
      # Computes the Kolmogorov-Smirnov statistic between parent EFDR and child EFDR
      for i, cu in enumerate(child_utils[HOME]):
        k, p = ks_2samp(root_utils[HOME], cu)
        if p < dist_min:  # significance_level=0.00005, if p below it, this distinction is significant
          dist_min = p
          cd_min = cd
          print("KS passed, p={}, d={}, back={}".format(p, cd.dimension, cd.back_idx))
      for i, cu in enumerate(child_utils[AWAY]):
        k, p = ks_2samp(root_utils[AWAY], cu)
        if p < dist_min:  # significance_level=0.00005, if p below it, this distinction is significant
          dist_min = p
          cd_min = cd
          print("KS passed, p={}, d={}, back={}".format(p, cd.dimension, cd.back_idx))
    
    # print the best
    if cd_min:
      print("Will be split, p={}, d={}, back={}".format(dist_min, cd_min.dimension, cd_min.back_idx))
    return cd_min
  
  def getEFDRs(self, node, index):
    """
    Get all expected future discounted returns for all instances in a node
    (q-value is just the average EFDRs)
    :param: index: if index is home, calculate based on Q_home, else Q_away
    """
    efdrs = np.zeros(len(node.instances))
    for i, inst in enumerate(node.instances):
      next_state = self.getInstanceLeaf(inst, previous=1)  # Get leaf that inst records a transition from
      # split home and away
      efdrs[i] = inst.reward if index == HOME else -inst.reward
      if node.parent and next_state != node.parent and next_state != self.term:
        next_state_util = next_state.utility(index == HOME)  # maximum Q value
        efdrs[i] += self.gamma * next_state_util  # r + gamma * maxQ
    return efdrs

  # def getEFDRs(self, node, index):
  #   """
  #   Get all expected future discounted returns for all instances in a node
  #   (q-value is just the average EFDRs)
  #   :param: index: if index is home, calculate based on Q_home, else Q_away
  #   """
  #   efdrs = np.zeros(len(node.instances))
  #   for i, inst in enumerate(node.instances):
  #     next_state = self.getInstanceLeaf(inst, previous=1)  # Get leaf that inst records a transition to
  #     if inst.action == AbsorbAction:
  #       efdrs[i] = inst.reward if index == HOME else -inst.reward
  #     else:
  #       if node.parent == next_state:
  #         efdrs[i] = inst.reward if index == HOME else -inst.reward
  #       else:
  #         if index == HOME:
  #           next_home_state_util = next_state.utility(True)  # maximum Q value
  #           efdrs[i] = inst.reward + self.gamma * next_home_state_util  # r + gamma * maxQ
  #         else:
  #           next_away_state_util = next_state.utility(False)  # maximum Q value
  #           efdrs[i] = -inst.reward + self.gamma * next_away_state_util  # r + gamma * maxQ
  #
  #   return efdrs

  def ksTestonQ(self, node, cds, diff_significanceLevel=float(0.01)):
    """
    KS test is performed here
    1. find all the possible distinction
    2. try to split node according to distinction and get expected future discounted returns
    3. perform ks test until find the proper distinction, otherwise, return None
    :param diff_significanceLevel:
    :param node:
    :return:
    """
    assert node.nodeType == NodeLeaf
    root_utils_home, root_utils_away = self.getQs(node)
    variance_home = np.var(root_utils_home)
    variance_away = np.var(root_utils_away)
    diff_max = float(0)
    cd_split = None
    for cd in cds:
      self.splitToFringe(node, cd)
      child_qs = []
      for c in node.children:
        child_qs.append(self.getQs(c))

      self.unsplit(node)
      for i, cq in enumerate(child_qs):

        if len(cq[0]) == 0 or len(cq[1]) == 0:
          continue
        else:
          variance_child_home = np.var(cq[0])
          variance_child_away = np.var(cq[1])

          diff_home = abs(variance_home - variance_child_home)
          diff_away = abs(variance_away - variance_child_away)
          diff = diff_home if diff_home > diff_away else diff_away
          if diff > diff_significanceLevel and diff > diff_max:
            diff_max = diff
            cd_split = cd
            print ('vanriance test passed, diff=', diff, ',d=', cd.dimension)

    if cd_split:
      print ('Will be split, p=', diff_max, ',d=', cd_split.dimension_name)
      return cd_split
    else:
      return cd_split

  # def varDiff(self, listA=[], listB=[], diff=0):
  #   if len(listA) == 0 or len(listB) == 0:
  #     return diff - 1
  #   mean_a = sum(listA) / len(listA)
  #   var_a = float(0)
  #   for number_a in listA:
  #     var_a += (number_a - mean_a) ** 2
  #
  #   mean_b = sum(listB) / len(listB)
  #   var_b = float(0)
  #   for number_b in listB:
  #     var_b += (number_b - mean_b) ** 2
  #
  #   return abs(var_a / len(listA) - var_b / len(listB))

  def getQs(self, node):
    """
    Get all expected future discounted returns for all instances in a node
    (q-value is just the average EFDRs)
    """
    efdrs_home = np.zeros(len(node.instances))
    efdrs_away = np.zeros(len(node.instances))
    for i, inst in enumerate(node.instances):
      efdrs_home[i] = inst.qValue[0]
      efdrs_away[i] = inst.qValue[1]

    return [efdrs_home, efdrs_away]

  def getCandidateDistinctions(self, node, select_interval=100):
    """
    construct all candidate distinctions
    :param node: target nodes
    :return: all candidate distinctions
    """
    p = node.parent
    anc_distinctions = []
    while p:
      # assert p.nodeType == NodeSplit
      anc_distinctions.append(p.distinction)
      p = p.parent  # append all the parent nodes' distinction to anc_distinctions list
    
    candidates = []
    for i in range(self.max_back_depth):
      for j in range(0, self.n_dim): # no action here
        if j > -1 and self.dim_sizes[j] == 'continuous':
          # value=sum([inst.currentObs[j] for inst in node.instances])/len(node.instances)
          # d = Distinction(dimension=j, back_idx=i, dimension_name=self.dim_names[j],
          #                 iscontinuous=True, continuous_divide_value=value)
          # if d in anc_distinctions:
          #     continue
          # candidates.append(d)
          count = 0
          for inst in sorted(node.instances, key=lambda inst: inst.currentObs[j]):
            count += 1
            # choose one from 30
            if count % select_interval != 0:
              continue
            d = Distinction(dimension=j,
                            back_idx=i,
                            dimension_name=self.dim_names[j],
                            iscontinuous=True,
                            continuous_divide_value=inst.currentObs[j])
            # we don't need duplicate distinction
            if d in anc_distinctions:
              continue
            candidates.append(d)
        else:
          d = Distinction(dimension=j,
                          back_idx=i,
                          dimension_name=self.dim_names[j])
          if d in anc_distinctions:
            continue
          candidates.append(d)
    
    return candidates


class UNode:
  def __init__(self, idx, nodeType, parent, n_actions, depth):
    self.idx = idx
    self.nodeType = nodeType
    self.parent = parent
    
    self.children = []
    
    # reward in instances maybe negative, but reward in node must be positive
    self.count = 0
    self.transitions = {}  # T(s, a, s')
    self.qValues_home = 0
    self.qValues_away = 0
    
    self.distinction = None
    self.instances = []
    
    self.depth = depth
  
  def utility(self, home_identifier):
    """
    :param: index: if index is HOME, return Q_home, else return Q_away
    :return: maximum Q value
    """
    return self.qValues_home if home_identifier else self.qValues_away
  
  def addInstance(self, instance, max_hist):
    """
    add new instance to node instance list
    if instance length exceed maximum history length, select most recent history
    :param instance:
    :param max_hist:
    :return:
    """
    # assert (self.nodeType == NodeLeaf or self.nodeType == NodeFringe)
    self.instances.append(instance)
    if len(self.instances) > max_hist:
      self.instances = self.instances[1:]
  
  def updateModel(self, new_state, action, home_identifier, next_home_identifier, qValue):
    """
    1. add action reward
    2. add action count
    3. record transition states
    :param new_state: new transition state
    :param action: new action
    :param reward: reward of action
    :param home_identifier: identify home and away
    :return:
    """
    self.qValues_home = (self.count * self.qValues_home + qValue[0]) \
                                / (self.count + 1)
    self.qValues_away = (self.count * self.qValues_away + qValue[1]) \
                                / (self.count + 1)
    self.count += 1

    # if new_state not in self.transitions:
    #   self.transitions[new_state] = 1
    # else:
    #   self.transitions[new_state] += 1
  
  def applyDistinction(self, history, idx, previous=0):
    """
    :param history: history of instances
    :param idx: the idx of instance to apply distinction
    :return: the index of children
    """
    # assert self.nodeType != NodeFringe
    # assert len(history) > self.distinction.back_idx
    # assert len(history) > idx
    # assert self.distinction.back_idx >= 0
    
    # if back_idx is too far for idx, pick the first child
    # if self.distinction.back_idx > idx:
    #   return 0
    
    # find the instance from history, may back trace to former instance
    inst = history[idx - self.distinction.back_idx]
    
    if self.distinction.dimension == ActionDimension:
      return inst.action  # action distinction
    # assert self.distinction.dimension >= 0
    # previous 0: current node, previous 1: last node
    if previous == 0:
      if self.distinction.iscontinuous:
        if inst.currentObs[self.distinction.dimension] <= self.distinction.continuous_divide_value:
          return 0
        else:
          return 1
      else:
        return int(inst.currentObs[self.distinction.dimension]!=0)
    else:
      if self.distinction.iscontinuous:
        if inst.nextObs[self.distinction.dimension] <= self.distinction.continuous_divide_value:
          return 0
        else:
          return 1
      else:
        return int(inst.nextObs[self.distinction.dimension])


class Instance:
  """
  records the transition as an instance
  """
  
  def __init__(self, timestep, currentObs, action, nextObs, reward, home_identifier, qValue):
    self.timestep = int(timestep)
    # self.action = int(action)
    self.nextObs = nextObs  # record the state data
    self.currentObs = currentObs  # record the state data
    # self.reward = reward  # reserve for getEFDR only
    # self.home_identifier = home_identifier
    self.qValue = qValue


class Distinction:
  """
  For split node
  """
  
  def __init__(self, dimension, back_idx, dimension_name='unknown', iscontinuous=False, continuous_divide_value=None):
    """
    initialize distinction
    :param dimension: split of the node is based on the dimension
    :param back_idx: history index, how many time steps backward from the current time this feature will be examined
    :param dimension_name: the name of dimension
    :param iscontinuous: continuous or not
    :param continuous_divide_value: the value of continuous division
    """
    self.dimension = dimension
    self.back_idx = back_idx
    self.dimension_name = dimension_name
    self.iscontinuous = iscontinuous
    self.continuous_divide_value = continuous_divide_value
  
  def __eq__(self, distinction):
    return self.dimension == distinction.dimension and self.back_idx == distinction.back_idx \
           and self.continuous_divide_value == distinction.continuous_divide_value
