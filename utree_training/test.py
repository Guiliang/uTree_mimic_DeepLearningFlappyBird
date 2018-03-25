import optparse
from utree_training import Problem_flappyBird, Agent_oracle as Agent
# from utree_training import Problem_flappyBird, Agent_regression as Agent
# import Problem_flappyBird, Agent_oracle as Agent
# import Problem_flappyBird, Agent_regression as Agent
import sys

optparser = optparse.OptionParser()
optparser.add_option("-m", "--max_node_hist", dest="MAX_NODE_HIST", default=3000,
                     help="max number of instance in every node (default = 10000)")
optparser.add_option("-c", "--check_fringe_freq", dest="CHECK_FRINGE_FREQ", default=300,
                     help="check fringe frequent (default = 100)")
optparser.add_option("-d", "--directory_of_games", dest="GAME_DIRECTORY", default="",
                     help="games dir of all the games")

opts = optparser.parse_args()[0]

if __name__ == "__main__":
  ice_hockey_problem = Problem_flappyBird.flappyBird(games_directory=opts.GAME_DIRECTORY)
  CUTreeAgent = Agent.CUTreeAgent(problem=ice_hockey_problem, max_hist=opts.MAX_NODE_HIST,
                                  check_fringe_freq=opts.CHECK_FRINGE_FREQ, is_episodic=0)
  CUTreeAgent.episode(int(sys.argv[1]))
