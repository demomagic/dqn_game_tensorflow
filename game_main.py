# -----------------------------
# File: Main
# Author: Yiting Xie
# Date: 2018.9.10
# E-mail: 369587353@qq.com
# -----------------------------
import gym
import numpy as np
import argparse
from agent_dqn import Agent,process_observation
'''
ENV_NAME = 'MsPacman-v0' # game name
EPISODES = 15000
TEST_EPISODES = 100
ISTRAIN = True # False to test, true to train
'''
def main():
    parse = argparse.ArgumentParser(description="进行模型的训练。")
    parse.add_argument("--episode", help="training frequency", type = int, default=15000)
    parse.add_argument("--env_name", help="game name", default='MsPacman-v0')
    parse.add_argument("--model_type", help="'dqn' is DQN, 'ddqn' is DDoubleQN", default='dqn')
    parse.add_argument("--train", help="train or test model, False is test, True is train", default=True)
    parse.add_argument("--load_network", help="load model True or False", default=False)
    
    args = parse.parse_args()
    EPISODES = args.episode
    ENV_NAME = args.env_name
    MODEL_TYPE = args.model_type
    ISTRAIN = args.train
    LOAD_NETWORK = args.load_network
    
    env = gym.make(ENV_NAME)
    # train model
    if ISTRAIN:
        # init agent
        agent = Agent(env.action_space.n, ENV_NAME, load_network = LOAD_NETWORK, agent_model = MODEL_TYPE)
        for _ in range(EPISODES): 
            game_status = False
            # reset the state of the environment
            last_observation = env.reset()
            observation, _, _, _ = env.step(0)
            state = agent.initial_state(observation, last_observation)
            
            while not game_status:
                last_observation = observation
                action = agent.get_action(state)
                observation, reward, game_status, _ = env.step(action)
                #redraw a frame of the environment
                env.render()
                next_observation = process_observation(observation, last_observation)
                state = agent.run_agent(action, reward, game_status, next_observation, state)
    #test model
    else:
        agent = Agent(env.action_space.n, ENV_NAME, load_network = LOAD_NETWORK, agent_model = MODEL_TYPE)
        for _ in range(EPISODES): 
            game_status = False
            last_observation = env.reset()
            observation, _, _, _ = env.step(0)
            state = agent.initial_state(observation, last_observation)
            
            while not game_status:
                last_observation = observation
                action = agent.get_action_test(state)
                observation, _, game_status, _ = env.step(action)
                env.render()
                next_observation = process_observation(observation, last_observation)
                state = np.append(state[: ,:, 1:], next_observation, axis = 2)
        
if __name__ == '__main__':
    main()