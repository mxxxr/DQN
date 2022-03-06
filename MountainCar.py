import argparse
import gym
import numpy as np
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="choose algorithm")
parser.add_argument("-a", "--algorithm", 
                    choices=['origin', 'reshape', 'HER', 'RND'], 
                    required=True, 
                    default="reshape",
                    help="choose the algorithm \
                        origin: the origin DQN \
                        reshape: the origin DQN with reshaped rewards \
                        HER: the origin DQN with HER \
                        RND: the origin DQN with RND ")
args = parser.parse_args()

network = "MountainCar_" + args.algorithm + "_net"
exec("from " + network + " import DQN")
print("Use algorithm: " + "MountainCar_" + args.algorithm)
env = gym.make('MountainCar-v0')

env = env.unwrapped
ENV_wrapped = 2000
action_space = env.action_space.n
state_space = env.observation_space.shape[0]
dqn = DQN(action_space, state_space,
                      reward_decay=0.9,
                      e_greedy=0.9,
                      replace_target_iter=100,
                      memory_size=50000,
                      )

step = 0
writer = SummaryWriter('runs/MountainCar_curve/m_' + args.algorithm)

if args.algorithm != "HER":
    for episode in range(2000):
        state = env.reset()
        terminated = False
        reward_sum = 0
        while terminated == False:
            env.render()

            if args.algorithm == "RND":
                action, rnd_reward = dqn.choose_action(state)
                next_state, reward, terminated, _ = env.step(action)
                reward_sum += reward
                dqn.store_transition(state, action, reward + rnd_reward, next_state)

            if args.algorithm == "reshape":
                action = dqn.choose_action(state)
                next_state, reward, terminated, _ = env.step(action)
                reward_sum += reward
                position, velocity = next_state
                reward = abs(position - (-0.5))     # r in [0, 1] , the higher the better
                dqn.store_transition(state, action, reward, next_state)

            if args.algorithm == "origin":
                action = dqn.choose_action(state)
                next_state, reward, terminated, _ = env.step(action)
                reward_sum += reward
                dqn.store_transition(state, action, reward, next_state)

            if step > 2000 and step % 3==0:
                dqn.learn()

            if terminated:
                dqn.learn()
                print(reward_sum)

            if step == ENV_wrapped:
                terminated = True

            state = next_state
            step += 1

        writer.add_scalar("Reward/Episode", reward_sum, episode)

else: # HER
    for episode in range(2000):
        state = env.reset()
        goal = np.random.rand(2)
        goal[0] = goal[0] * 1.8 - 1.2
        goal[1] = goal[1] * 0.14 - 0.07
        terminated = False
        reward_sum = 0
        trajectory_length = 1500

        for i in range(trajectory_length):
            env.render()
            if terminated:
                print("win")
                break
            action = dqn.choose_action(np.concatenate((state, goal)))
            dqn.store_trajectory(state, action)
            next_state, reward, terminated, _ = env.step(action)
            reward_sum += reward
            state = next_state

        for i in range(trajectory_length):
            state, action = dqn.get_trajectory(i)
            # print(state,action)
            r_t = dqn.get_new_reward(state, action, goal)
            dqn.store_transition(np.concatenate((state, goal)), action, r_t, np.concatenate((next_state, goal)))
            new_goal = dqn.get_trajectory_end()
            r_newgoal = dqn.get_new_reward(state, action, new_goal)
            dqn.store_transition(np.concatenate((state, new_goal)), action, r_newgoal, np.concatenate((next_state, new_goal)))
            
        for i in range(500):
            dqn.learn()

        dqn.distory_trajectory()

        writer.add_scalar("Reward/Episode", reward_sum, episode)
        print(reward_sum)
    
env.close()