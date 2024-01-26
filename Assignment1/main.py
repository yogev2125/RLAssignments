import random

import gym
import numpy as np


ACTION_SPACE_SIZE = 2


def get_initial_policy(n_s_a):
    ans = np.zeros(n_s_a.shape)
    for state_idx in range(n_s_a.shape[0]):
        denominator = n_s_a[state_idx].sum() # total visits in state s
        if denominator > 0:
            for action_idx in range(n_s_a.shape[1]):
                nominator = n_s_a[state_idx][action_idx] # total moves from s with action a
                ans[state_idx][action_idx] = nominator / denominator
                ## print(state_idx, action_idx, ans[state_idx][action_idx], nominator, denominator)
    return ans # a matrix where cell i,j is the chance to do the ith action from the jth state

def learn_model():


    # n_s_a_stag[i][a][j] = enumeration of the times during simulation we reached from ith state to  jth state by action a
    n_s_a_stag = np.zeros(shape=(state_space_size, ACTION_SPACE_SIZE, state_space_size))

    # n_s_stag[i][j] = enumeration of the times during simulation we reached from the ith state to the jth state
    n_s_a = np.zeros(shape=(state_space_size, ACTION_SPACE_SIZE))

    env = gym.make('Blackjack-v1', natural=False, sab=False)
    observation, info = env.reset()

    for _ in range(1000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        s = observation  # state (s)
        print(f'prev observation = {s}')
        # make action a from state s, and get new observation (s')
        observation, reward, terminated, truncated, info = env.step(action)
        print(f'action = {action}, reward = {reward}')
        print(f'new observation = {observation}')
        if terminated or truncated:
            print("Game over!")
            observation, info = env.reset()
            print("New game:")

        # count for the key of s,a (current observation and action)
        n_s_a[reversed_map_all_poss_states[s]][action] += 1
        # increment count for the key of times performed a from s and got s' (updated observation)
        n_s_a_stag[reversed_map_all_poss_states[s]][action][reversed_map_all_poss_states[observation]] += 1

    env.close()
    return transition_matrix(n_s_a, n_s_a_stag), get_initial_policy(n_s_a)



def transition_matrix(n_s_a:np.ndarray, n_s_a_stag:np.ndarray):
    """
    Return TR^ - the approximation of the real transition matrix TR.

    :param n_s_a: |S|x|A|  matrix which index i,j contains the observed number of transitions from observation i by making the jth action
    :param n_s_a_stag: |S|x|A|x|S| tensor which index i,j,k contains the observed number of transitions from observation i to observation k by making the jth action
    :return: The *estimator* of transition matrix 'TR' ('learned TR') = 'TR^' which produced by mse, see slide 5 here: https://moodle.bgu.ac.il/moodle/pluginfile.php/4327963/mod_resource/content/2/RL_Course-Class%205.pdf
    item i,j,k in the return val is n(s,a,s')/n(s,a) while s is the ith state (or observation), a is the jth action, and k is the kth state (or obervation).
    It represents the best approximation we have to the real value of p(s(t+1) = s' | s(t) = s, action(t) = a)
    """
    ans = np.zeros(n_s_a_stag.shape) # TR_hat
    num_states, num_actions = n_s_a.shape

    for i in range(num_states):
        for j in range(num_actions):
            # Calculate the denominator (n(s, a))
            denominator = n_s_a[i, j] # scaler
            # check if the denominator is non-zero to avoid division by zero
            if denominator != 0:
                # calculate the numerator (n(s, a, s'))
                numerator = n_s_a_stag[i, j, :] # vector

                # calculate the estimated transition probabilities and update
                ans[i, j, :] = numerator / denominator

    rounded_ans = np.round(ans, 3)
    print(rounded_ans)
    return rounded_ans

if __name__ == '__main__':
    agent_hand_possible_values = range(0,
                                       33)  # from 4 to 32 (31 is max because if agent has 21 and he picked a card he can reach at most 32 (with ace))
    dealer_hand_possible_values = range(0, 12)  # one observable card only (1 (if has ace) to 11 (if has ace))
    agent_has_ace_possible_values = range(2)  # true or false

    all_possible_states = [(i, j, k) for i in agent_hand_possible_values for j in dealer_hand_possible_values for k in
                           agent_has_ace_possible_values]
    state_space_size = len(all_possible_states)
    reversed_map_all_poss_states = {state: i for i, state in enumerate(all_possible_states)}

    tr_hat, initial_policy = learn_model()

