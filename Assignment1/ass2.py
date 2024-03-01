import random
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D

ACTION_SPACE_SIZE = 2
np.set_printoptions(threshold=sys.maxsize)
agent_hand_possible_values = range(0,
                                    33)  # from 4 to 32 (31 is max because if agent has 21 and he picked a card he can reach at most 32 (with ace))
dealer_hand_possible_values = range(0, 33)  # one observable card only (1 (if has ace) to 11 (if has ace))
agent_has_ace_possible_values = range(2)  # true or false

all_possible_states = [(i, j, k) for i in agent_hand_possible_values for j in dealer_hand_possible_values for k in
                        agent_has_ace_possible_values]
state_space_size = len(all_possible_states)
reversed_map_all_poss_states = {state: i for i, state in enumerate(all_possible_states)}

visits_per_state = np.zeros(state_space_size)
Vs = np.zeros(state_space_size)
env = gym.make('Blackjack-v1', natural=False, sab=False)
pi = np.zeros(state_space_size)

relevent_states = [(i, j, k) for i in range(10,22) for j in range(1,11) for k in
                        agent_has_ace_possible_values]

def update_Vs(states,Gt):
    for state in states:
        rep_state= reversed_map_all_poss_states[state]
        visits_per_state[rep_state]+=1
        Vs[rep_state] = Vs[rep_state]+(Gt-Vs[rep_state])/visits_per_state[rep_state]
        

def state_reward(state,action):
    agent , dealer, ace = state
    if action == 1:
        if agent > 21: return -1
    else:
        if agent>dealer or dealer>21:
            return 1
        elif dealer >agent:
            return -1
        else: return 0

def choose_policy(state):
    state = reversed_map_all_poss_states[state]
    if pi[state] == 0:
        choise =random.choice([0,1]) 
        return choise
    elif pi[state] ==1 : return 1
    else: return 0

def policy_eval(repeats):
    for i in range(repeats):
        observation, info = env.reset()
        states = [observation]
        Gt = 0
        terminated = False
        while(not terminated):
            # make action a from state s, and get new observation (s')
            best_policy = random.randrange(0,1)
            if best_policy<0.7:
                action = choose_policy(observation)  
            else:
                action = 1- choose_policy(observation)
            if action == 0:
                my_cards, dealer ,acc = observation
                ace = 0
                while dealer<17:
                    card = random.randint(1,13)
                    if card ==1:
                        ace = 1
                        card == 11
                    elif 14 > card > 9:
                        card = 10
                    dealer += card
                    if ace == 1 and dealer >21:
                        dealer -= 10
                        ace = 0
                    # print(f"new card:{card}, new obs: {new_obs}")
                observation = (my_cards, dealer ,acc)
                states.append(observation)
                terminated = True
                
            if (not terminated):
                observation, _, terminated, truncated, info = env.step(action)
                states.append(observation)

                
        Gt = state_reward(observation,action)
        # print(states,action,Gt)
        update_Vs(states,Gt)
        
    env.close()


def improve_policy(state):
    agent, dealer, ace = state
    stay = Vs[reversed_map_all_poss_states[(agent,dealer,ace)]]
    hit = 0
    for i in range(2,10):
        hit += Vs[reversed_map_all_poss_states[(agent+i,dealer,ace)]]
    hit += Vs[reversed_map_all_poss_states[(agent+11,dealer,True)]]*3
    if stay>hit: pi[reversed_map_all_poss_states[state]] = -1
    else: pi[reversed_map_all_poss_states[state]] = 1
    
    
def learn(repets,improv):
    for i in range(improv):
        policy_eval(repets)
        
        for state in relevent_states:
            improve_policy(state)
    
if __name__ == '__main__':
    learn(1000,1000)
    value_pi = np.zeros((11,11))
    value_Vs = np.zeros((11,11))
    for agent in range(11,22):
        for dealer in range(1,12):
            value_Vs[agent-11][dealer-1] = Vs[reversed_map_all_poss_states[(agent,dealer,0)]]

    for agent in range(11,22):
        for dealer in range(1,12):
            value_pi[agent-11][dealer-1] = pi[reversed_map_all_poss_states[(agent,dealer,0)]]


    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    # for i in range(1000):
    #     print(all_possible_states[i] , "   :  " ,visits_per_state[i])
    agent = np.arange(11,22)
    dealer = np.arange(1,12)      
    agent, dealer = np.meshgrid( dealer,agent)
    ax.plot_surface( dealer,agent, value_pi, cmap='cividis')
    plt.show()
    ax.plot_surface( dealer,agent, value_Vs, cmap='cividis')
    plt.show()


    # for agent in range(10,22):
    #     for dealer in range(0,12):
    #         if value[agent-10][dealer] > 0: 
    #             value[agent-10][dealer] =  1
    #         elif value[agent-10][dealer] == 0 : 
    #             value[agent-10][dealer] =  0
    #         else:  value[agent-10][dealer] =  -1

    # agent = np.arange(10,22)
    # dealer = np.arange(0,12)      
    # agent, dealer = np.meshgrid(agent, dealer)
    
    # ax.plot_surface(dealer,agent, value, cmap='cividis')
    # plt.show()