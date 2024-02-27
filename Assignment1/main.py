import random
import sys
import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

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

    observation, info = env.reset()

    for _ in range(600000):
        action = env.action_space.sample()  # agent policy that uses the observation and info
        s = observation  # state (s)
        # make action a from state s, and get new observation (s')
        observation, reward, terminated, truncated, info = env.step(action)
        # print(f"action = {action}")
        if action == 0:
            my_cards, dealer ,acc = observation
            ace = 0
            new_obs = observation
            # print("action = 0, dealer pull card")
            while dealer<17:
                old_obs = new_obs
                card = random.randint(1,14)
                if card ==1:
                    ace = 1
                    card == 11
                elif 14 > card > 9:
                    card = 10
                dealer += card
                if ace == 1 and dealer >21:
                    dealer -= 10
                    ace = 0
                new_obs = (my_cards, dealer ,acc)
                # print(f"new card:{card}, new obs: {new_obs}")
                
                n_s_a[reversed_map_all_poss_states[old_obs]][action] += 1
                n_s_a_stag[reversed_map_all_poss_states[old_obs]][action][reversed_map_all_poss_states[new_obs]] += 1
            terminated = True
        # count for the key of s,a (current observation and action)
        else:
            # print(f'action result: observation = {observation}, reward = {reward}, terminated : {terminated }')
            n_s_a[reversed_map_all_poss_states[s]][action] += 1
            # increment count for the key of times performed a from s and got s' (updated observation)
            n_s_a_stag[reversed_map_all_poss_states[s]][action][reversed_map_all_poss_states[observation]] += 1
        
        if terminated or truncated:
            # print("game over\n")
            observation, info = env.reset()
            # print(f'init observation = {observation}')

            
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
    return rounded_ans

def get_Rsa(action,s_index):
    R_sa = 0 
    for new_state in all_possible_states:
        new_state_index = rev_map[new_state]
        r = get_reward(new_state)
        prob = tr_hat[s_index][action][new_state_index]
        R_sa += r*prob
    return R_sa

def policy_Evaluation(V_s,tr_hat,pi):
    """ evaluating a policy based on the policy evaluation method learned in class

    Args:
        V_s (_type_): _description_
        tr_hat (_type_): _description_
        pi (_type_): _description_

    Returns:
        _type_: _description_
    """
    V_s_new = np.zeros(state_space_size)
    for state in all_possible_states:
        v = 0
        R_sa = 0
        s_index = rev_map[state]
        for action in [0,1]:
            prob = pi[s_index][action]
            tr_vector = tr_hat[s_index,action]
            R_sa = get_Rsa(action,s_index)
            v += prob*(R_sa + np.dot(tr_vector,V_s))
        V_s_new[s_index] = v
    
    return V_s_new         
            
            
def get_reward(state):
    agent, dealer, agent_has_ace = state
    if agent > 21 or (agent < dealer and 21 >= dealer>= 17):
          return -1
    elif dealer > 16 and dealer!=agent: 
          return 1
    return 0


def policy_improvement(V_s):
    new_pi = np.zeros((state_space_size,2))
    for state in all_possible_states:
        s_index = rev_map[state]
        tr_vector_0 = tr_hat[s_index,0]
        tr_vector_1 = tr_hat[s_index,1]
        q_0 = get_Rsa(0,s_index) + np.dot(tr_vector_0,V_s)
        q_1 = get_Rsa(1,s_index) + np.dot(tr_vector_1,V_s)
        if q_0>q_1:
            new_pi[s_index][0]=1
            new_pi[s_index][1]=0
        else:
            new_pi[s_index][1]=1
            new_pi[s_index][0]=0
    return new_pi

def generate_pi(mode):
    """_summary_

    Args:
        mode (_type_): median - uniform distribution, 

    Returns:
        _type_: _description_
    """
    if mode =="median": 
        return np.full((state_space_size,2),0.5)
    
    elif mode == '21_take':
        pi =np.zeros((state_space_size,2))
        for state in all_possible_states:
            s_index = rev_map[state]
            agent = state[0]
            if agent<21:
                pi[s_index][1] = 1
            else: pi[s_index][0] = 1
        return pi

def is_init_state(state):
    agent, dealer, _ = state
    if 1 <agent< 22 and 0<dealer<12:
        return True
    return False

def calc_mean_val_of_policy(pi):
    # assmming policy is 0 or 1
    take =0 
    stop = 0
    sum_of = 0 
    for state in all_possible_states:
        if is_init_state(state):
            s_index = rev_map[state]
            if pi[s_index][1]>0.5: take+=1
            else: stop +=1
            sum_of +=1
    return take/sum_of

def Approximate_Policy_Evaluation(pi:np.ndarray) ->np.ndarray:
    """Performing the policy evaluation steps until we reach a convergence by infinite norm

    Args:
        pi (_type_): _description_

    Returns:
        _type_: _description_
    """
    V_s = np.zeros(state_space_size)
    epsilon = 0.01
    converge = False
    while not converge:
        converge = True
        # evaluate policy
        V_s_new = policy_Evaluation(V_s,tr_hat,pi)

        # check for convergence by infinite norm
        for i in range(state_space_size):
            if abs(V_s_new[i] - V_s[i])>epsilon:
                # print(abs(V_s_new[i] - V_s[i]))
                V_s = V_s_new 
                converge = False
                break
    return V_s_new
    
def policy_iteraion(k:int,mode:str) -> tuple:
    """Modified policy iteration algorithm. Returning the value of each state

    Args:
        k (_type_): num of iterations
        mode (_type_): arg for initial policy generation

    Returns:
        _type_: a tuple t where:
        t[0] (np.ndarry) = the value of each state atthe end of the last kth iter  policy itertaion process (in index i - the value of the ith state in the final policy)
        t[1] (list) = the avergae value of action by the ith plolicy for each i (reminding that action is either 0 or1) 
    """
    pi = generate_pi(mode)
    pi_array = [calc_mean_val_of_policy(pi)]
    V_s = np.zeros(state_space_size)
    for i in range(k):
       V_s = Approximate_Policy_Evaluation(pi)
       pi = policy_improvement(V_s)
       pi_array.append(calc_mean_val_of_policy(pi))
    return V_s, pi_array,pi

def is_initial_and_no_ace(state):
    """is state initial but contains no ace in the hand of agent

    Args:
        state (_type_): _description_

    Returns:
        _type_: _description_
    """
    agent, dealer, ace = state
    return ace == 0 and (4 <= agent <= 20 and 2 <= dealer <=  10)




def value_function_q3(state:tuple,V_s:np.ndarray)->float:
    """ given a value function and a state, returns the value of the state.

    Args:
        state (_type_): a state
        V_s (_type_): a value function

    Returns:
        _type_: value
    """
    s_index = rev_map[state]
    return V_s[s_index]

def select_action(state:tuple, policy:np.ndarray)->float:
    """ a method that given a policy and a state, returns the action to play (0 or 1) based on the policy

    Args:
        state (_type_): _description_
        policy (_type_): _description_

    Returns:
        _type_: an action (0 = don't take a card, 1 = take a card)
    """
    stop = policy[rev_map[state]][0]
    rand = random.random()
    if rand> stop : return 1
    else: return 0
    
    
    
    
    
if __name__ == '__main__':
    env = gym.make('Blackjack-v1', natural=False, sab=False)
    # learn Transition matrix
    tr_hat, initial_policy = learn_model()
    
    rev_map = reversed_map_all_poss_states
    k = 6
    V_s, pi_array,pi = policy_iteraion(k,'21_take')
    V_s = np.round(V_s,3)
    table = np.zeros((17,9))
    for i in range(state_space_size):
        state = all_possible_states[i]
        if is_initial_and_no_ace(state):
            agent,dealer,_ = state
            print("initial state:  ", state," value: ",pi[rev_map[state]])
            table[agent-4][dealer-2] = V_s[i]
            
    print(table)
    
    

    
    df = pd.DataFrame(table)
    # Save the DataFrame to an Excel file
    df.to_excel('output.xlsx', index=False)
    
    # plt.plot([i for i in range(k+1)], pi_array, marker='o', linestyle='-', color='b', label='graph')
    # # Adding labels and title
    # plt.xlabel('Iteration')
    # plt.ylabel('Avg. Action Value')
    # plt.title('Action prefrence to Iteration')
    
    # # Adding legend
    # plt.legend()
    # # Display the graph
    # plt.show()
    while True:
        msg = input()
        if msg == "q": break
        agent = int(input("agent:"))
        dealer = int(input("dealer:"))
        ace = int(input("ace:"))
        print("do: ", select_action((agent,dealer,ace),pi))
