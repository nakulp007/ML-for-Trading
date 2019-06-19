"""
Template for implementing QLearner  (c) 2015 Tucker Balch
"""

import numpy as np
import random as rand

class QLearner(object):

    def __init__(self, \
        num_states=100, \
        num_actions = 4, \
        alpha = 0.2, \
        gamma = 0.9, \
        rar = 0.5, \
        radr = 0.99, \
        dyna = 0, \
        verbose = False):
        """
        num_states integer, the number of states to consider
        num_actions integer, the number of actions available.
        alpha float, the learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
        gamma float, the discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
        rar float, random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
        radr float, random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
        dyna integer, conduct this number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
        verbose boolean, if True, your class is allowed to print debugging statements, if False, all printing is prohibited.
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose

        self.s = -1
        self.a = -1
        self.Q = (2*np.random.rand(num_states,num_actions))-1

        #Dyna
        self.d_alpha = 0.2
        #Transition count
        self.Tc = np.zeros((num_states,num_actions,num_states))
        self.Tc.fill(0.00001)
        #Transition probability
        self.T = np.zeros((num_states,num_actions,num_states))
        self.R = np.zeros((num_states,num_actions))

    """
    A special version of the query method that sets the state to s, and returns an integer action according to
    the same rules as query(), but it does not execute an update to the Q-table. This method is typically only
    used once, to set the initial state.
    """
    def querysetstate(self, s):
        """
        @summary: Update the state without updating the Q-table
        @param s: The new state
        @returns: The selected action
        """
        self.s = s
        if np.random.rand() < self.rar:
            action = rand.randint(0, self.num_actions-1)
        else:
            action = self.Q[s].argmax()
        if self.verbose: print "s =", s,"a =",action
        return action

    """
    query(s_prime, r) is the core method of the Q-Learner. It should keep track of the last state s and the last action a,
    then use the new information s_prime and r to update the Q table. The learning instance, or experience tuple is
    <s, a, s_prime, r>. query() should return an integer, which is the next action to take. Note that it should choose a
    random action with probability rar, and that it should update rar according to the decay rate radr at each step.
    """
    def query(self,s_prime,r):
        """
        @summary: Update the Q table and return an action
        @param s_prime: The new state
        @param r: The ne state
        @returns: The selected action
        """
        #get new value to update the Q table using s_prime, r
        #q_value = (1-alpha)Q[s,a] + alpha(r + gamma * Q[s', argmax_a(Q[s',a'])])
        self.Q[self.s,self.a] = ((1-self.alpha) * self.Q[self.s,self.a]) + (self.alpha * (r + (self.gamma * self.Q[s_prime,self.Q[s_prime].argmax()])))

        #should we take random action
        if np.random.rand() < self.rar:
            #take random action
            action = rand.randint(0, self.num_actions-1)
        else:
            #choose action
            action = self.Q[s_prime].argmax()

        if self.verbose: print "s =", s_prime,"a =",action,"r =",r

        #Dyna
        if self.dyna > 0:
            #print "yolo"
            #increment number of count s,a,'s happened
            self.Tc[self.s,self.a,s_prime] += 1
            #Transition probability
            self.T[self.s,self.a] = (self.Tc[self.s,self.a])/(self.Tc[self.s,self.a].sum())
            self.R[self.s, self.a] = ( ((1-self.d_alpha) * self.R[self.s, self.a]) + (self.d_alpha * r) )
            for i in range(0, self.dyna):
                d_s = rand.randint(0, self.num_states-1)
                d_a = rand.randint(0, self.num_actions-1)
                d_s_prime = self.T[d_s,d_a].argmax()
                d_r = self.R[d_s,d_a]
                self.Q[d_s,d_a] = ((1-self.alpha) * self.Q[d_s,d_a]) + (self.alpha * (d_r + (self.gamma * self.Q[d_s_prime,self.Q[d_s_prime].argmax()])))

        self.s = s_prime
        self.a = action
        self.rar = self.rar * self.radr
        return action

if __name__=="__main__":
    print "Remember Q from Star Trek? Well, this isn't him"
