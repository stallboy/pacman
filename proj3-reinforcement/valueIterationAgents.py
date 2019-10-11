# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for iter in range(self.iterations):
            nextValues = util.Counter()
            for state in self.mdp.getStates():
                qValues = [self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)]
                if qValues:
                    nextValues[state] = max(qValues)
                else:
                    nextValues[state] = self.values[state]

            self.values = nextValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
          动作发生后，必然进入一个状态
        """
        return sum([(self.mdp.getReward(state, action, nextState) + self.discount * self.getValue(nextState)) * prob
                    for nextState, prob in self.mdp.getTransitionStatesAndProbs(state, action)])

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
          可能这个状态没有可用动作
        """
        qValueActions = [(self.getQValue(state, action), action) for action in
                         self.mdp.getPossibleActions(state)]
        if qValueActions:
            return max(qValueActions)[1]
        else:
            return None

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        iter = 0
        while True:
            for state in self.mdp.getStates():
                qValues = [self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)]
                if qValues:
                    self.values[state] = max(qValues)
                iter = iter + 1
                if iter >= self.iterations:
                    return


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        allStates = self.mdp.getStates()
        parents = {state: set() for state in self.mdp.getStates()}
        for state in allStates:
            for action in self.mdp.getPossibleActions(state):
                for nextState, _ in self.mdp.getTransitionStatesAndProbs(state, action):
                    parents[nextState].add(state)

        pq = util.PriorityQueue()
        for state in allStates:
            curValue = self.getValue(state)
            qValues = [self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)]
            if qValues:
                maxQValue = max(qValues)
                diff = abs(maxQValue - curValue)
                pq.push(state, -diff)

        for iter in range(self.iterations):
            if pq.isEmpty():
                return
            state = pq.pop()
            maxQValue = max([self.getQValue(state, action) for action in self.mdp.getPossibleActions(state)])
            self.values[state] = maxQValue

            for pState in parents[state]:
                pCurValue = self.getValue(pState)
                pMaxQValue = max([self.getQValue(pState, action) for action in self.mdp.getPossibleActions(pState)])
                pDiff = abs(pMaxQValue - pCurValue)
                if pDiff > self.theta:
                    pq.update(pState, -pDiff)
