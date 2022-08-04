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
    def __init__(self, mdp, discount = 0.9, iterations = 100):
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
        self.values = util.Counter() # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"
        for k in range(self.iterations):
            copyValues = self.values.copy()
            for state in self.mdp.getStates():
                bestAction = self.getAction(state)
                if bestAction is not None:
                    copyValues[state] = self.getQValue(state, bestAction)

            self.values = copyValues

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        QValue = 0
        for nextState, probs in transitionStatesAndProbs:
            currReward = self.mdp.getReward(state, action, nextState)
            prevValue = self.values[nextState]
            QValue += probs * (currReward + self.discount * prevValue)

        return QValue


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        maxValue = float("-inf")
        maxAction = None
        for action in self.mdp.getPossibleActions(state):
            currValue = self.computeQValueFromValues(state, action)
            if currValue > maxValue:
                maxValue = currValue
                maxAction = action

        return maxAction


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
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
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
        "*** YOUR CODE HERE ***"
        states = self.mdp.getStates()
        numStates = len(states)

        for k in range(self.iterations):
            copyValues = self.values.copy()
            currState = states[(k % numStates)]
            bestAction = self.getAction(currState)
            if bestAction is not None:
                copyValues[currState] = self.getQValue(currState, bestAction)

            self.values = copyValues


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        self.statePredecessors = collections.defaultdict(set)
        ValueIterationAgent.__init__(self, mdp, discount, iterations)


    def computePredecessors(self):
        for state in self.mdp.getStates():
            for action in self.mdp.getPossibleActions(state):
                transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
                for nextState, probs in transitionStatesAndProbs:
                    self.statePredecessors[nextState].add(state)


    def runValueIteration(self):
        "*** YOUR CODE HERE ***"
        self.computePredecessors()
        priorityQueue = util.PriorityQueue()

        for state in self.mdp.getStates():
            if not self.mdp.isTerminal(state):
                currValue = self.values[state]
                bestAction = self.getAction(state)
                if bestAction is not None:
                    QValue = self.getQValue(state, bestAction)

                diff = abs(currValue - QValue)
                priorityQueue.push(state, -diff)

        for k in range(self.iterations):
            if priorityQueue.isEmpty():
                break

            currState = priorityQueue.pop()

            if not self.mdp.isTerminal(currState):
                bestAction = self.getAction(currState)
                if bestAction is not None:
                    self.values[currState] = self.getQValue(currState, bestAction)

            for p in self.statePredecessors[currState]:
                currPredecessorValue = self.getValue(p)

                if not self.mdp.isTerminal(p):
                    bestAction = self.getAction(p)
                    if bestAction is not None:
                        predecessorQValue = self.getQValue(p, bestAction)
                        diff = abs(currPredecessorValue - predecessorQValue)

                        if diff > self.theta:
                            priorityQueue.update(p, -diff)














