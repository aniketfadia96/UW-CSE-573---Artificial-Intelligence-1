# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)

        successorGameState = currentGameState.generatePacmanSuccessor(action)
        #print(successorGameState)

        newPos = successorGameState.getPacmanPosition()
        #print("Pacman position is: ", str(newPos))

        newFood = successorGameState.getFood()
        #print(str(newFood))

        newGhostStates = successorGameState.getGhostStates()
        #print("Ghost states are: ", [type(x) for x in newGhostStates])

        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        #print("Scared Times are: ", str(newScaredTimes))

        "*** YOUR CODE HERE ***"


        if successorGameState.isWin():
            return float("inf")

        if(successorGameState.isLose()):
            return float("-inf")

        ghost_positions = [ghost.getPosition() for ghost in newGhostStates]
        ghost_distances = [util.manhattanDistance(newPos, ghost_pos) for ghost_pos in ghost_positions]
        nearest_ghost_distance = min(ghost_distances)

        food_distances = [util.manhattanDistance(newPos, food_pos) for food_pos in currentGameState.getFood().asList()]
        nearest_food_distance = min(food_distances)

        # Greedy Heuristic: Go towards the nearest food position
        evaluation_score = 1/(0.1 + nearest_food_distance)

        # Pacman has very high chances of being eaten
        if nearest_ghost_distance < 2:
            return float("-inf")

        return evaluation_score


def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"

        def minimax(gameState, agent, depth):
            # Terminal State
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            # Max's turn to play
            elif agent == 0:
                return max_agent(gameState, depth)
            # Min's turn to play
            else:
                return min_agent(gameState, agent, depth)

        def max_agent(gameState, depth):
            max_eval = float("-inf")

            for legal_action in gameState.getLegalActions(0):
                successor_game_state = gameState.generateSuccessor(0, legal_action)
                current_eval = minimax(successor_game_state, 1, depth)
                max_eval = max(max_eval, current_eval)

            return max_eval

        def min_agent(gameState, agent, depth):
            min_eval = float("inf")

            for legal_action in gameState.getLegalActions(agent):
                successor_game_state = gameState.generateSuccessor(agent, legal_action)

                # All Min's have played at the current depth
                if agent == gameState.getNumAgents() - 1:
                    current_eval = minimax(successor_game_state, 0, depth + 1)
                    min_eval = min(min_eval, current_eval)

                # Min's still playing at the same depth
                else:
                    current_eval = minimax(successor_game_state, agent + 1, depth)
                    min_eval = min(min_eval, current_eval)

            return min_eval

        max_eval = float("-inf")
        minimax_action = ""

        # Start minimax
        for legal_action in gameState.getLegalActions(0):
            successor_game_state = gameState.generateSuccessor(0, legal_action)
            current_eval = minimax(successor_game_state, 1, 0)

            if current_eval > max_eval:
                max_eval = current_eval
                minimax_action = legal_action

        return minimax_action


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        def minimax(gameState, agent, depth, alpha, beta):
            # Terminal State
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            # Max's turn to play
            elif agent == 0:
                return max_agent(gameState, depth, alpha, beta)
            # Min's turn to play
            else:
                return min_agent(gameState, agent, depth, alpha, beta)

        def max_agent(gameState, depth, alpha, beta):
            max_eval = float("-inf")

            for legal_action in gameState.getLegalActions(0):
                successor_game_state = gameState.generateSuccessor(0, legal_action)
                current_eval = minimax(successor_game_state, 1, depth, alpha, beta)
                max_eval = max(max_eval, current_eval)

                alpha = max(alpha, current_eval)
                if beta < alpha:
                    break

            return max_eval

        def min_agent(gameState, agent, depth, alpha, beta):
            min_eval = float("inf")

            for legal_action in gameState.getLegalActions(agent):
                successor_game_state = gameState.generateSuccessor(agent, legal_action)

                # All Min's have played at the current depth
                if agent == gameState.getNumAgents() - 1:
                    current_eval = minimax(successor_game_state, 0, depth + 1, alpha, beta)
                    min_eval = min(min_eval, current_eval)

                # Min's still playing at the same depth
                else:
                    current_eval = minimax(successor_game_state, agent + 1, depth, alpha, beta)
                    min_eval = min(min_eval, current_eval)

                beta = min(beta, current_eval)
                if beta < alpha:
                    break

            return min_eval

        max_eval = float("-inf")
        minimax_action = ""

        alpha = float("-inf")
        beta = float("inf")

        # Start alpha-beta pruning
        for legal_action in gameState.getLegalActions(0):
            successor_game_state = gameState.generateSuccessor(0, legal_action)
            current_eval = minimax(successor_game_state, 1, 0, alpha, beta)

            if current_eval > max_eval:
                max_eval = current_eval
                minimax_action = legal_action

            alpha = max(alpha, current_eval)
            if beta < alpha:
                break

        return minimax_action


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"

        def minimax(gameState, agent, depth):
            # Terminal State
            if depth == self.depth or gameState.isWin() or gameState.isLose():
                return self.evaluationFunction(gameState)
            # Max's turn to play
            elif agent == 0:
                return max_agent(gameState, depth)
            # Min's turn to play
            else:
                return chance_agent(gameState, agent, depth)

        def max_agent(gameState, depth):
            max_eval = float("-inf")

            for legal_action in gameState.getLegalActions(0):
                successor_game_state = gameState.generateSuccessor(0, legal_action)
                current_eval = minimax(successor_game_state, 1, depth)
                max_eval = max(max_eval, current_eval)

            return max_eval

        def chance_agent(gameState, agent, depth):
            expected_val = 0

            for legal_action in gameState.getLegalActions(agent):
                successor_game_state = gameState.generateSuccessor(agent, legal_action)

                # All Min's have played at the current depth
                if agent == gameState.getNumAgents() - 1:
                    current_eval = minimax(successor_game_state, 0, depth + 1)

                # Min's still playing at the same depth
                else:
                    current_eval = minimax(successor_game_state, agent + 1, depth)

                # Calculate expected values for the chance nodes
                expected_val = expected_val + current_eval

            return expected_val

        max_eval = float("-inf")
        minimax_action = ""

        # Start expectimax
        for legal_action in gameState.getLegalActions(0):
            successor_game_state = gameState.generateSuccessor(0, legal_action)
            current_eval = minimax(successor_game_state, 1, 0)

            if current_eval > max_eval:
                max_eval = current_eval
                minimax_action = legal_action

        return minimax_action


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: I have chosen the heuristic on the below criteria -
    (1) If the pacman is less than two distance away from the ghost, then there are very high chances that pacman will
    lose the game -> Return -infinity
    (2) In cases other than (1), the score will be dependent upon the nearest food distance, nearest ghost distance,
    the number of food left to eat, and the sum of scared timer. I have given weights of 6 to number of food left to
    eat. This is because this is the ultimate goal of our game. I have also given a weight of 0.5 to sum of scared
    timer.

    """
    "*** YOUR CODE HERE ***"
    if currentGameState.isWin():
        return float("inf")

    if currentGameState.isLose():
        return float("-inf")

    current_score = currentGameState.getScore()

    pacman_position = currentGameState.getPacmanPosition()
    ghost_state = currentGameState.getGhostStates()

    ghost_positions = [ghost.getPosition() for ghost in ghost_state]
    ghost_distances = [util.manhattanDistance(pacman_position, ghost_pos) for ghost_pos in ghost_positions]
    nearest_ghost_distance = min(ghost_distances)

    if nearest_ghost_distance < 2:
        return float("-inf")

    food_distances = [util.manhattanDistance(pacman_position, food_pos) for food_pos in currentGameState.getFood().asList()]
    nearest_food_distance = min(food_distances)

    newScaredTimes = [ghostState.scaredTimer for ghostState in ghost_state]

    return current_score - nearest_food_distance - 6*currentGameState.getNumFood() + nearest_ghost_distance +\
           0.5*sum(newScaredTimes)


# Abbreviation
better = betterEvaluationFunction
