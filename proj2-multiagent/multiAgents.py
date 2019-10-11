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

    def __init__(self, index=0):
        self.index = index
        self.lastMoveReverse = None

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
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        lastMove = legalMoves[chosenIndex]
        self.lastMoveReverse = Directions.REVERSE[lastMove]
        # print(lastMove)
        return lastMove

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
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
        curPos = currentGameState.getPacmanPosition()

        score = successorGameState.getScore()

        nearestScore = 0
        nearestDis = 10000
        for foodPos in newFood.asList():
            curDis = manhattanDistance(foodPos, curPos)
            distanceReduce = curDis - manhattanDistance(foodPos, newPos)
            if curDis < nearestDis:
                nearestDis = curDis
                nearestScore = distanceReduce

        for capPos in successorGameState.getCapsules():
            curDis = manhattanDistance(capPos, curPos)
            distanceReduce = curDis - manhattanDistance(capPos, newPos)
            if curDis < nearestDis:
                nearestDis = curDis
                nearestScore = distanceReduce

        if action == Directions.STOP:
            allScore = -100000
        else:
            allScore = score * 10 + nearestScore
        return allScore


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

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


import sys


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
        mv, ma = self.maxValue(gameState, 0)
        # print(mv, ma)
        return ma

    def nextNodeValue(self, gameState, depth, agentIndex):
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0

        if agentIndex == 0:
            return self.maxValue(gameState, depth + 1)
        else:
            return self.minValue(gameState, depth, agentIndex)

    def maxValue(self, gameState, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        mv = -sys.maxsize
        ma = Directions.STOP
        for action in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, action)
            v, na = self.nextNodeValue(state, depth, 1)
            if v > mv:
                mv = v
                ma = action
        return mv, ma

    def minValue(self, gameState, depth, agentIndex):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        mv = sys.maxsize
        ma = Directions.STOP
        for action in gameState.getLegalActions(agentIndex):
            state = gameState.generateSuccessor(agentIndex, action)
            v, na = self.nextNodeValue(state, depth, agentIndex + 1)
            if v < mv:
                mv = v
                ma = action
        return mv, ma


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """

        mv, ma = self.maxValue(gameState, 0, -sys.maxsize, sys.maxsize)
        # print(mv, ma)
        return ma

    def nextNodeValue(self, gameState, depth, agentIndex, alpha, beta):
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0

        if agentIndex == 0:
            v, na = self.maxValue(gameState, depth + 1, alpha, beta)
            return v, True
        else:
            v, na = self.minValue(gameState, depth, agentIndex, alpha, beta)
            return v, False

    def maxValue(self, gameState, depth, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        mv = -sys.maxsize
        ma = Directions.STOP
        for action in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, action)
            v, _ = self.nextNodeValue(state, depth, 1, alpha, beta)
            if v > mv:
                mv = v
                ma = action

            if mv > beta:
                return mv, ma

            alpha = max(alpha, v)

        return mv, ma

    def minValue(self, gameState, depth, agentIndex, alpha, beta):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        mv = sys.maxsize
        ma = Directions.STOP
        for action in gameState.getLegalActions(agentIndex):
            state = gameState.generateSuccessor(agentIndex, action)
            v, _ = self.nextNodeValue(state, depth, agentIndex + 1, alpha, beta)
            if v < mv:
                mv = v
                ma = action

            if mv < alpha:
                return mv, ma

            beta = min(beta, v)

        return mv, ma


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
        mv, ma = self.maxValue(gameState, 0)
        return ma

    def nextNodeValue(self, gameState, depth, agentIndex):
        if agentIndex >= gameState.getNumAgents():
            agentIndex = 0

        if agentIndex == 0:
            return self.maxValue(gameState, depth + 1)
        else:
            return self.expectValue(gameState, depth, agentIndex)

    def maxValue(self, gameState, depth):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        mv = -sys.maxsize
        ma = Directions.STOP
        for action in gameState.getLegalActions(0):
            state = gameState.generateSuccessor(0, action)
            v, na = self.nextNodeValue(state, depth, 1)
            if v > mv:
                mv = v
                ma = action
        return mv, ma

    def expectValue(self, gameState, depth, agentIndex):
        if depth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState), Directions.STOP

        mv = -sys.maxsize
        ev = 0
        ec = 0
        ma = Directions.STOP
        for action in gameState.getLegalActions(agentIndex):
            state = gameState.generateSuccessor(agentIndex, action)
            v, a = self.nextNodeValue(state, depth, agentIndex + 1)
            if v > mv:
                mv = v
                ma = action
            ev += v
            ec += 1
        if ec == 0:
            ec = 1
        return ev / ec, ma


def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: 当前得分+ 剩余物品的折现值，用价值除以距离来折现。
    距离计算时如果2条折线路上都有墙则距离加上4
    药物的价值按照还没有恐惧的怪物价值和来计算
    """

    def hasWallbyWalkXFirst(x1, y1, x2, y2):
        step = 1
        if x1 > x2:
            step = -1
        for x in range(x1, x2, step):
            if currentGameState.hasWall(x, y1):
                return True

        step = 1
        if y1 > y2:
            step = -1
        for y in range(y1, y2, step):
            if currentGameState.hasWall(x2, y):
                return True

        return False

    def hasWallbyWalkYFirst(x1, y1, x2, y2):
        step = 1
        if y1 > y2:
            step = -1
        for y in range(y1, y2, step):
            if currentGameState.hasWall(x1, y):
                return True

        step = 1
        if x1 > x2:
            step = -1
        for x in range(x1, x2, step):
            if currentGameState.hasWall(x, y2):
                return True

        return False

    def manhattanDistanceWithWallPunish(xy1, xy2):
        x1, y1 = xy1
        x2, y2 = xy2
        dis = abs(x1 - x2) + abs(y1 - y2)
        if hasWallbyWalkXFirst(x1, y1, x2, y2) and hasWallbyWalkYFirst(x1, y1, x2, y2):
            dis += 4
        return dis

    curPos = currentGameState.getPacmanPosition()

    foodValue = 0
    potentialGhostValue = 0
    for ghost in currentGameState.getGhostStates():
        ghostPos = ghost.configuration.getPosition()
        ghostPos = (int(ghostPos[0]), int(ghostPos[1]))
        curDis = manhattanDistanceWithWallPunish(ghostPos, curPos)
        if ghost.scaredTimer > 0:
            foodValue += 200 / (curDis + 1)
        else:
            potentialGhostValue += 200 / (curDis + 1)

    for foodPos in currentGameState.getFood().asList():
        curDis = manhattanDistanceWithWallPunish(foodPos, curPos)
        foodValue += 10 / (curDis + 1)

    for capPos in currentGameState.getCapsules():
        curDis = manhattanDistanceWithWallPunish(capPos, curPos)
        foodValue += potentialGhostValue / (curDis + 1)

    score = currentGameState.getScore() + foodValue

    return score


# Abbreviation
better = betterEvaluationFunction
