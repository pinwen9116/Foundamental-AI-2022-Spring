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
# Reference: berkely, 
#           https://github.com/jiminsun/berkeley-cs188-pacman/edit/master/hw2/multiagent/multiAgents.py
#           https://github.com/zhangjiedev/pacman/blob/a182b2dc129f8077566a640bcc5ccf2e7c8c3bcd/multiagent/multiAgents.py#L126


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState: GameState):
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

    def evaluationFunction(self, currentGameState: GameState, action):
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
        
        "*** YOUR CODE HERE ***"

        shortest = 1200
        ghost_location = successorGameState.getGhostPosition(1)
        GhostDistance = util.manhattanDistance(ghost_location, newPos)
        score = successorGameState.getScore()
        score += max(GhostDistance, 3)
        
        for food in newFood.asList():
            dis = util.manhattanDistance(food, newPos)
            if dis < shortest:
                shortest = dis
        #print("go")
        if len(newFood.asList())< len(currentGameState.getFood().asList()):
            #print("1")
            score += 200
        score += 200/shortest
        if newPos in currentGameState.getCapsules():
            #print("2")
            score += 150
        if action == Directions.STOP:
            #print("stop")
            score -= 30
        return score

        return successorGameState.getScore()

def scoreEvaluationFunction(currentGameState: GameState):
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

    def getAction(self, gameState: GameState):
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
        #util.raiseNotDefined()
        utility = util.Counter()
        last_agent = gameState.getNumAgents() - 1 # last min agent in ply

        def minn(state, agent, depth):
            min_score = float('inf')
            if state.getLegalActions(agent):
                if agent == last_agent:
                    for act in state.getLegalActions(agent):
                        min_score = min(min_score, maxx(state.generateSuccessor(agent, act), 0, depth + 1))
                else:
                    for act in state.getLegalActions(agent):
                        min_score = min(min_score, minn(state.generateSuccessor(agent, act), agent + 1, depth))
                return min_score
            return self.evaluationFunction(state)

        def maxx(state, agent, depth):
            max_score = float('-inf')
            if depth == self.depth:
                return self.evaluationFunction(state)
            if len(state.getLegalActions(agent)) == 0:
                #print("no way")
                return self.evaluationFunction(state)
            for action in state.getLegalActions(agent):
                max_score = max(max_score, minn(state.generateSuccessor(agent, action), agent + 1, depth))
            return max_score
        
        for act in gameState.getLegalActions(0):
            utility[act] = minn(gameState.generateSuccessor(0, act), 1, 0)
        return utility.argMax()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"        
        #util.raiseNotDefined()
        utility = util.Counter()
        last_agent = gameState.getNumAgents() - 1 # last min agent in ply
        alpha = float('-inf')
        beta = float('inf')

        score = float('inf')

        def minn(state, agent, depth, alpha, beta):
            min_score = float('inf')
            if state.getLegalActions(agent):
                if agent == last_agent:
                    for act in state.getLegalActions(agent):
                        min_score = min(min_score, maxx(state.generateSuccessor(agent, act), 0, depth + 1, alpha, beta))
                        if min_score < alpha:
                            return min_score
                        beta = min(beta, min_score)
                else:
                    for act in state.getLegalActions(agent):
                        min_score = min(min_score, minn(state.generateSuccessor(agent, act), agent + 1, depth, alpha, beta))
                        if min_score < alpha:
                            return min_score
                        beta = min(beta, min_score)
                return min_score
            return self.evaluationFunction(state)

        def maxx(state, agent, depth, alpha, beta):
            max_score = float('-inf')
            if depth == self.depth:
                return self.evaluationFunction(state)
            if len(state.getLegalActions(agent)) == 0:
                #print("no way")
                return self.evaluationFunction(state)
            for action in state.getLegalActions(agent):
                max_score = max(max_score, minn(state.generateSuccessor(agent, action), agent + 1, depth, alpha, beta))
                if max_score > beta:
                    return max_score
                alpha = max(alpha, max_score)
            return max_score

        for act in gameState.getLegalActions(0):
            utility[act] = minn(gameState.generateSuccessor(0, act), 1, 0, alpha, beta)
            alpha = max(alpha, utility[act])
        return utility.argMax()

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        utility = util.Counter()
        last = gameState.getNumAgents() - 1

        def maxx(state, agent, depth):
            score = float('-inf')
            if depth == self.depth:
                return self.evaluationFunction(state)
            if len(state.getLegalActions(agent)):
                for act in state.getLegalActions(agent):
                    score = max(score, expect(state.generateSuccessor(agent, act), agent + 1, depth))
                return score
            return self.evaluationFunction(state)

        def expect(state, agent, depth):
            score = 0
            if state.getLegalActions(agent):
                prob = 1.0 / len(state.getLegalActions(agent))
                if agent == last:
                    for act in state.getLegalActions(agent):
                        score += prob * maxx(state.generateSuccessor(agent, act), 0, depth+1)
                else:
                    for act in state.getLegalActions(agent):
                        score += prob * expect(state.generateSuccessor(agent, act), agent + 1, depth)
                return score
            return self.evaluationFunction(state)
        
        for act in gameState.getLegalActions(0):
            utility[act] = expect(gameState.generateSuccessor(0, act), 1, 0)
        return utility.argMax()
        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    currentPos = currentGameState.getPacmanPosition()
    currentFood = currentGameState.getFood().asList()
    currentGhostStates = currentGameState.getGhostStates()
    currentCapsule = currentGameState.getCapsules()
    currentScore = currentGameState.getScore()
    currentScaredTimes = [ghostState.scaredTimer for ghostState in currentGhostStates]
    walls = currentGameState.getWalls().asList()
    "*** YOUR CODE HERE ***"

    score = currentGameState.getScore()
    foodDistance = [manhattanDistance(currentPos, food) for food in currentFood]
    #print(newScaredTimes)

    def count_walls_between(pos, food):
           x, y = pos
           fx, fy = food
           fx, fy = int(fx), int(fy)
       
           return sum([wx in range(min(x, fx), max(x, fx)+1) and wy in range(min(y, fy), max(y, fy)+1) for (wx, wy) in walls])

    
    if currentGameState.isWin():
        score = 10000
    else:
        closestFood = sorted(foodDistance)
        closeFoodDistance = sum(closestFood[-6:])
        closestFoodDistance = sum(closestFood[-3:])
        ghostDistance = [manhattanDistance(currentPos, ghost.getPosition()) + 2* count_walls_between(currentPos, ghost.getPosition()) for ghost in currentGhostStates]
        minGhostDistance = min(min(ghostDistance), 6)
     
        score += 0.5 * currentScaredTimes[0] + 1.0 / len(currentFood) - len(currentCapsule) + minGhostDistance + 2.0 / closeFoodDistance + 2.5 / closestFoodDistance
        if minGhostDistance < 6 and minGhostDistance >= 3:
            score += minGhostDistance

    return score
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
