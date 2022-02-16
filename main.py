# CS441-003 Winter 2022
# Programming Assignment 3
#   (Robby the Robot)
# Christopher Juncker
#
#
#
#
#

import numpy as np
import random
from enum import IntEnum
from matplotlib import pyplot as plt


# CONSTANTS & CONFIGURATION
SIZE = 10  # Robby the Robot lives in a 10 x 10 grid, surrounded by a wall
CHANCE = 0.5  # each grid square has a probability of 0.5 to contain a can

# To do a run consisting of N episodes of M steps each, use the following parameter values:
#   N = 5,000 ; M = 200 ; 𝜂 = 0.2; 𝛾 = 0.9
EPISODES = 5000
STEPS = 200
ETA = 0.2
GAMMA = 0.9


# enumeration of reward values
class Reward(IntEnum):
    CAN = 10  # Robby receives a reward of 10 for each can he picks up
    CRASH = -5  # a “reward” of −5 if he crashes into a wall
    NO_CAN = -1  # and a reward of −1 if he tries to pick up a can in an empty square.
    MOVE = 0  # (there is no penalty for moving to another square)


# enumeration of world square states
class State(IntEnum):
    EMPTY = 0
    CAN = 1
    WALL = 2


class Robby:
    def __init__(self):
        # The initial state of the grid in each episode is a random placement of cans
        self.world = self.generate_world()

        # Robby is initially placed in a random grid square
        self.col, self.row = self.random_location()

        # Keep track of the total reward gained per episode.
        self.reward = []

        # A Q-matrix, in which the rows correspond to states and the columns correspond to actions.
        # The Q-matrix is initialized to all zeros at the beginning of a run.
        # states = 3**5  # possible square values ** observable squares
        # actions = 5  # number of actions
        # self.q = np.zeros((actions, states))
        self.q = np.zeros((3, 3, 3, 3, 3, 5))

    @staticmethod
    def generate_world():
        # the walls are also going to be represented by grid squares
        size = SIZE + 2
        world = np.zeros((size, size))

        # loop and configure the grid
        for i in range(size):
            for j in range(size):
                # set up the walls
                if i == 0 or i == size - 1 or \
                        j == 0 or j == size - 1:
                    world[i][j] = State.WALL
                # randomly place the cans
                else:
                    if random.uniform(0, 1) <= CHANCE:
                        world[i][j] = State.CAN
        return world

    @staticmethod
    def random_location():
        # return a random location in the grid which is not a wall
        return random.randrange(SIZE) + 1, random.randrange(SIZE) + 1

    # Robby has five “sensors”: Current, North, South, East, and West. At any time step, these each
    # return the “value” of the respective location, where the possible values are Empty, Can, and Wall.
    def current(self):
        return int(self.world[self.col][self.row])

    def north(self):
        return int(self.world[self.col][self.row - 1])

    def south(self):
        return int(self.world[self.col][self.row + 1])

    def east(self):
        return int(self.world[self.col + 1][self.row])

    def west(self):
        return int(self.world[self.col - 1][self.row])

    # Robby has five possible actions: Move-North, Move-South, Move-East, Move-West, and Pick-Up-Can.
    # Note: if Robby picks up a can, the can is then gone from the grid.
    def pick_up_can(self):
        if self.current() == State.CAN:
            self.world[self.col][self.row] = State.EMPTY
            return Reward.CAN
        else:
            return Reward.NO_CAN

    def move_north(self):
        if self.north() == State.WALL:
            return Reward.CRASH
        else:
            self.row += 1
            return Reward.MOVE

    def move_south(self):
        if self.south() == State.WALL:
            return Reward.CRASH
        else:
            self.row -= 1
            return Reward.MOVE

    def move_east(self):
        if self.east() == State.WALL:
            return Reward.CRASH
        else:
            self.col += 1
            return Reward.MOVE

    def move_west(self):
        if self.south() == State.WALL:
            return Reward.CRASH
        else:
            self.col -= 1
            return Reward.MOVE

    # At the end of each episode, generate a new distribution of cans and place Robby in a random grid
    # square to start the next episode. (Don’t reset the Q-matrix — you will keep updating this matrix
    # over the N episodes. Keep track of the total reward gained per episode.
    def episode(self):
        reward = 0
        for i in range(STEPS):
            reward += self.time_step(i)  # pass in episode number to calculate epsilon later
        self.reward.append(reward)

        self.world = self.generate_world()
        self.col, self.row = self.random_location()

        return reward

    # At each time step t during an episode, your code should do the following:
    # • Observe Robby’s current state s_t
    # • Choose an action a_t, using -greedy action selection
    # • Perform the action
    # • Receive reward r_t (which is zero except in the cases specified above)
    # • Observe Robby’s new state s_(t+1)
    # • Update 𝑄(𝑠_𝑡, 𝑎_𝑡) = 𝑄(𝑠_𝑡, 𝑎_𝑡) + 𝜂(𝑟_𝑡 + 𝛾𝑚𝑎𝑥_𝑎′𝑄(𝑠_(𝑡+1), 𝑎′) − 𝑄(𝑠_𝑡, 𝑎_𝑡))
    def time_step(self, episode):
        # Observe Robby’s current state s_t
        state = self.observe_state()

        # Choose an action a_t, using -greedy action selection
        action = self.epsilon_greedy_action(state, episode)

        # Perform the action
        # Receive reward r_t (which is zero except in the cases specified above)
        if action == 0:
            reward = self.pick_up_can()
        elif action == 1:
            reward = self.north()
        elif action == 2:
            reward = self.south()
        elif action == 3:
            reward = self.east()
        else:
            reward = self.west()

        # Observe Robby’s new state s_(t+1)
        new_state = self.observe_state()

        # Update 𝑄(𝑠_𝑡, 𝑎_𝑡) = 𝑄(𝑠_𝑡, 𝑎_𝑡) + 𝜂(𝑟_𝑡 + 𝛾𝑚𝑎𝑥_𝑎′𝑄(𝑠_(𝑡+1), 𝑎′) − 𝑄(𝑠_𝑡, 𝑎_𝑡))
        # n = 0.9  # this is for discounting # no y is for discounting

        q = self.get_q(state, action)
        max_a_q = self.get_q(new_state, self.best_action(new_state))  # need to calculate best action for these

        new_q = q + ETA * (reward + (GAMMA * max_a_q) - q)

        self.set_q(state, action, new_q)

        return reward

    def observe_state(self):
        # state = list(range(5))
        # state = [0, 0, 0, 0, 0]
        state = np.zeros(5)
        state[0] = self.current()
        state[1] = self.north()
        state[2] = self.south()
        state[3] = self.east()
        state[4] = self.west()
        return state

    def best_action(self, state):
        # Choose an action a_t, using -greedy action selection
        action_values = np.zeros(5)
        action_values[0] = self.action_value(state, 0)
        action_values[1] = self.action_value(state, 1)
        action_values[2] = self.action_value(state, 2)
        action_values[3] = self.action_value(state, 3)
        action_values[4] = self.action_value(state, 4)
        # action_value = max(action_values)
        # action = action_values.index(action_value)
        action = np.argmax(action_values)
        return action

    def epsilon_greedy_action(self, state, episode):
        # For choosing actions with -greedy action selection, set  = 0.1 initially, and progressively
        # decrease it every 50 epochs or so until it reaches 0. After that, it stays at 0.
        # (I think 'epoch' is intended to mean 'episode' here)
        epsilon = 0.1
        while epsilon > 0 and episode > 0:
            epsilon -= 0.0002  # epsilon will hit zero at 50*50 = 2500 = halfway
            episode -= 50

        if random.uniform(0, 1) <= epsilon:
            action = random.randrange(0, 5)
        else:
            action = self.best_action(state)
        return action

    # -greedy action selection
    def action_value(self, state, action):
        # look up the right square in the q matrix
        value = self.q[int(state[0]), int(state[1]), int(state[2]), int(state[3]), int(state[4]), int(action)]

        # return its value so that the largest can be selected
        return value

    def get_q(self, state, action):
        # why can it not recognize that these are ints? :'(
        return self.q[int(state[0]), int(state[1]), int(state[2]), int(state[3]), int(state[4]), int(action)]

    def set_q(self, state, action, value):
        self.q[int(state[0]), int(state[1]), int(state[2]), int(state[3]), int(state[4]), int(action)] = value


def main():
    print("\tRobby the Robot\n\n")

    robby = Robby()

    # print("Initial World:")
    # print(robby.world)
    # print("Robby's location:")
    # print(robby.col, robby.row)

    # Run the N episodes of learning, and plot the total sum of rewards per episode (plotting a point
    # every 100 episodes). This plot—let’s call it the Training Reward plot—indicates the extent to
    # which Robby is learning to improve his cumulative reward.
    for e in range(EPISODES):
        r = robby.episode()
        print("Episode", e, "reward:", r)  # r = robby.episode()

    x_values = []
    y_values = []

    for i in range(int(len(robby.reward) / 100)):
        x_value = i * 100
        y_value = 0

        for j in range(100):
            y_value += robby.reward[i*100 + j]
        y_value /= 100
        x_values.append(x_value)
        y_values.append(y_value)

    plt.plot(x_values, y_values)
    plt.xlim([100, EPISODES])
    plt.show()

    # After training is completed, run N test episodes using your trained Q-matrix, but with  = 0.1 for
    # all N episodes. Again, regenerate a grid of randomly placed cans at the beginning of each episode
    # and also place Robby in a random grid location. Calculate the average over sum-of-rewards-per-
    # episode, and the standard deviation. For simplicity in this writeup, let’s call these values Test-
    # Average and Test-Standard-Deviation. These values indicate how a trained agent performs this
    # task in new environments.

    #
    # this prints them all at the end. currently implementing live printing for sense of progress
    # for rw in range(len(robby.reward)):
    #    print("Episode", rw, "reward:", robby.reward[rw])

    # print("Lets look at the q matrix:\n", robby.q)


if __name__ == '__main__':
    main()
