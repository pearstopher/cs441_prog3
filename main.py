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


# CONSTANTS & CONFIGURATION
SIZE = 10  # Robby the Robot lives in a 10 x 10 grid, surrounded by a wall
CHANCE = 0.5  # each grid square has a probability of 0.5 to contain a can


# enumeration of reward values
class Reward(IntEnum):
    CAN = 10  # Robby receives a reward of 10 for each can he picks up
    CRASH = -5  # a “reward” of −5 if he crashes into a wall
    NO_CAN = -1  # and a reward of −1 if he tries to pick up a can in an empty square.


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
        self.row = random.randrange(0, SIZE)
        self.col = random.randrange(0, SIZE)

        # Keep track of the total reward gained per episode.
        self.reward = 0

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

    # Robby has five “sensors”: Current, North, South, East, and West. At any time step, these each
    # return the “value” of the respective location, where the possible values are Empty, Can, and Wall.
    def current(self):
        return self.world[self.col][self.row]

    def north(self):
        return self.world[self.col][self.row - 1]

    def south(self):
        return self.world[self.col][self.row + 1]

    def east(self):
        return self.world[self.col + 1][self.row]

    def west(self):
        return self.world[self.col - 1][self.row]

    # Robby has five possible actions: Move-North, Move-South, Move-East, Move-West, and Pick-Up-Can.
    # Note: if Robby picks up a can, the can is then gone from the grid.
    def move_north(self):
        if self.north() == State.WALL:
            self.reward += Reward.CRASH
        else:
            self.row += 1

    def move_south(self):
        if self.south() == State.WALL:
            self.reward += Reward.CRASH
        else:
            self.row -= 1

    def move_east(self):
        if self.east() == State.WALL:
            self.reward += Reward.CRASH
        else:
            self.col += 1

    def move_west(self):
        if self.south() == State.WALL:
            self.reward += Reward.CRASH
        else:
            self.col -= 1

    def pick_up_can(self):
        if self.current() == State.CAN:
            self.world[self.col][self.row] = State.EMPTY
            self.reward += Reward.CAN
        else:
            self.reward += Reward.NO_CAN


if __name__ == '__main__':
    print("\tRobby the Robot\n\n")

    robby = Robby()
