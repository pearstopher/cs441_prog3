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
from enum import Enum

# CONSTANTS & CONFIGURATION

SIZE = 10  # Robby the Robot lives in a 10 x 10 grid, surrounded by a wall
CHANCE = 0.5  # each grid square has a probability of 0.5 to contain a can

REWARD_CAN = 10  # Robby receives a reward of 10 for each can he picks up
REWARD_CRASH = -5  # a “reward” of −5 if he crashes into a wall
REWARD_NO_CAN = -1  # and a reward of −1 if he tries to pick up a can in an empty square.

# enumeration
class State(Enum):
    EMPTY = 0
    CAN = 1
    WALL = 2


def generate_world():
    # the walls are also going to be represented by grid squares
    size = SIZE + 2
    world = np.zeros((size, size))

    # loop and configure the grid
    for i in range(world):
        for j in range(world):
            # set up the walls
            if i == 0 or i == size or \
                    j == 0 or j == size:
                world[i][j] = State.WALL
            # randomly place the cans
            else:
                if random.uniform(0, 1) <= CHANCE:
                    world[i][j] = State.CAN
    return world


class Robby:
    def __init__(self, world_table):
        # The initial state of the grid in each episode is a random placement of cans
        self.table = world_table

        # Robby is initially placed in a random grid square
        self.row = random.randrange(0, SIZE)
        self.col = random.randrange(0, SIZE)

    # Robby has five “sensors”: Current, North, South, East, and West. At any time step, these each
    # return the “value” of the respective location, where the possible values are Empty, Can, and Wall.
    def


    # Robby has five possible actions: Move-North, Move-South, Move-East, Move-West, and Pick-Up-Can.
    # Note: if Robby picks up a can, the can is then gone from the grid.

    def up(self):
        if self.row == 0:
            return False
        self.row -= 1
        self.score -= self.PUNISHMENT
        return True

    def down(self):
        if self.row == 2:
            return False
        self.row += 1
        self.score -= self.PUNISHMENT
        return True

    def left(self):
        if self.col == 0:
            return False
        self.col -= 1
        self.score -= self.PUNISHMENT
        return True

    def right(self):
        if self.col == 2:
            return False
        self.col += 1
        self.score -= self.PUNISHMENT
        return True

    def clean(self):
        if self.agent == 3 or self.agent == 4:
            # introduce 25% chance of sucking failure
            if random.randint(1, 4) == 1:
                if self.table[self.row, self.col] == 0:
                    self.dirt += 1  # adds new dirt
                self.table[self.row, self.col] = 1
                # no punishment since no movement
                # but no reward either
                return False

        if self.table[self.row, self.col] == 1:
            self.found += 1  # square could be clean
        self.table[self.row, self.col] = 0
        if self.rewards_allowed > 0:
            self.score += self.REWARD
            self.rewards_allowed -= 1
            # limit # rewards to # original dirt piles
        return True

    def dirty(self):
        # check if current square is dirty
        found_dirt = False
        if self.table[self.row, self.col] == 1:
            found_dirt = True

        if self.agent == 3 or self.agent == 4:
            # introduce 10% chance of wrong result
            if random.randint(1, 10) == 1:
                found_dirt = not found_dirt

        return found_dirt


class Agent:
    def __init__(self, world_table, dirt_piles, agent_id):
        self.table = world_table
        self.dirt = dirt_piles
        self.agent = agent_id

        self.turns = 0
        self.found = 0
        self.score = 0
        # REWARD could be some function of dirt based on the theory that more dirt = easier to find dirt
        # ideally I want scores to be similar regardless of number of dirt piles
        # self.REWARD = 10 / self.dirt ** 0.5
        # self.REWARD = 9 - self.dirt  # number of squares w/o dirt
        self.REWARD = 12  # enough for a reasonable expectation of positive results
        self.PUNISHMENT = 1
        self.rewards_allowed = self.dirt  # don't want to be rewarded for cleaning up additional messes

        # Randomize the starting location of the robot,
        self.row = random.randrange(0, 3)
        self.col = random.randrange(0, 3)

    # The agent can go up, down, left right, suck or do nothing.
    def up(self):
        if self.row == 0:
            return False
        self.row -= 1
        self.score -= self.PUNISHMENT
        return True

    def down(self):
        if self.row == 2:
            return False
        self.row += 1
        self.score -= self.PUNISHMENT
        return True

    def left(self):
        if self.col == 0:
            return False
        self.col -= 1
        self.score -= self.PUNISHMENT
        return True

    def right(self):
        if self.col == 2:
            return False
        self.col += 1
        self.score -= self.PUNISHMENT
        return True

    def clean(self):
        if self.agent == 3 or self.agent == 4:
            # introduce 25% chance of sucking failure
            if random.randint(1, 4) == 1:
                if self.table[self.row, self.col] == 0:
                    self.dirt += 1  # adds new dirt
                self.table[self.row, self.col] = 1
                # no punishment since no movement
                # but no reward either
                return False

        if self.table[self.row, self.col] == 1:
            self.found += 1  # square could be clean
        self.table[self.row, self.col] = 0
        if self.rewards_allowed > 0:
            self.score += self.REWARD
            self.rewards_allowed -= 1
            # limit # rewards to # original dirt piles
        return True

    def dirty(self):
        # check if current square is dirty
        found_dirt = False
        if self.table[self.row, self.col] == 1:
            found_dirt = True

        if self.agent == 3 or self.agent == 4:
            # introduce 10% chance of wrong result
            if random.randint(1, 10) == 1:
                found_dirt = not found_dirt

        return found_dirt

    def nothing(self):
        return

    # define a “rule table” based on the location/dirt present – just as in the book
    # in this case you can just make up a rule for where the robot moves next, according to its current location
    def act_rules(self):

        # RULE TABLE

        # dirty square
        if self.dirty():
            self.clean()

        # navigation rules
        elif self.row == 0 and self.col < 2:
            self.right()
        elif (self.col == 2 and self.row < 2) \
                or (self.row == 1 and self.col == 1):
            self.down()
        elif self.row == 2 and self.col > 0:
            self.left()
        elif self.row == 2 and self.col == 0:
            self.up()
        elif self.row == 1 and self.col == 0:
            rand = random.randint(0, 1)
            if rand == 1:
                self.right()  # hit the middle
            else:
                self.up()  # back to beginning

        # print ("row ", self.row, " col ", self.col)
        # 3x3 grid is tough to navigate with no memory
        # I am just going in a circle
        # 1 2 3
        # 8 ? 4
        # 7 6 5
        # with a random element that determines whether I
        # go to the middle or start over
        # otherwise I either miss a square or am forced to loop
        self.turns += 1

    def act_random(self):

        # dirty square
        if self.dirty():
            self.clean()

        # navigation rules
        # try random directions until one works
        else:
            sm = False  # successful_move
            while not sm:
                move = random.randint(1, 4)

                if move == 1:
                    sm = self.up()
                elif move == 2:
                    sm = self.down()
                elif move == 3:
                    sm = self.left()
                elif move == 4:
                    sm = self.right()

        self.turns += 1

    def run(self):
        for i in range(100):
            if self.agent == 1 or self.agent == 2:
                self.act_rules()
            else:
                self.act_random()

            if self.dirt == self.found:
                break

    def info(self, silent=True):
        if not silent:
            print("turns ", self.turns)
            print("score ", self.score)
            print("dirt  ", self.dirt)
            print("found ", self.found)

        return self.turns, self.score, self.dirt, self.found


if __name__ == '__main__':

    # chance of getting a bad sensor read, dropping dirt on a clean square, thinking it's clean and moving on:
    # 1/4 * 1/10 * 1/4 = 1/160
    # I want more than 100 trials to get a good average including rare cases like this
    TRIALS = 10000

    for agent_type in range(1, 5):

        print("Agent ", agent_type, " running...")
        for dirt_amount in (1, 3, 5):
            total_turns = 0
            total_score = 0

            for trial in range(0, TRIALS):
                # print("\tAgent ", agent_type + 1, " trial ", trial)
                table = generate_table(dirt_amount)
                agent = Agent(table, dirt_amount, agent_type)
                agent.run()
                turns, score, dirt, found = agent.info()  # save vars for use
                # if dirt != found: failure

                # calculate trial totals
                total_turns += turns
                total_score += score

            print("\tDirt: ", dirt_amount, "\tTrials: ", TRIALS,
                  "\tAvg Turns: ", '{:.3f}'.format(total_turns / TRIALS).zfill(6),
                  "\tAvg Score: ", '{:.3f}'.format(total_score / TRIALS).zfill(6))
        print()
