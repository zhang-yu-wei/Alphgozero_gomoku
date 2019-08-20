import os
from Gomoku import Board, Game
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"]=""


class Human(object):
    """human player"""

    def __init__(self):
        self.player = None

    def set_player_ind(self, p):
        self.player = p

    def get_action(self, board):
        """
        the acceptable input form:
        """

        x1 = int(input("Your move(line):"))
        x2 = int(input("Your move(column):"))
        print('\n')
        move = board.location_to_move([x1, x2])
        return move

    def __str__(self):
        return "Human {}".format(self.player)


def run():
    board = Board(width=6, height=6, n_in_row=4)
    game = Game(board)

    human_player1 = Human()
    human_player2 = Human()

    game.start_play(human_player1, human_player2, np.random.randint(0, 2), is_shown=1)


if __name__ == '__main__':
    run()