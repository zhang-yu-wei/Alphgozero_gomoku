import numpy as np
import os
import copy
os.environ["CUDA_VISIBLE_DEVICES"]=""


class Board(object):  # maybe we'll need to inherit Array
	def __init__(self, width, height, n_in_row):
		self.height = height
		self.width = width
		self.n_in_row = n_in_row
		self.states = {}  # {int: int} i.e. {move: player} no chesspieces means it doesn't exist in this dict
		self.players = [1, 2]
		self.states_history = [] # a list of self.states in the past
		self.states_history_2d = [] # x*height*width dimension

	def init_board(self, start_player=np.random.randint(0, 2)):
		self.current_player = self.players[start_player]
		self.availables = set(range(self.height*self.width))  # locations that are blank, not considering restriction from rules
		self.availables_history = set()
		self.last_move = -1
		self.states = {}
		self.states_history = []
		self.states_history_2d = []

	def move_to_location(self, move):
		"""
		4*3 board's moves like:
		 0 1 2
		 3 4 5
		 6 7 8
		 9 10 11
		and move 5's location is [1,2]
		"""
		h = move // self.width
		w = move % self.width
		return [h, w]

	def location_to_move(self, location):
		h = location[0]
		w = location[1]
		move = h * self.width + w
		return move

	def add_history(self):
		"""add the current locations of all chesspieces the current player put
		   shape: width*height
		   get something like:
		   [[1., 0., 0.]
			[0., 1., 0.]
			[0., 0., 0.]]
			where 1. indicates the loc of current player's pieces
		"""
		# add states to states_history
		self.states_history.append(self.states)

		# add states to states_history_2d
		state = np.zeros((4, self.width, self.height))
		if self.states:
			moves, players = np.array(list(zip(*self.states.items())))
			move_curr = moves[players == self.current_player]
			move_oppo = moves[players != self.current_player]
			state[0][move_curr // self.width,
							move_curr % self.height] = 1.0
			state[1][move_oppo // self.width,
							move_oppo % self.height] = 1.0
			# indicate the last move location
			state[2][self.last_move // self.width,
							self.last_move % self.height] = 1.0
		if len(self.states) % 2 != 0:
			state[3][:, :] = 1.0  # indicate the colour to play
		self.states_history_2d.append(state)

	def do_move(self, move):
		#print('do_move')
		# Check if coordinates are occupied(this is for human player's case)
		if move not in self.availables:
			raise Exception("Move unavailable.")

		self.states[move] = self.current_player

		# Store history for the opposite player's use
		self.add_history()
		self.last_move = move
		self.availables_history = copy.deepcopy(self.availables)

		# change player
		self.current_player = (
			self.players[1] if self.current_player == self.players[0]
			else self.players[0])

		self.availables.discard(move)

	def if_equal(self, colors):
		return 1 if len(set(colors))<=1 else 0

	def find_list(self, start, middle, between):
		h_dist = (middle[0] - start[0]) / between
		w_dist = (middle[1] - start[1]) / between

		list = []
		for i in range(self.n_in_row*2-1):
			list.append(self._get_color(start[0] + i*h_dist, start[1] + i*w_dist))
		return list

	def get_windows(self, color_list):
		windows = []
		for i in range(self.n_in_row):
			windows.append(color_list[i:i+self.n_in_row])
		return windows

	def check_if_end(self, move, is_shown):
		winner = -1
		if move == -1:
			return False, winner
		else:
			h, w = self.move_to_location(move)
			between = self.n_in_row - 1
			starts = [
				(h - between, w - between),
				(h - between, w),
				(h - between, w + between),
				(h, w - between)
			]

			last_player = (self.players[1] if self.current_player == self.players[0]
						   else self.players[0])

			for start in starts:
				color_list = self.find_list(start, (h, w), between)
				windows = self.get_windows(color_list)
				is_equal = [self.if_equal(window) for window in windows]
				if sum(is_equal) > 0:
					winner = last_player
					break
			#print(winner)

			if winner == -1:
				if len(self.availables) == 0:
					if is_shown:
						print("Game end. Tie")
					return True, winner
				else:
					return False, winner
			else:
				if is_shown:
					print("Game end. The winner is", winner)
				return True, winner

	def get_current_player(self):
		return self.current_player

	def _within_boundary(self, h, w):
		if h >= 0 and h < self.height and w >= 0 and w < self.width:
			return True
		else:
			return False

	def _get_color(self, h, w):
		"""
		Same thing as Array.__getitem__, but returns None if coordinates are
		not within array dimensions.
		"""
		move = self.location_to_move([h, w])
		if not self._within_boundary(h, w):
			return False
		elif not move in self.states:
			return -1
		else:
			return self.states[move]

	def _get_current_state(self):
		if len(self.states_history_2d) == 0:
			return np.zeros((4, self.height, self.width))
		else:
			return self.states_history_2d[-1]


class Game(object):
	"""Let board become active!"""

	def __init__(self, board):
		self.board = board

	def graphic(self, board, player1, player2):
		"""
		Draw the board and show game info
		player1, player2 are both ints, i.e. 1 or 2
		"""
		width = board.width
		height = board.height

		print("Player %s with +" % player1)
		print("Player %s with -" % player2)

		print(' ', end=' ')
		for x in range(width):
			print("{0:2}".format(x), end=' ')
		print('')

		for i in range(height):
			print("{0:<2d}".format(i), end=' ')
			for j in range(width):
				loc = i * width + j
				p = board.states.get(loc, -1)  # ?
				if p == player1:
					print('+'.center(2), end=' ')
				elif p == player2:
					print('-'.center(2), end=' ')
				else:
					print(' '.center(2), end=' ')
			print('')

	def start_play(self, player1, player2, start_player, is_shown=1): # what is  inside the player1?
		"""start a game between two human players"""
		self.board.init_board(start_player=start_player)

		# player settings
		p1, p2 = self.board.players  # remember board.players is only a list
		player1.set_player_ind(p1)   # player1 and player2 are objects
		player2.set_player_ind(p2)
		players = {p1: player1, p2: player2}

		# running the game
		if is_shown:
			current_player = self.board.get_current_player()
			player_in_turn = players[current_player]
			print('current player: ' + str(player_in_turn))
			self.graphic(self.board, player1.player, player2.player)  # player1.player is an int, i.e. 1 or 2
		while(True):
			current_player = self.board.get_current_player()
			player_in_turn = players[current_player]
			move = player_in_turn.get_action(self.board)
			self.board.do_move(move)
			if is_shown:
				print('current player: ' + str(player_in_turn))
				self.graphic(self.board, player1.player, player2.player)
			end, winner = self.board.check_if_end(move, is_shown)
			if end:
				if is_shown:
					if winner != -1:
						print("Game end. Winner is", players[winner])
					else:
						print("Game end. Tie")
				return winner

	def start_self_play(self, player, is_shown=0, temp=1e-3):
		self.board.init_board()
		#print('new game')
		p1, p2 = self.board.players
		states, mcts_probs, current_players = [], [], []
		while True:
			move, move_probs = player.get_action(self.board,  # move: an int
												 temp=temp,
												 return_prob=1)
			# store the data
			states.append(self.board._get_current_state())
			mcts_probs.append(move_probs)
			current_players.append(self.board.current_player)
			# perform a move
			self.board.do_move(move)
			if is_shown:
				self.graphic(self.board, p1, p2)
			end, winner = self.board.check_if_end(move, is_shown)

			if end:
				winner_z = np.zeros(len(current_players))
				if winner != -1:
					winner_z[np.array(current_players) == winner] = 1.0
					winner_z[np.array(current_players) != winner] = -1.0
				player.reset_player()
				if is_shown:
					if winner != -1:
						print("Game end. Winner is player:", winner)
					else:
						print("Game end. Tie")
				return winner, zip(states, mcts_probs, winner_z)

