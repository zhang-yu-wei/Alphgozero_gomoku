import random
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from Gomoku import Board, Game
#from mcts_pure import MCTSPlayer as MCTS_Pure
from mcts_zero import MCTSPlayer
from net import PolicyValueNet



class TrainPipeline():
	def __init__(self, init_model=None):
		self.board_width = 6
		self.board_height = 6
		self.n_in_row = 4
		self.board = Board(width=self.board_width,
						   height=self.board_height,
						   n_in_row=self.n_in_row
						   )
		self.game = Game(self.board)

		self.learn_rate = 2e-3
		self.lr_multiplier = 1.0
		self.temp = 1.0
		self.n_playout = 400 # how many times we are going to traverse the tree
		self.c_puct = 5
		self.buffer_size = 10000 # the amount of (state, policy, winner) tuples
		self.batch_size = 512  # the dataset that is used in optimizer
		self.data_buffer = deque(maxlen=self.buffer_size)
		self.epochs = 10 # how many times you run sess.run(optimizer)
		self.kl_targ = 0.02
		self.check_freq = 100
		self.game_num = 7500  # number of games for self-play
		self.pure_mcts_playout_num = 1000
		self.steps_considered = 4

		self.loss_history = []

		if init_model:
			self.policy_value_net = PolicyValueNet(self.board_width,
												   self.board_height,
												   self.steps_considered,
												   model_file=init_model)
		else:
			self.policy_value_net = PolicyValueNet(self.board_width,
												   self.board_height,
												   self.steps_considered)
		self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
									  self.steps_considered,
									  c_puct=self.c_puct,
									  n_playout=self.n_playout,
									  is_selfplay=1)
	def get_equi_data(self, play_data):
		extend_data = []
		for state, mcts_prob, winner in play_data:
			#print("state.shape", state.shape)
			for i in [1, 2, 3, 4]:
				# rotate counter clockwise
				equi_state = np.array(np.rot90(state, i, axes=(1, 2)))
				#print("equi_state.shape", equi_state.shape)
				equi_mcts_prob = np.rot90(np.flipud(
						mcts_prob.reshape(self.board_height, self.board_width)
					), i
				)
				extend_data.append((equi_state, np.flipud(equi_mcts_prob).flatten(),
									winner))
		#print("len(extend_data)", len(extend_data))
		#print("extend_data[0][0].shape", extend_data[0][0].shape)
		#print("extend_data[5][1].shape", extend_data[5][1].shape)
		return extend_data

	def collect_selfplay_data(self):
		"collect data for training in one self-play game"
		winner, play_data = self.game.start_self_play(self.mcts_player,
													  temp=self.temp)
		play_data = list(play_data)[:]  # [(state, policy, winner),  ...]  ps: (3-d, 1-d, 1-d)
		self.episode_len = len(play_data)
		# augment the data
		play_data = self.get_equi_data(play_data)
		self.data_buffer.extend(play_data) # [(state, policy, winner),  ...] too

	def network_update(self):
		mini_batch = random.sample(self.data_buffer, self.batch_size)
		state_batch = [data[0] for data in mini_batch]
		mcts_probs_batch = [data[1] for data in mini_batch]
		winner_batch = [data[2] for data in mini_batch]
		old_probs, old_v = self.policy_value_net.policy_value(state_batch)
		for i in range(self.epochs):
			loss = self.policy_value_net.train_step(
				state_batch,
				mcts_probs_batch,
				winner_batch,
				self.learn_rate * self.lr_multiplier)
			# save the data for plotting
			self.loss_history.append(loss)
			# evaluate the policy
			new_probs, new_v = self.policy_value_net.policy_value(state_batch)
			kl = np.mean(np.sum(old_probs*
								(np.log(old_probs+1e-10) - np.log(new_probs+1e-10)),
								axis=1))
			if kl > self.kl_targ * 4: # if the step is too large, we stop and adjust the lr multiplier
				break
			# adaptively adjust the learning rate
			if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
				self.lr_multiplier /= 1.5
			elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
				self.lr_multiplier *= 1.5

			print(("kl:{:.5f},"
					"lr_multiplier:{:.3f},"
					"loss:{},").format(kl,
										self.lr_multiplier,
										loss))
			return loss

	def policy_evaluate(self, n_games=1):
		"Evaluate the trained policy by beating against the past version"
		try:
			old_policy_value_net = PolicyValueNet(self.board_width,
												   self.board_height,
												   self.steps_considered,
												   model_file='./current_policy4.model')
			old_mcts_player = MCTSPlayer(old_policy_value_net.policy_value_fn,
										  self.steps_considered,
										  c_puct=self.c_puct,
										  n_playout=self.n_playout,
										  is_selfplay=0)
		except Exception as e:
			# the first time of model evaluation
			print("Exception:", e)
			win_cnt = np.zeros(2)
			win_cnt[0] = n_games
			return win_cnt

		current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
										 self.steps_considered,
										 c_puct=self.c_puct,
										 n_playout=self.n_playout)

		# [num_player1 win, num_player2 win]
		win_cnt = np.zeros(2)
		for i in range(n_games):
			winner = self.game.start_play(current_mcts_player,
										  old_mcts_player,
										  is_shown=1)
			# winner =1 : current mscts ;  winner = 2 : old mcts; winner = -1 : Tie
			if winner != -1:
				win_cnt[winner - 1] += 1
		print("Evaluation: num_games:{}, win:{}, lose:{}, tie:{}"
			  .format(n_games,
					  win_cnt[0], win_cnt[1], n_games - win_cnt[0] - win_cnt[1]))
		return win_cnt

	def plot_loss(self, loss_history):
		print("plot loss called")
		try:
			font = {'family': 'serif',
					'color': 'black',
					'weight': 'normal',
					'size': 16}
			# Training process
			fig = plt.figure()
			plt.plot(np.arange(len(loss_history)), loss_history, marker='x', linestyle='-', markersize=5, linewidth=0.2, color='r')
			ax = plt.gca()
			ax.set_xlabel('epochs', fontdict=font)
			ax.set_ylabel('loss', fontdict=font)
			plt.title('loss')
			plt.savefig("./training_loss.png")
			plt.close()
		except:
			print("WARNING: loss figure saving error")

	def run(self):
		"run the training pipeline"
		try:
			for i in range(self.game_num):
				# we let it self-play for one episode of game
				#print("collect_selfplay_data(): start one selfplay game.")
				self.collect_selfplay_data() #
				print("game {}:, episode_len:{}"
					  .format(i+1, self.episode_len))
				# and train the network several epochs
				if len(self.data_buffer) > self.batch_size:
					#print("len(self.data_buffer)", len(self.data_buffer))
					#print("self.batch_size", self.batch_size)
					self.network_update()
				# save and plot the loss
				self.plot_loss(self.loss_history)
				# check the current model performance & save it
				if (i+1) % self.check_freq == 0:
					print("the game numbers current model has played: {}".format(i+1))
					if not (i+1) == self.check_freq: # jumps the first time where there's no current.model yet.
						win_cnt = self.policy_evaluate()
						if win_cnt[0] > win_cnt[1]:
							print("NEW BEST POLICY FOUND")
							self.policy_value_net.save_model('./best_policy4.model')
					self.policy_value_net.save_model('./current_policy4.model')

		except KeyboardInterrupt:
			print('\n\rquit')

if __name__ == '__main__':
	training_pipeline = TrainPipeline()
	training_pipeline.run()