import numpy as np
import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

class PolicyValueNet():
	def __init__(self, board_width, board_height,
				 steps_considered=4, model_file=None):
		self.board_width = board_width
		self.board_height = board_height
		self.steps_considered = steps_considered

		tf.reset_default_graph()  # I add this to solve the restoring bug

		## Network
		self.input_states = tf.placeholder(
			tf.float32, shape=[None, steps_considered, board_height, board_width]
		)
		self._input_states = tf.transpose(self.input_states, [0, 2, 3, 1])

		# Mutual layers
		self.conv1 = tf.layers.conv2d(inputs=self._input_states,
									  filters=32, kernel_size=[3, 3],
									  padding="same", data_format="channels_last",
									  activation=tf.nn.relu)
		self.conv2 = tf.layers.conv2d(inputs=self.conv1, filters=64,
									  kernel_size=[3, 3], padding="same",
									  activation=tf.nn.relu)
		self.conv3 = tf.layers.conv2d(inputs=self.conv2, filters=128,
									 kernel_size=[3, 3], padding="same",
									 activation=tf.nn.relu)

		# "Action Way"
		self.action_conv = tf.layers.conv2d(inputs=self.conv3, filters=4,
											kernel_size=[1, 1], padding="same",
											activation=tf.nn.relu)
		self.action_conv_flat = tf.reshape(
			self.action_conv, [-1, 4 * board_height * board_width]
		)
		self.action_fc = tf.layers.dense(inputs=self.action_conv_flat,
										 units= board_height * board_width,
										 activation=tf.nn.softmax
		)

		#"Evaluation Path"
		self.eval_conv = tf.layers.conv2d(inputs=self.conv3, filters=2,
										  kernel_size=[1, 1], padding="same",
										  activation=tf.nn.relu)
		self.eval_conv_flat = tf.reshape(
			self.eval_conv, [-1, 2 * board_height * board_width]
		)
		self.eval_fc1 = tf.layers.dense(inputs=self.eval_conv_flat,
										units=64, activation=tf.nn.relu
		)
		self.eval_fc2 = tf.layers.dense(inputs=self.eval_fc1,
										units=1, activation=tf.nn.tanh
		)

		# Loss
		self.mcts_probs = tf.placeholder(
			tf.float32, shape=[None, board_height * board_width]
		)
		self.policy_loss = -tf.reduce_mean(tf.reduce_sum(
			tf.multiply(self.mcts_probs, tf.log(self.action_fc)), 1
		))

		self.labels = tf.placeholder(tf.float32, shape=[None, 1])
		self.value_loss = tf.losses.mean_squared_error(self.labels,
													   self.eval_fc2)

		self.l2_rate = 1e-4
		self.vars = tf.trainable_variables()
		self.l2 = self.l2_rate * tf.add_n(
			[tf.nn.l2_loss(v) for v in self.vars if 'bias' not in v.name.lower()])#?
		self.loss = self.value_loss + self.policy_loss + self.l2

		# Optimizer
		self.lr = tf.placeholder(tf.float32)
		self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr
		).minimize(self.loss)

		# Session
		self.session = tf. Session()

		# Init
		init = tf.global_variables_initializer()
		self.session.run(init)

		# Saving Restoring
		self.saver = tf.train.Saver()
		if model_file is not None:
			self.restore_model(model_file)

	def policy_value(self, current_state):
		act_probs, value = self.session.run([self.action_fc, self.eval_fc2],
									feed_dict={self.input_states: current_state}
		)
		return act_probs, value

	def policy_value_fn(self, board): # the policy_value_fn takes game.board as input
		legal_positions = list(board.availables) # board.availables is a set
		current_state = np.ascontiguousarray( # send game.board.current_state() to the network input
			board._get_current_state().reshape(
				-1, self.steps_considered, self.board_width, self.board_height)
		)
		act_probs, value = self.policy_value(current_state)
		act_probs = zip(legal_positions, act_probs[0][legal_positions])
		return act_probs, value

	def train_step(self, state_batch, mcts_probs, winner_batch, lr):
		winner_batch = np.reshape(winner_batch, (-1, 1))
		loss = self.session.run(
			[self.loss, self.optimizer],
			feed_dict={self.input_states: state_batch,
					   self.mcts_probs: mcts_probs,
					   self.labels: winner_batch,
					   self.lr: lr}
		)
		return loss

	def save_model(self, model_path):
		self.saver.save(self.session, model_path)

	def restore_model(self, model_path):
		self.saver.restore(self.session, model_path)