<img src="https://encrypted-tbn0.gstatic.com/images?q=tbn%3AANd9GcQnpeknkdIauzjtJT-DP8Ea7pz_7IgMoCGooLtNIz_3LqvN6rdT&usqp=CAU" alt="Front" width="100%" height="100%" align="middle">

# Gomoku Player
This is an implementation of Gomoku game player using Reinforcement learning.

## Methodology
Unlike AlphaGo Fan and AlphaGo Lee which train the networks using human players’ data and then guide the moves
through the process of MCTS, AlphaGo Zero uses MCTS to directly generate gaming process. The steps in a training
pipeline are outlined below: 1. We extract data from the process which indicates state (moves on the board), states
history (usually a few previous steps), the current player, and the final winner of the whole game; 2. We feed the
network with a set of historical states in 2D, and then we expect to get the state value and the probability vector of the
next move from the network; 3. We use the output from the network to guide MCTS moving; 4. We train the network
with a batch of data which contains the recent self-play result through the game.

<img src="https://github.com/zhang-yu-wei/Alphgozero_gomoku/blob/master/imgs/net.png" align="center" width="100%" height="100%">

The network is relatively simple - it starts with four convolutional layers, and then fans out into two fully connected
layers generating value and policy, respectively. As the state space of interest here is relatively small, there is no need to
include a residual block as AlphaGo Zero does.

MCTS player is implemented based on the conventional approach. It traverses the tree first and greedily select a proper
leaf node. If this node is not the end of the game, then we randomly choose an available place on the board, which is
made the children of that node (move). Note that this node uses the policy from the current network, i.e., the MCTS and
the neural network cooperate with each other. MCTS will search a reasonable place for next move and neural network
helps save the information for MCTS from the previous experience.

## Experiment
A Go-like game usually has a square board, where two players take turns to place his/her playing piece (“stone”) on
the vacant intersections (“points”) of the board. Different players will have different colors and every stone has equal
values where you have to learn the strategy of distributing your units. Despite its relatively simple rules, Go is very
complex. Compared to chess, Go has both a larger board with more scope for play and longer games, and, on average,
many more alternatives to consider per move. The lower bound on the number of legal board positions in Go has been
estimated to be 2\*10^170, making it essential to use MCTS to help reduce the search space.

Given our goal to observe how AlphaGo Zero functions and to evaluate its performance, there is no need to test on a
full-scale 19\*19 board. As such, we used a smaller 8\*8 board to run our tests. Considering that Go is too abstruse
for us to analyze the underlying mechanism of the algorithm, we choose to run our tests on the Gomoku first. The
remainder of this report will illustrate the learning process derived from the tests.

The training process is somewhat challenging as the dataset is not given to the network upfront. As such, we have to
run the program in the self-play mode for several steps to generate an initial dataset before starting the optimization.
We iterate for several steps so as to update the model parameters, and in every step, we take data stochastically so as to
give the network a full coverage over the dataset. Note that one needs to avoid excessive iterations which may make the
model overfit the data.

After several rounds, we can evaluate the performance of the network. The current policy network will compete with
the best historical policy network derived from the games played before. If the current policy network can beat the old
one over several games (i.e., the number games the current network wins surpasses the number it loses), we will update
the best policy to the current one.

The training parameters used here are outlined below: batch size is 512; in every optimizing stage, we train the model for five times; loss function is the sum of cross entropy for policy output and mean square error for value output;learning rate is initially set to 2e-3 and is adjusted subsequently according to the probability distribution of the policy;also, the algorithm traverse the tree for 400 times within each search so as to update the action probability.

## Results

We started with playing on a 6\*6 board with the four-stones-in-line strategy. Times to traverse tree search is set up
to 400. After training the model with 5000 games, we are delighted to find that our network has gained the ability
to defend when two stones are lined up, indicating that it has developed an understanding in the game rule. More
importantly, AlphaGo can perform some strategies (i.e. to subjectively plan efficient moves a few steps ahead).

Since playing 4-in-line game on a 6\*6 board seems quite easy for AlphaGo Zero, we assigned it a more complex task,
i.e., playing 5-in-line game on a 8\*8 board. Times to traverse tree search is set up to 600. We find it interesting to
observe the process through learning. At first, game episodes could be around 50 playing, indicating that it has not
acquired the concept of placing stones together. It seems to get the sense of preventing the opponent from getting 5
stones in line after around 1000 games. It is also worth noticing that AlphaGo Zero can play with a good start strategy
after around 8000 games.

We stopped at 10,000 games and tested the ability of the model. We discovered AlphaGo Zero can sense the danger
from the pattern of the current state and defend when there are 3 stones lined up, and it is also found that the program
can search around the board to see if there is a chance to offend the opponent. However, our network seems not able to
use strategies on 5-in-line game.

<strong>Here is a game result played by our model and human (sadly just myself)</strong>

<img src="https://github.com/zhang-yu-wei/Alphgozero_gomoku/blob/master/imgs/6.png" align="center">

## References
[1] Silver D, Schrittwieser J, Simonyan K, et al. Mastering the game of Go without human knowledge[J]. Nature,
2017, 550(7676):354-359.

[2] Silver D, Hubert T, Schrittwieser J, et al. Mastering Chess and Shogi by Self-Play with a General Reinforcement
Learning Algorithm[J]. 2017.

[3] Sutton R S, Barto A G. Reinforcement Learning: An Introduction[J]. IEEE Transactions on Neural Networks,
1998, 9(5):1054-1054.
