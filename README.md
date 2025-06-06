# Chess AI - Deep Q-Reinforcement-Learning

## Overview

This repository contains a simple chess AI that uses reinforcement learning to decide on moves during a game. The code sets up a custom chess environment using the `chess` and `gym` libraries, handling board setup, move validation, and reward calculation (e.g., rewarding captures). A basic convolutional neural network (ChessNet) processes the board state and move information to predict how good a move is. The training loop uses an epsilon-greedy approach to explore legal moves and updates the network using mean squared error q-loss and reinforcement learning (inspired by DeepQ-Learning) with the Adam optimizer.

The Image bellow shows one of the game states that was played.

![Chess AI Game](eda-notebooks/chessai-game.PNG)


## Results

The results can be seen in the `reinforcement_learning` notebook right at the top of the project repository. It shows some of the game-states that the engine played and also plots the loss and q-values.

## Reconchess AI

In another project, I built an AI for Reconchess that learns which square on the board to focus on. This model is trained using sensor data from an accompanying CSV file, which provides information on board conditions. The goal is to use this data to predict the most promising square to inspect, making it easier to gather the most important information during a game.
