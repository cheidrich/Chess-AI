import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from environment import ChessEnv
from model import ChessNet
import config


def train():
    """
    Trains the ChessNet model using the ChessEnv environment.

    Implements a simplified Deep Q-Learning loop:
    - Uses epsilon-greedy action selection.
    - Evaluates legal moves via state-action encoding.
    - Optimizes the Q-value prediction via MSE loss.
    - Periodically updates target network for stability.
    """
    env = ChessEnv()
    policy_net = ChessNet()
    target_net = ChessNet()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config.LR)
    criterion = nn.MSELoss()

    for episode in range(config.NUM_EPISODES):
        print(f"Episode: {episode}")
        state = env.reset()
        done = False

        while not done:
            # epsilon-greedy action selection: choose a random legal move with probability epsilon
            if np.random.rand() < config.EPSILON:
                action = np.random.choice(list(env.board.legal_moves))
            else:
                with torch.no_grad():
                    legal_moves = list(env.board.legal_moves)
                    # Predict Q-values for all legal moves and select the best one
                    action_values = [
                        policy_net(
                            torch.cat((env.get_state_for_move_action(m, env.board), state), dim=0)
                            .unsqueeze(0).float()
                        ).item()
                        for m in legal_moves
                    ]
                    action = legal_moves[np.argmax(action_values)]

            action_tensor = env.get_state_for_move_action(action, env.board)
            next_state, reward, done, _ = env.step(action.uci())
            target = torch.tensor(reward, dtype=torch.float32)

            q_value = policy_net(torch.cat((action_tensor, state), dim=0).unsqueeze(0).float())
            loss = criterion(q_value, target.unsqueeze(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

        if episode % config.TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

if __name__ == "__main__":
    train()
