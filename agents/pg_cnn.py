import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PG_CNN(nn.Module):
    def __init__(
        self,
        env,
        hidden_dim=64,
        layer_type='softmax',
        distance='L2',
        lr=1e-4,
        gamma=0.99,
        harmonic_exponent=4,
        weight_reg=0.01,
        n_filters1=16,
        n_filters2=32
    ):
        """
        CNN-based Policy Gradient
        Args:
            env: Gym-like environment (with a 2D observation space)
            hidden_dim: dimension of the final CNN embedding
            layer_type: 'softmax' or 'harmonic'
            distance: 'L2' or 'squared' for harmonic
            lr: learning rate
            gamma: discount factor
            harmonic_exponent: exponent for harmonic distance
            weight_reg: L2 regularization on the action embeddings (if harmonic)
            n_filters1, n_filters2: number of filters in the 2 conv layers
        """
        super(PG_CNN, self).__init__()

        self.env = env
        # We assume env.observation_space.shape = (H, W) or (channels, H, W)
        obs_shape = env.observation_space.shape
        if len(obs_shape) == 2:
            # e.g. (H, W)
            self.obs_channels = 1
            self.obs_height = obs_shape[0]
            self.obs_width = obs_shape[1]
        elif len(obs_shape) == 3:
            # e.g. (C, H, W)
            self.obs_channels = obs_shape[0]
            self.obs_height = obs_shape[1]
            self.obs_width = obs_shape[2]
        else:
            raise ValueError(
                f"Unexpected observation shape {obs_shape}. "
                "Expected 2D or 3D (channels, H, W)."
            )

        self.action_dim = env.action_space.n
        self.layer_type = layer_type
        self.distance = distance
        self.gamma = gamma
        self.lr = lr
        self.harmonic_exponent = harmonic_exponent
        self.weight_reg = weight_reg

        # Build a small CNN feature extractor
        # Input: [batch_size, obs_channels, obs_height, obs_width]
        # Output: [batch_size, hidden_dim]
        self.cnn = nn.Sequential(
            nn.Conv2d(self.obs_channels, n_filters1, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_filters1, n_filters2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(n_filters2 * self.obs_height * self.obs_width, hidden_dim),
            nn.ReLU()
        )

        # Final layer
        if layer_type == 'softmax':
            self.output_layer = nn.Linear(hidden_dim, self.action_dim, bias=False)
        else:
            # Harmonic: store per-action embeddings
            # shape: [action_dim, hidden_dim]
            self.weights = nn.Parameter(torch.randn(self.action_dim, hidden_dim))

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        """
        x shape: [batch_size, H, W] or [batch_size, C, H, W]
        We assume we pass in a batch dimension.
        returns: probabilities over actions [batch_size, num_actions]
        """
        # If x is [batch_size, H, W], add channel dim
        if x.dim() == 3:  # [batch, H, W]
            x = x.unsqueeze(1)  # [batch, 1, H, W]

        features = self.cnn(x)  # shape [batch_size, hidden_dim]

        if self.layer_type == 'softmax':
            logits = self.output_layer(features)
            return torch.softmax(logits, dim=1)
        else:
            # Harmonic approach
            # (batch_size, 1, hidden_dim) - (1, action_dim, hidden_dim)
            diff = features[:, None, :] - self.weights[None, :, :]

            # squared L2
            dist_sq = (diff ** 2).sum(dim=-1)  # shape [batch_size, action_dim]
            if self.distance == 'L2':
                dist = torch.sqrt(dist_sq + 1e-8)
            else:
                dist = dist_sq

            dist_exp = dist ** self.harmonic_exponent
            inv_dist = 1.0 / (dist_exp + 1e-8)
            probs = inv_dist / inv_dist.sum(dim=1, keepdim=True)
            return probs

    def predict(self, state, deterministic=False):
        """
        Single-step action selection
        state: shape [H, W] or [C, H, W] (no batch dimension)
        """
        with torch.no_grad():
            # Convert to [1, H, W] or [1, C, H, W]
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            action_probs = self.forward(state_tensor)  # [1, num_actions]
            if deterministic:
                return torch.argmax(action_probs, dim=1).item()
            else:
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                return action.item()

    def store_outcome(self, log_prob, reward):
        self.saved_log_probs.append(log_prob)
        self.rewards.append(reward)

    def learn(self, total_timesteps=10000, callback=None):
        """
        Collect rollouts until total_timesteps reached,
        then do policy update after each episode.
        """
        timestep = 0
        while timestep < total_timesteps:
            state = self.env.reset()
            self.saved_log_probs = []
            self.rewards = []
            done = False

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                # forward pass
                action_probs = self.forward(state_tensor)

                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                next_state, reward, done, info = self.env.step(action.item())

                self.store_outcome(log_prob, reward)
                state = next_state
                timestep += 1

                if callback is not None:
                    callback._on_step(reward)
                    if callback.early_stop:
                        return

                if timestep >= total_timesteps:
                    break

            self._update_policy()

    def _update_policy(self):
        """
        Standard REINFORCE: compute discounted returns, standardize,
        multiply by -log_prob, sum, and backprop.
        """
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        policy_loss = []
        for log_prob, ret in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * ret)
        policy_loss = torch.cat(policy_loss).sum()

        # If harmonic, add L2 penalty on self.weights
        if self.layer_type == 'harmonic':
            reg_loss = self.weight_reg * torch.norm(self.weights, p=2)
            policy_loss += reg_loss

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        self.saved_log_probs = []
        self.rewards = []

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
