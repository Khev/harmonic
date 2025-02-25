import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class A2C(nn.Module):
    """
    A simple synchronous Actor-Critic agent, using an n-step update.
    """
    def __init__(
        self,
        env,
        hidden_dim=64,
        layer_type='softmax',   # 'softmax' or 'harmonic'
        distance_norm='L2',     # if harmonic, 'L1' or 'L2'
        harmonic_exponent=4,
        weight_reg=0.01,
        lr=0.001,
        gamma=0.99,
        value_coef=0.5,
        entropy_coef=0.01,
        rollout_steps=5,
        n_layers=2,
        dropout_prob=0.3
    ):
        super(A2C, self).__init__()

        self.env = env
        self.gamma = gamma
        self.layer_type = layer_type
        self.distance_norm = distance_norm
        self.harmonic_exponent = harmonic_exponent
        self.weight_reg = weight_reg
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.rollout_steps = rollout_steps

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Shared MLP feature extractor
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())  # final ReLU
        self.feature_extractor = nn.Sequential(*layers)

        # Policy head
        if layer_type == 'softmax':
            self.policy_head = nn.Linear(hidden_dim, action_dim, bias=False)
        else:
            # Harmonic layer stores action embeddings (class centers)
            self.weights = nn.Parameter(torch.randn(action_dim, hidden_dim))

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # Rollout storage
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []

    def _forward_features(self, x):
        """
        Pass observation(s) x through the shared feature extractor.
        """
        return self.feature_extractor(x)  # shape [batch_size, hidden_dim]

    def _policy_distribution(self, features):
        """
        Given the extracted features, return action probabilities.
        Supports both softmax and harmonic layers.
        """
        if self.layer_type == 'softmax':
            logits = self.policy_head(features)
            return torch.softmax(logits, dim=1)

        else:
            # Harmonic layer: Compute distance-based probabilities
            diff = features[:, None, :] - self.weights[None, :, :]

            # Select the appropriate distance metric
            if self.distance_norm.upper() == 'L1':
                dist = diff.abs().sum(dim=-1)  # L1 norm
            else:  # Default: L2
                dist_sq = (diff ** 2).sum(dim=-1)
                dist = torch.sqrt(dist_sq + 1e-8)  # L2 norm

            # Normalize by min distance for stable scaling
            min_dist = torch.min(dist, dim=-1, keepdim=True)[0]
            dist = dist / torch.clamp(min_dist, min=1e-6)

            # Raise to harmonic exponent and invert
            dist_exp = dist ** self.harmonic_exponent
            inv_dist = 1.0 / (dist_exp + 1e-8)

            # Normalize to sum to 1
            probs = inv_dist / inv_dist.sum(dim=1, keepdim=True)
            return probs

    def _value_function(self, features):
        """
        Compute the scalar value function V(s).
        """
        return self.value_head(features)

    def forward(self, x):
        """
        Compute policy probabilities and value function for a given state.
        """
        feats = self._forward_features(x)
        probs = self._policy_distribution(feats)
        value = self._value_function(feats)
        return probs, value

    def predict(self, state, deterministic=False):
        """
        Single-step action selection.
        """
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            feats = self._forward_features(state_t)
            probs = self._policy_distribution(feats)
            if deterministic:
                action = probs.argmax(dim=1).item()
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample().item()
        return action

    def store_transition(self, state, action, log_prob, reward, value, done):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)

    def learn(self, total_timesteps=100000, callback=None):
        """
        Train using Advantage Actor-Critic (A2C) with rollouts.
        """
        timestep = 0
        state = self.env.reset()
        while timestep < total_timesteps:
            # Gather rollout
            for _ in range(self.rollout_steps):
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                feats = self._forward_features(state_t)
                probs = self._policy_distribution(feats)
                value = self._value_function(feats).squeeze(1)

                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                next_state, reward, done, info = self.env.step(action.item())
                self.store_transition(state, action.item(), log_prob, reward, value.item(), done)

                state = next_state
                timestep += 1

                if callback is not None:
                    callback._on_step(reward)
                    if callback.early_stop:
                        return

                if done or timestep >= total_timesteps:
                    state = self.env.reset()
                    break

            # A2C Update
            self._update_policy()

    def _update_policy(self):
        """
        Compute A2C updates using n-step returns and advantage estimation.
        """
        # Convert rollouts to tensors
        states_np = np.array(self.states, dtype=np.float32)  # Convert list to NumPy array
        states_t = torch.from_numpy(states_np)  # Then convert to tensor
        actions_t = torch.tensor(self.actions, dtype=torch.long)
        log_probs_t = torch.stack(self.log_probs)
        rewards_t = torch.tensor(self.rewards, dtype=torch.float32)
        values_t = torch.tensor(self.values, dtype=torch.float32)

        # Compute discounted returns
        returns = []
        R = 0
        for r in reversed(rewards_t.tolist()):
            R = r + self.gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Compute advantage
        advantage = returns_t - values_t

        # Compute losses
        actor_loss = -(log_probs_t * advantage).mean()
        critic_loss = (returns_t - values_t).pow(2).mean()

        # Entropy bonus
        feats = self._forward_features(states_t)
        probs = self._policy_distribution(feats)
        dist = torch.distributions.Categorical(probs)
        entropy = dist.entropy().mean()

        loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy

        # Add weight regularization if harmonic
        if self.layer_type == 'harmonic':
            reg_loss = self.weight_reg * torch.norm(self.weights, p=2)
            loss += reg_loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Clear rollouts
        self.states.clear()
        self.actions.clear()
        self.log_probs.clear()
        self.rewards.clear()
        self.values.clear()
        self.dones.clear()



    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

