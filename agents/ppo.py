import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class PPO(nn.Module):
    """
    A minimal Proximal Policy Optimization (PPO) with clipped objective.
    """
    def __init__(
        self,
        env,
        hidden_dim=64,
        layer_type='softmax',    # or 'harmonic'
        distance_norm='L2',
        harmonic_exponent=4,
        weight_reg=0.01,
        lr=0.0003,
        gamma=0.99,
        lam=0.95,
        eps_clip=0.2,
        value_coef=0.5,
        entropy_coef=0.0,
        rollout_steps=2048,
        minibatch_size=64,
        update_epochs=10,
        n_layers=2,
        dropout_prob=0.3
    ):
        super(PPO, self).__init__()

        self.env = env
        self.gamma = gamma
        self.lam = lam
        self.layer_type = layer_type
        self.distance_norm = distance_norm
        self.harmonic_exponent = harmonic_exponent
        self.weight_reg = weight_reg

        self.eps_clip = eps_clip
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.rollout_steps = rollout_steps
        self.minibatch_size = minibatch_size
        self.update_epochs = update_epochs

        obs_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # Shared feature extractor
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_prob))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())  # final
        self.feature_extractor = nn.Sequential(*layers)

        # Policy head
        if layer_type == 'softmax':
            self.policy_head = nn.Linear(hidden_dim, action_dim, bias=False)
        else:
            self.weights = nn.Parameter(torch.randn(action_dim, hidden_dim))

        # Value head
        self.value_head = nn.Linear(hidden_dim, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)

        # We'll store transitions for a rollout
        self.states = []
        self.actions = []
        self.log_probs_old = []
        self.rewards = []
        self.dones = []
        self.values = []

    def _forward_features(self, x):
        return self.feature_extractor(x)

    def _policy_distribution(self, features):
        action_dim = self.env.action_space.n
        if self.layer_type == 'softmax':
            logits = self.policy_head(features)
            probs = torch.softmax(logits, dim=1)
            return probs
        else:
            diff = features[:, None, :] - self.weights[None, :, :]
            if self.distance_norm == 'L1':
                dist = diff.abs().sum(dim=-1)
            else:  # Default to L2
                dist = torch.sqrt((diff**2).sum(dim=-1) + 1e-8)

            dist_exp = dist ** self.harmonic_exponent
            inv_dist = 1.0 / (dist_exp + 1e-8)
            probs = inv_dist / inv_dist.sum(dim=1, keepdim=True)
            return probs

    def _value_function(self, features):
        return self.value_head(features)

    def forward(self, x):
        feats = self._forward_features(x)
        probs = self._policy_distribution(feats)
        value = self._value_function(feats)
        return probs, value

    def predict(self, state, deterministic=False):
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            feats = self._forward_features(state_t)
            probs = self._policy_distribution(feats)
            if deterministic:
                return probs.argmax(dim=1).item()
            else:
                dist = torch.distributions.Categorical(probs)
                return dist.sample().item()

    def store_transition(self, state, action, log_prob, reward, done, value):
        self.states.append(state)
        self.actions.append(action)
        self.log_probs_old.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def learn(self, total_timesteps=1_000_00, callback=None):
        """
        Main PPO loop:
          - collect rollout_steps transitions
          - compute advantage, returns
          - update policy multiple epochs
        """
        timestep = 0
        state = self.env.reset()
        while timestep < total_timesteps:
            self.states.clear()
            self.actions.clear()
            self.log_probs_old.clear()
            self.rewards.clear()
            self.dones.clear()
            self.values.clear()

            for _ in range(self.rollout_steps):
                state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                feats = self._forward_features(state_t)
                probs = self._policy_distribution(feats)
                value = self._value_function(feats).squeeze(1)

                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                next_state, reward, done, info = self.env.step(action.item())

                self.store_transition(
                    state, action.item(), log_prob.item(), reward, done, value.item()
                )

                state = next_state
                timestep += 1

                if callback is not None:
                    callback._on_step(reward)
                    if callback.early_stop:
                        return

                if done or timestep >= total_timesteps:
                    state = self.env.reset()
                    break

            self._update_policy()

    def _update_policy(self):
        """
        Perform multiple epochs of PPO updates on the gathered rollout.
        """
        # Convert lists to NumPy array before Tensor conversion (fix slow warning)
        states_np = np.array(self.states, dtype=np.float32)
        actions_np = np.array(self.actions, dtype=np.int64)
        log_probs_old_np = np.array(self.log_probs_old, dtype=np.float32)
        rewards_np = np.array(self.rewards, dtype=np.float32)
        dones_np = np.array(self.dones, dtype=np.float32)
        values_np = np.array(self.values, dtype=np.float32)

        states_t = torch.from_numpy(states_np)
        actions_t = torch.from_numpy(actions_np)
        log_probs_old_t = torch.from_numpy(log_probs_old_np)
        rewards_t = torch.from_numpy(rewards_np)
        dones_t = torch.from_numpy(dones_np)
        values_t = torch.from_numpy(values_np)

        # Compute returns
        returns = []
        R = 0
        for i in reversed(range(len(rewards_t))):
            if dones_t[i] == 1.0:
                R = 0
            R = rewards_t[i] + self.gamma * R
            returns.insert(0, R)
        returns_t = torch.tensor(returns, dtype=torch.float32)

        # Advantage
        advantage_t = returns_t - values_t
        advantage_t = (advantage_t - advantage_t.mean()) / (advantage_t.std() + 1e-8)

        # Minibatch SGD
        rollout_size = len(states_t)
        indices = np.arange(rollout_size)

        for _ in range(self.update_epochs):
            np.random.shuffle(indices)

            for start in range(0, rollout_size, self.minibatch_size):
                end = start + self.minibatch_size
                mb_idx = indices[start:end]

                mb_states = states_t[mb_idx]
                mb_actions = actions_t[mb_idx]
                mb_log_probs_old = log_probs_old_t[mb_idx]
                mb_advantages = advantage_t[mb_idx]
                mb_returns = returns_t[mb_idx]

                feats = self._forward_features(mb_states)
                probs = self._policy_distribution(feats)
                dist = torch.distributions.Categorical(probs)
                mb_log_probs_new = dist.log_prob(mb_actions)

                ratio = torch.exp(mb_log_probs_new - mb_log_probs_old)
                clipped_ratio = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip)
                actor_loss = -torch.min(ratio * mb_advantages, clipped_ratio * mb_advantages).mean()
                
                critic_loss = (mb_returns - self._value_function(feats).squeeze(1)).pow(2).mean()

                loss = actor_loss + self.value_coef * critic_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))

