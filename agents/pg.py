import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PG(nn.Module):
    def __init__(self, 
                 env, 
                 hidden_dim=64, 
                 embedding_dim=16, 
                 layer_type='softmax', 
                 distance_norm='L2', 
                 lr=0.001, # 0.001
                 gamma=0.99, 
                 n_layers=3, 
                 dropout_prob=0.3, 
                 harmonic_exponent=4, 
                 weight_reg=0.00):
        super(PG, self).__init__()

        self.env = env
        input_dim = env.observation_space.shape[0]
        output_dim = env.action_space.n

        self.lr = lr
        self.gamma = gamma
        self.layer_type = layer_type
        self.distance = distance_norm   # 'L2' or 'squared'
        self.harmonic_exponent = harmonic_exponent
        self.weight_reg = weight_reg

        # Build MLP feature extractor
        layers = []
        in_dim = input_dim
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(p=dropout_prob))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, hidden_dim))
        self.mlp = nn.Sequential(*layers)

        # Output layer
        if layer_type == 'softmax':
            # Standard classification head
            self.output_layer = nn.Linear(hidden_dim, output_dim, bias=False)
        else:
            # Harmonic approach: store class centers (weights)
            # shape: [output_dim, hidden_dim]
            self.weights = nn.Parameter(torch.randn(output_dim, hidden_dim))

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        """
        x: [batch_size, obs_dim]
        returns: probabilities over actions [batch_size, num_actions]
        """
        features = self.mlp(x)  # Feature extraction

        if self.layer_type == 'softmax':
            logits = self.output_layer(features)
            return torch.softmax(logits, dim=1)
        else:
            # Compute difference between features and class centers
            diff = features[:, None, :] - self.weights[None, :, :]
            
            # Choose distance metric based on self.distance
            if self.distance.upper() == 'L1':
                dist = diff.abs().sum(dim=-1)  # L1 norm
            else:  # Default: L2
                dist_sq = (diff ** 2).sum(dim=-1)
                dist = torch.sqrt(dist_sq + 1e-8)  # L2 norm

            #print(f"Dist stats - min: {dist.min().item():.4f}, max: {dist.max().item():.4f}, mean: {dist.mean().item():.4f}")

            # Normalize by min distance for stable scaling
            min_dist = torch.min(dist, dim=-1, keepdim=True)[0]
            if torch.any(min_dist < 1e-6):
                print(f"⚠️ Warning: Extremely small min distance detected, potential instability!")

            
            # ✅ Normalize by min distance for stable scaling (prevent division instability)
            dist = dist / torch.clamp(torch.min(dist, dim=-1, keepdim=True)[0], min=1e-6)

            # Raise distance to harmonic_exponent and invert
            dist_exp = dist ** self.harmonic_exponent
            inv_dist = 1.0 / (dist_exp + 1e-8)

            # Normalize to sum to 1
            probs = inv_dist / inv_dist.sum(dim=1, keepdim=True)
            return probs



    def predict(self, state, deterministic=False):
        """
        Single-step action prediction
        """
        with torch.no_grad():
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
        Roll out episodes until we hit total_timesteps, then do a policy update
        after each episode. 
        """
        timestep = 0
        log_probs_all = []
        while timestep < total_timesteps:
            state = self.env.reset()
            self.saved_log_probs = []
            self.rewards = []
            done = False

            while not done:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
                action_probs = self.forward(state_tensor)

                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)

                #print(f"log_prob stats - min: {log_prob.min().item():.4f}, max: {log_prob.max().item():.4f}, mean: {log_prob.mean().item():.4f}")

                next_state, reward, done, info = self.env.step(action.item())

                #self.env.render()

                self.store_outcome(log_prob, reward)
                state = next_state
                timestep += 1

                if callback is not None:
                    callback._on_step(reward)
                    if callback.early_stop:
                        return

                if timestep >= total_timesteps:
                    break

            # Update policy at the end of the episode
            self._update_policy()


    # def learn(self, total_timesteps=10000, callback=None):
    #     """
    #     Roll out episodes until we hit total_timesteps, then do a policy update
    #     after each episode. Includes debugging for log probabilities and entropy.
    #     """
    #     timestep = 0
    #     log_probs_all = []  # Stores all log probs across episodes
    #     entropy_all = []  # Tracks entropy for debugging
        
    #     while timestep < total_timesteps:
    #         state = self.env.reset()
    #         self.saved_log_probs = []
    #         self.rewards = []
    #         episode_log_probs = []  # Track log probs for this episode
    #         episode_entropies = []  # Track entropies
    #         episode_action_probs = []  # Store sampled action probabilities
    #         done = False

    #         while not done:
    #             state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
    #             action_probs = self.forward(state_tensor)  # [1, num_actions]

    #             dist = torch.distributions.Categorical(action_probs)
    #             action = dist.sample()
    #             log_prob = dist.log_prob(action).clamp(min=-3)  # Prevent extreme values
    #             entropy = dist.entropy().item()  # Track entropy
                
    #             next_state, reward, done, info = self.env.step(action.item())

    #             self.store_outcome(log_prob, reward)
    #             state = next_state
    #             timestep += 1

    #             # Collect debugging stats
    #             episode_log_probs.append(log_prob.item())
    #             episode_entropies.append(entropy)
    #             episode_action_probs.append(action_probs.squeeze().tolist())

    #             if callback is not None:
    #                 callback._on_step(reward)
    #                 if callback.early_stop:
    #                     return

    #             if timestep >= total_timesteps:
    #                 break

    #         # Update policy at the end of the episode
    #         self._update_policy()

    #         # Store episode logs
    #         log_probs_all.extend(episode_log_probs)
    #         entropy_all.extend(episode_entropies)

    #         # Print log_prob stats per episode
    #         if len(episode_log_probs) > 0:
    #             print(f"[EPISODE LOG PROB] Min: {min(episode_log_probs):.4f}, "
    #                   f"Max: {max(episode_log_probs):.4f}, "
    #                   f"Mean: {np.mean(episode_log_probs):.4f}, "
    #                   f"Std: {np.std(episode_log_probs):.4f}")

    #         # Print entropy stats per episode
    #         if len(episode_entropies) > 0:
    #             print(f"[EPISODE ENTROPY] Min: {min(episode_entropies):.4f}, "
    #                   f"Max: {max(episode_entropies):.4f}, "
    #                   f"Mean: {np.mean(episode_entropies):.4f}, "
    #                   f"Std: {np.std(episode_entropies):.4f}")

    #         # Debug action probabilities to check for deterministic collapse
    #         print(f"[EPISODE ACTION PROBS] Sample: {np.mean(episode_action_probs, axis=0)}")

    #     # Optional: Visualize log probabilities over time
    #     import matplotlib.pyplot as plt
    #     plt.figure(figsize=(10, 4))
    #     plt.plot(log_probs_all, label="Log Probabilities")
    #     plt.xlabel("Timestep")
    #     plt.ylabel("Log Probability")
    #     plt.title("Log Probability Over Time")
    #     plt.legend()
    #     plt.show()



    def _update_policy(self):
        """
        Standard REINFORCE: discounted returns, advantage ~ (Return - baseline),
        but here we just use normalized returns as a simple baseline-subtraction.
        """
        R = 0
        returns = []
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)

        # ✅ Ensure correct dtype and device
        device = next(self.parameters()).device  # Use any model parameter to get device
        returns = torch.tensor(returns, dtype=torch.float32, device=device)
        #returns = torch.tensor(returns, dtype=torch.float32, device=self.weights.device)

        # ✅ Normalize returns
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # ✅ Use sum() instead of cat() for efficiency
        policy_loss = sum(-log_prob * ret for log_prob, ret in zip(self.saved_log_probs, returns))

        # ✅ Improved weight regularization
        if self.layer_type == 'harmonic':
            reg_loss = self.weight_reg * torch.mean(self.weights ** 2)
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
