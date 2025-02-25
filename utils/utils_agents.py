import torch
import gymnasium as gym 
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.distributions import make_proba_distribution

class HarmonicPolicy(ActorCriticPolicy):
    """
    A custom ActorCriticPolicy that replaces the usual linear 'policy_net'
    with a distance-based 'harmonic' layer.

    The main idea:
      - We have `self.weights` of shape [action_dim, pi_latent_dim]
      - We compute distance from policy features to each action's 'center'
      - We create logits = -exponent * log(distance + eps)
      - Return those logits to a standard Categorical distribution
    """

    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        distance_norm="L2",     # "L2" or "squared"
        harmonic_exponent=4,
        weight_reg=0.0,
        init_std=0.01,
        **kwargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Let SB3 build the default feature extractor, etc.
            **kwargs
        )
        self.distance_norm = distance_norm
        self.harmonic_exponent = harmonic_exponent
        self.weight_reg = weight_reg

        self.action_dist = make_proba_distribution(action_space)

        # The size of the policy's latent (output of mlp_extractor.latent_pi)
        pi_latent_dim = self.mlp_extractor.latent_dim_pi
        # The number of discrete actions
        action_dim = self.action_space.n

        # Our "action center" embeddings
        self.weights = nn.Parameter(
            torch.randn(action_dim, pi_latent_dim) * init_std
        )

        # We won't use self.action_net anymore, but let's delete or block it
        # so we don't accidentally use it. Or just leave it unused.
        # del self.action_net

        # Important: call reset_parameters() if desired
        # self._initialize_weights()

    def _initialize_weights(self):
        # You can do custom init here if you like
        nn.init.normal_(self.weights, mean=0.0, std=0.01)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        """
        Overriding how we compute the 'logits' for the action distribution.
        :param features: shape [batch_size, pi_latent_dim]
        :return: shape [batch_size, n_actions], which is the "logits"
        """
        # features: [batch_size, pi_latent_dim]
        # self.weights: [n_actions, pi_latent_dim]
        # We compute distance from each sample to each action center

        # Expand to shape [batch_size, n_actions, pi_latent_dim]
        # so we can do a diff along the last dimension
        diff = features.unsqueeze(1) - self.weights.unsqueeze(0)
        # shape: (batch_size, n_actions, pi_latent_dim)

        # L2 or squared
        dist_sq = (diff ** 2).sum(dim=-1)  # shape: (batch_size, n_actions)
        if self.distance_norm == "L2":
            dist = torch.sqrt(dist_sq + 1e-8)
        else:
            dist = dist_sq

        # dist^exponent
        dist_exp = dist ** self.harmonic_exponent
        # logits = -exponent * log(dist) => we can do:
        # but let's go more stable: logit = -exponent * log(dist+eps)
        # That matches the "harmonic" 1/d^exponent approach once you exponentiate
        logits = -self.harmonic_exponent * torch.log(dist_exp + 1e-8)
        # Actually that is the same as -exponent * exponent * log(dist) => watch out
        # simpler to do: logits = -exponent * log(dist + 1e-8)
        # We'll do that:
        logits = -self.harmonic_exponent * torch.log(dist + 1e-8)

        return logits

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """
        Standard value head from the default ActorCriticPolicy
        """
        return self.value_net(features)

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        #obs = obs.float()
        features_pi, features_vf = self.mlp_extractor(obs)

        logits = self.forward_actor(features_pi)
        values = self.forward_critic(features_vf)

        distribution = self.get_distribution(logits)
        actions = distribution.get_actions(deterministic=deterministic)
        
        # Compute log_probs for those sampled actions:
        log_probs = distribution.log_prob(actions)

        # Now return all three:
        return actions, values, log_probs


    def get_distribution(self, logits: torch.Tensor):
        """
        Create the Distribution (Categorical if discrete, DiagGaussian if continuous) 
        from the raw 'logits' or mean/sigma parameters. 
        In a discrete-action setting, 'logits' -> CategoricalDistribution.
        """
        # If your environment is discrete, we can do:
        # The parent ActorCriticPolicy usually has self.action_dist = CategoricalDistribution(n_actions)
        # or you can define it yourself. Then do:
        # return self.action_dist.proba_distribution(action_logits=logits)

        return self.action_dist.proba_distribution(action_logits=logits)


    def _predict(self, observation: torch.Tensor, deterministic: bool = False):
        """
        This is what stable-baselines calls to get an action from obs.
        """
        features_pi, features_vf = self.mlp_extractor(observation)
        logits = self.forward_actor(features_pi)
        values = self.forward_critic(features_vf)
        distribution = self._make_dist(logits)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions

    def evaluate_actions(
        self, obs, actions
    ):
        """
        Called during training to get log_probs, entropy, and values
        for a batch of obs + actions.
        """
        features_pi, features_vf = self.mlp_extractor(obs)
        logits = self.forward_actor(features_pi)
        values = self.forward_critic(features_vf)
        distribution = self.get_distribution(logits)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy().mean()
        
        # if we want weight regularization, do it here or in custom loss
        # e.g., reg_loss = self.weight_reg * self.weights.norm(p=2)
        # but typically you'd patch that into the on_training_step callback.

        return values, log_prob, entropy