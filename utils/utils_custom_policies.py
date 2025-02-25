import torch
import numpy as np
import gymnasium as gym 
import torch.nn as nn

from sb3_contrib.common.maskable.policies import MaskableActorCriticPolicy
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.distributions import make_proba_distribution

class HarmonicPolicy(MaskableActorCriticPolicy):
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
        harmonic_exponent=256,
        weight_reg=0.0,
        init_std=0.01,
        **kwargs
    ):
        kwargs.pop('use_sde', None) # cchanged here
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

    def forward_actor(self, features: torch.Tensor, action_masks=None) -> torch.Tensor:
        """
        Compute logits for the action distribution with optional action masking.
        """
        diff = features.unsqueeze(1) - self.weights.unsqueeze(0)
        dist_sq = (diff ** 2).sum(dim=-1)  # shape: (batch_size, n_actions)
        
        if self.distance_norm == "L2":
            dist = torch.sqrt(dist_sq + 1e-8)
        else:
            dist = dist_sq

        dist = dist / torch.min(dist, dim=-1, keepdim = True)[0]

        logits = -self.harmonic_exponent * torch.log(dist + 1e-8)

        # **Apply action mask (set masked actions to a large negative number)**
        if action_masks is not None:
            action_masks = torch.tensor(action_masks, dtype=torch.bool, device=logits.device)
            logits = logits.masked_fill(~action_masks, float('-inf'))

        return logits


    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        """
        Standard value head from the default ActorCriticPolicy
        """
        return self.value_net(features)


    def forward(self, obs: torch.Tensor, deterministic: bool = False, action_masks=None):
        """
        Forward pass with action masking support.
        """
        obs = obs.float()
        features_pi, features_vf = self.mlp_extractor(obs)

        # **Pass action masks to forward_actor**
        logits = self.forward_actor(features_pi, action_masks=action_masks)
        values = self.forward_critic(features_vf)

        distribution = self.get_distribution(logits)

        actions = distribution.get_actions(deterministic=deterministic)
        log_probs = distribution.log_prob(actions)

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


    def _predict(self, observation: torch.Tensor, deterministic: bool = False, action_masks=None):
        """
        This is what stable-baselines calls to get an action from obs.
        """
        observation = observation.float()
        features_pi, features_vf = self.mlp_extractor(observation)
        logits = self.forward_actor(features_pi)
        values = self.forward_critic(features_vf)
        distribution = self.get_distribution(logits)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions

    def evaluate_actions(
        self, obs, actions, action_masks=None
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