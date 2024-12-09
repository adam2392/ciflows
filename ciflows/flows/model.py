import normflows as nf
import torch

from ciflows.distributions.linear import MultidistrCausalFlow


class InjectiveFlow(nf.NormalizingFlow):
    def __init__(self, q0, flows):
        # we will error-check the flows to make sure the dimensionality is consistent
        for flow in flows:
            assert flow.d == q0.d

        # initialize the base distribution and the resulting flows
        super().__init__(q0, flows)

    def log_prob(self, x):
        # surrogate log probability
        pass

    def sample(self, num_samples=1):
        return super().sample(num_samples)


class CausalNormalizingFlow(nf.NormalizingFlow):
    """
    Normalizing Flow model with causal latent structure.
    """

    q0: MultidistrCausalFlow

    def __init__(self, q0, flows, p=None):
        """Constructor

        Args:

          q0: List of base distribution
          flows: List of list of flows for each level
          p: Target distribution
        """
        super().__init__(q0=q0, flows=flows, p=p)

    def forward_kld(self, x, distr_idx=None, intervention_targets=None, hard_interventions=None):
        """Estimates forward KL divergence, see [arXiv 1912.02762](https://arxiv.org/abs/1912.02762)

        Args:
            x: Batch sampled from target distribution
            distr_idx: List of distribution index

        Returns:
          Estimate of forward KL divergence averaged over batch
        """
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(z, distr_idx=distr_idx)
        return -torch.mean(log_q)

    def sample(self, num_samples=1, **kwargs):
        """Samples from flow-based approximate distribution

        Args:
          num_samples: Number of samples to draw

        Returns:
          Samples, log probability
        """
        z, log_q = self.q0(num_samples, **kwargs)
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        return z, log_q

    def log_prob(self, x, distr_idx=None, intervention_targets=None, hard_interventions=None):
        """Get log probability for batch

        Args:
          x: Batch

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        log_q += self.q0.log_prob(
            z,
            distr_idx=distr_idx,  # intervention_targets, hard_interventions
        )
        return log_q
