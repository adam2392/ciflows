import normflows as nf


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
