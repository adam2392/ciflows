import numpy as np


def space_timesteps(num_timesteps, desired_count, type="uniform"):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :return: a set of diffusion steps from the original process to use.
    """
    if type == "uniform":
        for i in range(1, num_timesteps):
            if len(range(0, num_timesteps, i)) == desired_count:
                return range(0, num_timesteps, i)
        raise ValueError(f"cannot create exactly {desired_count} steps with an integer stride")
    elif type == "quad":
        seq = np.linspace(0, np.sqrt(num_timesteps * 0.8), desired_count) ** 2
        seq = [int(s) for s in list(seq)]
        return seq
    else:
        raise NotImplementedError
