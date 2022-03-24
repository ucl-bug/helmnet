import collections
import numpy as np
from torch import stack

# The ReplayBuffer class and Experience object is built on top of this tutorial:
# https://towardsdatascience.com/en-lightning-reinforcement-learning-a155c217c3de

Experience = collections.namedtuple(
    "Experience",
    field_names=[
        "wavefield",
        "hidden_state",
        "k_sq",
        "residual",
        "source",
        "iteration",
    ],
)

class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = [None for _ in range(capacity)]
        self.capacity = capacity

    def __len__(self):
        return self.capacity

    def append(self, experience, index):
        self.buffer[index] = experience

    def sample(self, batch_size: int):
        if batch_size > self.capacity:
            batch_size = self.capacity
        indices = np.random.choice(self.capacity, batch_size, replace=False)

        wavefields, h_states, k_sqs, residual, source, iterations = zip(
            *[self.buffer[t] for t in indices]
        )

        # Cat them
        wavefields = stack(wavefields, 0)
        h_states = stack(h_states, 0)
        k_sqs = stack(k_sqs, 0)
        residual = stack(residual, 0)
        source = stack(source, 0)

        return (wavefields, h_states, k_sqs, residual, source, iterations, indices)
