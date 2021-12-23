from collections import defaultdict
import numpy as np

# Environment object
from .env3D_4x4 import GridWorld_3D_env
env = GridWorld_3D_env()

TARGET = defaultdict(lambda: np.zeros(env.action_space.n))

TARGET[0] = np.array([0, 0, 1, 0])
TARGET[4] = np.array([0, 0, 1, 0])
TARGET[8] = np.array([0, 0, 1, 0])
TARGET[12] = np.array([0, 1, 0, 0])
TARGET[13] = np.array([0, 1, 0, 0])