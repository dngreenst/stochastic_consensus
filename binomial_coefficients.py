
from scipy.stats import binom
import numpy as np


def binomial_coefficients(n_elements: int, p: float) -> np.ndarray:
    binomial_dist_pmf = binom(n_elements, p).pmf
    return np.array([binomial_dist_pmf(i) for i in range(n_elements + 1)])
