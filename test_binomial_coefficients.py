import unittest

import numpy as np
from scipy.special import factorial

from binomial_coefficients import binomial_coefficients


class BinomialCoefficientsTest(unittest.TestCase):
    def test_binomial0(self):
        n_elements = 5
        prob = 0.1
        coefficients = binomial_coefficients(n_elements, prob)
        expected_probabilities = []
        n_factorial = factorial(n_elements)
        for k in range(n_elements + 1):
            k_factorial = factorial(k)
            n_minus_k_factorial = factorial(n_elements - k)

            expected_probability = (n_factorial / (k_factorial * n_minus_k_factorial)) * (prob ** k) * ((1 - prob) ** (n_elements - k))
            expected_probabilities.append(expected_probability)

        expected_probabilities = np.array(expected_probabilities)
        max_delta = np.linalg.norm(coefficients - expected_probabilities, np.inf)

        self.assertAlmostEqual(max_delta, 0.0)

    def test_binomial2(self):
        n_elements = 10
        prob = 0.1
        coefficients = binomial_coefficients(n_elements, prob)
        expected_probabilities = []
        n_factorial = factorial(n_elements)
        for k in range(n_elements + 1):
            k_factorial = factorial(k)
            n_minus_k_factorial = factorial(n_elements - k)

            expected_probability = (n_factorial / (k_factorial * n_minus_k_factorial)) * (prob ** k) * ((1 - prob) ** (n_elements - k))
            expected_probabilities.append(expected_probability)

        expected_probabilities = np.array(expected_probabilities)
        max_delta = np.linalg.norm(coefficients - expected_probabilities, np.inf)

        self.assertAlmostEqual(max_delta, 0.0)

    def test_binomial3(self):
        n_elements = 15
        prob = 0.07
        coefficients = binomial_coefficients(n_elements, prob)
        expected_probabilities = []
        n_factorial = factorial(n_elements)
        for k in range(n_elements + 1):
            k_factorial = factorial(k)
            n_minus_k_factorial = factorial(n_elements - k)

            expected_probability = (n_factorial / (k_factorial * n_minus_k_factorial)) * (prob ** k) * ((1 - prob) ** (n_elements - k))
            expected_probabilities.append(expected_probability)

        expected_probabilities = np.array(expected_probabilities)
        max_delta = np.linalg.norm(coefficients - expected_probabilities, np.inf)

        self.assertAlmostEqual(max_delta, 0.0)

if __name__ == '__main__':
    unittest.main()
