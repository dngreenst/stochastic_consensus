import pickle
from typing import List, Dict, Tuple

import numpy as np
import matplotlib.pyplot as plt


def construct_coefficients_matrix(epsilon_values: List[float], polynomial_rank: int) -> np.ndarray:
    coefficients_matrix = np.ones((polynomial_rank + 1, len(epsilon_values)))
    for idx, epsilon in enumerate(epsilon_values):
        for rank in range(1, polynomial_rank + 1):
            coefficients_matrix[rank, idx] = np.power(epsilon, rank)

    return coefficients_matrix


def predict(x: np.ndarray, epsilon_values: List[float]) -> List[float]:
    ranks = x.shape[0]
    predictions = []
    for epsilon in epsilon_values:
        prediction = 0.0
        for rank in range(ranks):
            prediction += x[rank] * np.power(epsilon, rank)
        predictions.append(prediction)
    return predictions


def construct_target(results: Dict[Tuple[int, float], Tuple[float, float, List[float], List[float]]],
                     epsilon_values: List[float],
                     agents_num: int) -> np.ndarray:
    b = np.zeros(len(epsilon_values))
    for idx, epsilon in enumerate(epsilon_values):
        b[idx] = np.abs(results[(agents_num, epsilon)][0])

    return b


with open('solutions_summary.pkl', 'rb') as f:
    solutions_summary = pickle.load(f)

with open('solutions_summary_validity_respecting.pkl', 'rb') as f:
    solutions_summary_validity_respecting = pickle.load(f)

for key in solutions_summary.keys():
    assert key in solutions_summary_validity_respecting.keys()

for key in solutions_summary_validity_respecting.keys():
    assert key in solutions_summary.keys()

for key in solutions_summary.keys():
    print(f'{key}:')
    print(f'Unconstrained value = {np.abs(solutions_summary[key][0])}')
    print(f'Validity respecting value = {np.abs(solutions_summary_validity_respecting[key][0])}')

epsilon_values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
# epsilon_values = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
# epsilon_values = [0.1, 0.2, 0.3, 0.4]
coefficients_matrix = construct_coefficients_matrix(epsilon_values=epsilon_values, polynomial_rank=2)
b = construct_target(results=solutions_summary, epsilon_values=epsilon_values, agents_num=2)

x, residuals, _, _ = np.linalg.lstsq(a=coefficients_matrix.T, b=b)
print(x)
print(f'Residuals = {residuals}')
predictions = predict(x, epsilon_values=epsilon_values)
plt.plot(epsilon_values, b, 'b.', label='Values')
plt.plot(epsilon_values, predictions, 'r', label='Predictions')
plt.title(f'2 Agents, Unconstrained')

plt.show()

coefficients_matrix = construct_coefficients_matrix(epsilon_values=epsilon_values, polynomial_rank=3)
b = construct_target(results=solutions_summary, epsilon_values=epsilon_values, agents_num=3)
x, residuals, _, _ = np.linalg.lstsq(a=coefficients_matrix.T, b=b)
print(x)
print(f'Residuals = {residuals}')
predictions = predict(x, epsilon_values=epsilon_values)
plt.plot(epsilon_values, b, 'b.', label='Values')
plt.plot(epsilon_values, predictions, 'r', label='Predictions')
plt.title(f'3 Agents, Unconstrained')

plt.show()


coefficients_matrix = construct_coefficients_matrix(epsilon_values=epsilon_values, polynomial_rank=4)
b = construct_target(results=solutions_summary, epsilon_values=epsilon_values, agents_num=4)
x, residuals, _, _ = np.linalg.lstsq(a=coefficients_matrix.T, b=b)
print(x)
print(f'Residuals = {residuals}')
predictions = predict(x, epsilon_values=epsilon_values)
plt.plot(epsilon_values, b, 'b.', label='Values')
plt.plot(epsilon_values, predictions, 'r', label='Predictions')
plt.title(f'4 Agents, Unconstrained')

plt.show()


coefficients_matrix = construct_coefficients_matrix(epsilon_values=epsilon_values, polynomial_rank=5)
b = construct_target(results=solutions_summary, epsilon_values=epsilon_values, agents_num=5)
x, residuals, _, _ = np.linalg.lstsq(a=coefficients_matrix.T, b=b)
print(x)
print(f'Residuals = {residuals}')
predictions = predict(x, epsilon_values=epsilon_values)
plt.plot(epsilon_values, b, 'b.', label='Values')
plt.plot(epsilon_values, predictions, 'r', label='Predictions')
plt.title(f'5 Agents, Unconstrained')

plt.show()

coefficients_matrix = construct_coefficients_matrix(epsilon_values=epsilon_values, polynomial_rank=2)
b = construct_target(results=solutions_summary_validity_respecting, epsilon_values=epsilon_values, agents_num=2)

x, residuals, _, _ = np.linalg.lstsq(a=coefficients_matrix.T, b=b)
print(x)
print(f'Residuals = {residuals}')
predictions = predict(x, epsilon_values=epsilon_values)
plt.plot(epsilon_values, b, 'b.', label='Values')
plt.plot(epsilon_values, predictions, 'r', label='Predictions')
plt.title(f'2 Agents, Validity Respecting')

plt.show()

coefficients_matrix = construct_coefficients_matrix(epsilon_values=epsilon_values, polynomial_rank=3)
b = construct_target(results=solutions_summary_validity_respecting, epsilon_values=epsilon_values, agents_num=3)
x, residuals, _, _ = np.linalg.lstsq(a=coefficients_matrix.T, b=b)
print(x)
print(f'Residuals = {residuals}')
predictions = predict(x, epsilon_values=epsilon_values)
plt.plot(epsilon_values, b, 'b.', label='Values')
plt.plot(epsilon_values, predictions, 'r', label='Predictions')
plt.title(f'3 Agents, Validity Respecting')

plt.show()


coefficients_matrix = construct_coefficients_matrix(epsilon_values=epsilon_values, polynomial_rank=4)
b = construct_target(results=solutions_summary_validity_respecting, epsilon_values=epsilon_values, agents_num=4)
x, residuals, _, _ = np.linalg.lstsq(a=coefficients_matrix.T, b=b)
print(x)
print(f'Residuals = {residuals}')
predictions = predict(x, epsilon_values=epsilon_values)
plt.plot(epsilon_values, b, 'b.', label='Values')
plt.plot(epsilon_values, predictions, 'r', label='Predictions')
plt.title(f'4 Agents, Validity Respecting')

plt.show()


coefficients_matrix = construct_coefficients_matrix(epsilon_values=epsilon_values, polynomial_rank=5)
b = construct_target(results=solutions_summary_validity_respecting, epsilon_values=epsilon_values, agents_num=5)
x, residuals, _, _ = np.linalg.lstsq(a=coefficients_matrix.T, b=b)
print(x)
print(f'Residuals = {residuals}')
predictions = predict(x, epsilon_values=epsilon_values)
plt.plot(epsilon_values, b, 'b.', label='Values')
plt.plot(epsilon_values, predictions, 'r', label='Predictions')
plt.title(f'5 Agents, Validity Respecting')

plt.show()
