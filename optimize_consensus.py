import pickle

import numpy as np
from scipy.optimize import Bounds, minimize
from zoopt import Objective, Dimension, Opt, Parameter

from problem_construction import construct_optimization_problem

solutions_summary = {}

# n_agents = 4
# epsilon = 0.9

for n_agents in range(2, 6):
    for epsilon in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]:

        problem, n_decision_vars, state_to_index_mapper = construct_optimization_problem(n_agents=n_agents, epsilon=epsilon)
        index_to_state_mapper = {}
        for state, index in state_to_index_mapper.items():
            index_to_state_mapper[index] = state

        bounds = np.zeros((n_decision_vars, 2))
        bounds[:, 1] = 1.0

        bounds = Bounds(bounds)
        attempts_num = 1000
        best_val = np.inf
        best_val_solution = None
        for _ in range(attempts_num):
            p0 = np.random.normal(loc=0.5, size=n_decision_vars)
            p0 = np.clip(p0, a_min=0.0, a_max=1.0)

            # constraints = {'type': 'ineq',
            #                'fun' : lambda p: np.concatenate([p - 1, -p]),
            #                'jac' : lambda p: np.concatenate([np.eye(p.shape[0]), -np.eye(p.shape[0])])}

            # res = minimize(fun=problem, x0=p0, constraints=constraints)
            # res = minimize(method='nelder-mead', fun=problem, x0=p0)
            # res = minimize(method='Powell', fun=problem, x0=p0)

            dim = n_decision_vars  # dimension
            obj = Objective(problem, Dimension(dim, [[0, 1]] * dim, [True] * dim))
            # perform optimization
            solution = Opt.min(obj, Parameter(budget=1000 * dim))
            # print the solution
            # print(solution.get_x(), solution.get_value())
            if solution.get_value() < best_val:
                best_val = solution.get_value()
                best_val_solution = solution.get_x()

        print(f'Best val is: {best_val}')
        print(f'Best val solution is: {best_val_solution}')

        for i in range(len(best_val_solution)):
            print(f'Probability of outputting {1} at state {index_to_state_mapper[i]} is {best_val_solution[i]}')

        for i in range(len(best_val_solution)):
            if 1.0 - best_val_solution[i] < 0.05:
                solution.set_x_index(index=i, x=1.0)
            elif best_val_solution[i] < 0.05:
                solution.set_x_index(i, 0.0)
            else:
                solution.set_x_index(i, best_val_solution[i])

        new_val = problem(solution)
        print(f'New val is: {new_val}')
        print(f'New solution is: {solution.get_x()}')
        for i in range(len(best_val_solution)):
            print(f'Probability of outputting {1} at state {index_to_state_mapper[i]} is {solution.get_x()[i]}')

        solutions_summary[(n_agents, epsilon)] = (best_val, new_val, best_val_solution, solution.get_x())

print(solutions_summary)
filename = 'solutions_summary.pkl'
with open(filename,"wb") as file:
    pickle.dump(solutions_summary, file)

