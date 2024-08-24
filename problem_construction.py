from typing import Dict, Tuple, cast, Callable

import numpy as np

from binomial_coefficients import binomial_coefficients


def construct_optimization_problem(n_agents: int, epsilon: float) -> (
        Tuple)[Callable[[np.ndarray], float], int, Dict[Tuple[int, int, int], int]]:
    possible_settings = set()
    event_probabilities: Dict[Tuple[int, int], Dict[Tuple[int, int, int], float]] = {}
    for self_state in range(2):
        for n_other_ones in range(n_agents):
            event_probabilities[(self_state, n_other_ones)] = {}
            n_other_zeros = n_agents - 1 - n_other_ones
            if n_other_ones == 0:
                num_viewed_ones_coefficients = np.array([1.0])
            else:
                num_viewed_ones_coefficients = binomial_coefficients(n_elements=n_other_ones, p=epsilon)
            if n_other_zeros == 0:
                num_viewed_zeros_coefficients = np.array([1.0])
            else:
                num_viewed_zeros_coefficients = binomial_coefficients(n_elements=n_other_zeros, p=epsilon)

            for observed_ones_num in range(n_other_ones + 1):
                for observed_zeros_num in range(n_other_zeros + 1):

                    objective_state = (self_state, n_other_ones)
                    subjective_state = (self_state, observed_ones_num, observed_zeros_num)

                    possible_settings.add(subjective_state)

                    event_probabilities[objective_state][subjective_state] = (
                        cast(float, num_viewed_ones_coefficients[observed_ones_num] *
                             num_viewed_zeros_coefficients[observed_zeros_num]))

    possible_settings = list(possible_settings)
    num_settings = len(possible_settings)
    setting_to_index_dict: Dict[Tuple[int, int, int], int] = {}
    for i, setting in enumerate(possible_settings):
        setting_to_index_dict[setting] = i

    def f(p: np.ndarray) -> float:
        violation_penalty_constant = 0.0
        clipped_p = p.get_x() # np.clip(p, a_min=0.0, a_max=1.0)
        # violations = p - clipped_p
        # violation_penalty = violation_penalty_constant * (np.exp(np.linalg.norm(violations, 1.0)) - 1.0)
        min_val = np.inf
        for n_ones in range(n_agents + 1):
            n_zeros = n_agents - n_ones
            agent_states = [1] * n_ones
            agent_states.extend([0] * n_zeros)

            all_agents_output_one_prob = 1.0
            all_agents_output_zero_prob = 1.0

            for agent_state in agent_states:
                possible_remaining_ones = n_ones - agent_state
                possible_remaining_zeros = n_zeros - (1 - agent_state)

                objective_state = (agent_state, possible_remaining_ones)

                output_one_prob = 0.0
                output_zero_prob = 0.0

                for observed_ones_num in range(possible_remaining_ones + 1):
                    for observed_zeros_num in range(possible_remaining_zeros + 1):
                        state = (agent_state, observed_ones_num, observed_zeros_num)
                        state_probability = event_probabilities[objective_state][state]
                        output_one_prob_in_state = clipped_p[setting_to_index_dict[state]]

                        output_one_prob += state_probability * output_one_prob_in_state
                        output_zero_prob += state_probability * (1.0 - output_one_prob_in_state)

                all_agents_output_one_prob = all_agents_output_one_prob * output_one_prob
                all_agents_output_zero_prob = all_agents_output_zero_prob * output_zero_prob

            if n_ones == n_agents:
                val = all_agents_output_one_prob
            elif n_zeros == n_agents:
                val = all_agents_output_zero_prob
            else:
                val = all_agents_output_zero_prob + all_agents_output_one_prob

            min_val = min(min_val, val)

        return -min_val

    return f, num_settings, setting_to_index_dict



