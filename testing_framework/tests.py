import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from algorithms.base.base import MABAlgorithm


def test_pe_algorithm(algo: MABAlgorithm, arms, num_sims, horizon, num_save_step=100):
    """ For Pure Exploration (a.k.a. Best-arm Identification) problems.
    """
    chosen_arm_idxes = [0.0 for _ in range(num_sims * num_save_step)]
    estimated_best_arm_idxes = [0.0 for _ in range(num_sims * num_save_step)]
    error_list = [0.0 for _ in range(num_sims * num_save_step)]   # 1 if estimated arm is not in optimal set of arm else 0
    simple_regret = [0.0 for _ in range(num_sims * num_save_step)]
    sim_nums = [0.0 for _ in range(num_sims * num_save_step)]
    times = [0.0 for _ in range(num_sims * num_save_step)]

    algo_class_name = algo.__class__.__name__
    if algo_class_name == 'OptNN':
        # math: for time step t, track metric is defined as max_i |t * policy_i - N_i(t)| / t
        track_metric_list = [0.0 for _ in range(num_sims * num_save_step)]
        emp_dist_list = [0.0 for _ in range(num_sims * num_save_step)]   # empirical distribution

    # for sim in range(num_sims):
    for sim in tqdm(range(num_sims)):
        sim = sim + 1
        print(f'algo {algo_class_name} sim {sim}')

        algo.initialize(seed=sim)

        arms.initialize(seed=sim)

        max_reward = arms.max_reward
        best_arm = arms.best_arm

        tsav = np.linspace(0, horizon - 1, num_save_step, dtype=int)
        tsav_pointer = 0

        for t in range(horizon):
            # 1. select action for exploration
            chosen_arm_idx = algo.select_arm()

            # 2. observe reward for taken action
            reward = arms.draw(arm_idx=chosen_arm_idx, time=t)

            # 3. update algorithm parameter based on observation
            algo.update(chosen_arm_idx, reward)

            # 4. predict best arm from past observations
            estimated_best_arm_idx = algo.predict_best_arm()
            expected_ba_reward = arms.expected_reward_list[estimated_best_arm_idx]

            if t == tsav[tsav_pointer]:
                # index = (sim - 1) * num_save_step + tsav_pointer - 1  # index for above list
                index = (sim - 1) * num_save_step + tsav_pointer  # index for above list
                sim_nums[index] = sim
                times[index] = t
                chosen_arm_idxes[index] = chosen_arm_idx

                estimated_best_arm_idxes[index] = estimated_best_arm_idx
                simple_regret[index] = max_reward - expected_ba_reward
                error_list[index] = int(estimated_best_arm_idx != best_arm)  # todo: we may extend to eps-optimal setting

                if algo_class_name == 'OptNN':
                    track_metric = np.max(np.abs(algo.step * algo.policy - np.array(algo.counts)) / algo.step)
                    track_metric_list[index] = track_metric

                    emp_dist_list[index] = algo.values


                # print(f'algo {algo.__class__.__name__} sim {sim} regret at time {t} ', pseudo_regret_list[index])
                tsav_pointer += 1

    results_dict = {
        'sim_nums': sim_nums, 'times': times,
         'chosen_arm_idxes': chosen_arm_idxes,
         'estimated_best_arm_idxes': estimated_best_arm_idxes,
         'simple_regret': simple_regret,
         'error': error_list,
         # 'rewards': rewards, 'cumulative_reward': cumulative_rewards, 'pseudo_regret': pseudo_regret,
         # 'cumulative_regret': cumulative_regret,
     }
    if algo_class_name == 'OptNN':
        results_dict['track_metric'] = track_metric_list
        results_dict['emp_dist'] = emp_dist_list
    results_df = pd.DataFrame(results_dict)

    return results_df





# ***********************************************  # todo: refactor
from multiprocessing import Process, Queue


def test_pe_single_episode(algo: MABAlgorithm, arms, horizon, num_save_step, sim, q):
    """ taken from the inner loop of test_pe_algorithm for parallel experiments """
    # TODO: NOTE that followings are just for SINGLE episode
    chosen_arm_idxes = [0.0 for _ in range(num_save_step)]
    estimated_best_arm_idxes = [0.0 for _ in range(num_save_step)]
    error_list = [0.0 for _ in range(num_save_step)]   # 1 if estimated arm is not in optimal set of arm else 0
    simple_regret = [0.0 for _ in range(num_save_step)]
    sim_nums = [0.0 for _ in range(num_save_step)]
    times = [0.0 for _ in range(num_save_step)]

    algo_class_name = algo.__class__.__name__
    if algo_class_name == 'OptNN':
        # math: for time step t, track metric is defined as max_i |t * policy_i - N_i(t)| / t
        track_metric_list = [0.0 for _ in range(num_save_step)]
        emp_dist_list = [0.0 for _ in range(num_save_step)]   # empirical distribution
    print(f'algo {algo_class_name} sim {sim}')

    algo.initialize(seed=sim)
    arms.initialize(seed=sim)

    max_reward = arms.max_reward
    best_arm = arms.best_arm

    tsav = np.linspace(0, horizon - 1, num_save_step, dtype=int)
    tsav_pointer = 0

    for t in range(horizon):
        # 1. select action for exploration
        chosen_arm_idx = algo.select_arm()

        # 2. observe reward for taken action
        reward = arms.draw(arm_idx=chosen_arm_idx, time=t)

        # 3. update algorithm parameter based on observation
        algo.update(chosen_arm_idx, reward)

        # 4. predict best arm from past observations
        estimated_best_arm_idx = algo.predict_best_arm()
        expected_ba_reward = arms.expected_reward_list[estimated_best_arm_idx]

        if t == tsav[tsav_pointer]:
            index = tsav_pointer  # different from non-parallel implementation
            sim_nums[index] = sim
            times[index] = t
            chosen_arm_idxes[index] = chosen_arm_idx

            estimated_best_arm_idxes[index] = estimated_best_arm_idx
            simple_regret[index] = max_reward - expected_ba_reward
            error_list[index] = int(estimated_best_arm_idx != best_arm)  # todo: we may extend to eps-optimal setting

            if algo_class_name == 'OptNN':
                track_metric = np.max(np.abs(algo.step * algo.policy - np.array(algo.counts)) / algo.step)
                track_metric_list[index] = track_metric

                emp_dist_list[index] = algo.values

            tsav_pointer += 1

    episode_results_dict = {
        'sim_nums': sim_nums, 'times': times,
        'chosen_arm_idxes': chosen_arm_idxes,
        'estimated_best_arm_idxes': estimated_best_arm_idxes,
        'simple_regret': simple_regret,
        'error': error_list,
    }
    if algo_class_name == 'OptNN':
        episode_results_dict['track_metric'] = track_metric_list
        episode_results_dict['emp_dist'] = emp_dist_list

    q.put(episode_results_dict)


def test_pe_algorithm_parallel(algo: MABAlgorithm, arms, num_sims, horizon, num_save_step=100):
    """ same argument as that of test_pe_algorithm """
    chosen_arm_idxes = []
    estimated_best_arm_idxes = []
    error_list = []
    simple_regret = []
    sim_nums = []
    times = []

    algo_class_name = algo.__class__.__name__
    if algo_class_name == 'OptNN':
        track_metric_list = []
        emp_dist_list = []

    num_subproc = 5
    iteration_num = int(num_sims / num_subproc)

    # TODO: maybe
    assert num_sims == num_subproc * iteration_num

    q_list = [(proc_idx, Queue()) for proc_idx in range(num_subproc)]

    for iter_idx in tqdm(range(iteration_num)):
        p_list = [Process(target=test_pe_single_episode, args=(algo, arms, horizon, num_save_step, num_subproc * iter_idx + proc_idx, q)) for proc_idx, q in q_list]

        for p in p_list:
            p.start()

        for _, q in q_list:
            episode_result_dict = q.get()
            # TODO: writing here is ok?
            sim_nums.extend(episode_result_dict['sim_nums'])
            times.extend(episode_result_dict['times'])
            chosen_arm_idxes.extend(episode_result_dict['chosen_arm_idxes'])
            estimated_best_arm_idxes.extend(episode_result_dict['estimated_best_arm_idxes'])
            simple_regret.extend(episode_result_dict['simple_regret'])
            error_list.extend(episode_result_dict['error'])
            if algo_class_name == 'OptNN':
                track_metric_list.extend(episode_result_dict['track_metric'])
                emp_dist_list.extend(episode_result_dict['emp_dist'])

        for p in p_list:
            p.join()

    results_dict = {
        'sim_nums': sim_nums, 'times': times,
        'chosen_arm_idxes': chosen_arm_idxes,
        'estimated_best_arm_idxes': estimated_best_arm_idxes,
        'simple_regret': simple_regret,
        'error': error_list,
    }
    if algo_class_name == 'OptNN':
        results_dict['track_metric'] = track_metric_list
        results_dict['emp_dist'] = emp_dist_list
    results_df = pd.DataFrame(results_dict)

    assert len(chosen_arm_idxes) == num_sims * num_save_step

    return results_df




