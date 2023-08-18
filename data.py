import numpy as np
import random


# return an overlapping student list as a np array
def get_overlapping_student_list(overlap_coefficient, n):
    student_num = int(n * (1 - overlap_coefficient))
    raw_student_list: np.ndarray = np.arange(student_num)
    overlap_total_student_list: np.ndarray = np.array([raw_student_list[i % student_num] for i in range(n)])
    np.random.shuffle(overlap_total_student_list)
    return raw_student_list, overlap_total_student_list


# return {group_num : student indices as np array}
# requires overlap_coefficient < 1.0
def create_student_group_slot(overlap_coefficient, g_list, U_sampler, r):
    g_list = np.array(g_list)
    g_num = len(g_list)
    n = np.sum(g_list)
    g_probs = g_list / n
    raw_student_list, overlap_student_list = get_overlapping_student_list(overlap_coefficient, n)

    group_table = {g: set() for g in range(g_num)}
    U: np.ndarray = np.zeros((len(raw_student_list), g_num))
    for s in overlap_student_list:
        for g in np.random.choice(np.arange(g_num), size=g_num, p=g_probs, replace=False):
            if s in group_table[g] or len(group_table[g]) == g_list[g]: continue
            group_table[g].add(s)
            U[s, g] = 1 # produce qualification
            break # has been added to a group, so we want to braek the inner loop
    
    U_sample_list: np.ndarray = U_sampler(U, r)

    return raw_student_list, {g: np.array(sorted(list(student_set))) for g, student_set in group_table.items()}, U, U_sample_list


def iid_bernoulli_sampler(U, r, p=0.75):
    nonzero_indices = np.nonzero(U)
    ret: np.ndarray = np.zeros((r,) + U.shape)
    for i in range(r):
        ret[i, nonzero_indices] = np.random.binomial(n=1, p=p, size=nonzero_indices[0].shape)
    return ret

