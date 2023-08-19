import numpy as np
from basic_algorithm import *
from greedy_algorithm import *
from general_experiment import *
from data import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from copy import deepcopy
import random
import os
import pickle
import argparse


parser = argparse.ArgumentParser(description="ranking with synthetic dataset, misspecify the relevance level in training set")
parser.add_argument(
    "--magnitude",
    type=str,
    default='s',
    choices=['xs', 's', 'l', 'xl']
)
parser.add_argument(
    "--greedy_size",
    type=int,
    default=1500,
)
args = parser.parse_args()
print(vars(args))

seed_everything(0)

assigned_group = 2
total_group = 10
slots_per_group = 50
candidate_num = 10000
magnitude = args.magnitude
train_n = 200
test_n = 1000
slot_list = np.array([slots_per_group] * total_group, dtype=np.int32)

basic_name = f"relevance-misspecification-{args.magnitude}"


class SyntheticExp_Misspecification(GeneralExperiment):
    def __init__(self, candidate_num, slot_list, group_num, assigned_group, magnitude, train_n, test_n) -> None:
        super().__init__()
        self.slot_total_number = slot_list.sum()
        self.slot_list = slot_list

        group_assignment_obj = BinomialGroupAssignment(candidate_num, group_num, assigned_group)
        self.group_assignment = group_assignment_obj.assignment
        self.candidate_list = group_assignment_obj.candidate_list
        self.candidate_num = group_assignment_obj.candidate_num
        self.group_num = group_assignment_obj.group_num
        self.candidate_group_matrix = group_assignment_obj.candidate_group_matrix

        def pr_assigner(i):
            noise_maker = lambda: 0.1 * np.random.randn()
            pr = {
                'xs' : 0.1 + (0.03 * np.arange(group_num))[self.candidate_group_matrix[i].nonzero()[0]],
                's' : 0.2 + (0.03 * np.arange(group_num))[self.candidate_group_matrix[i].nonzero()[0]],
                'l' : 0.4 + (0.03 * np.arange(group_num))[self.candidate_group_matrix[i].nonzero()[0]],
                'xl' : 0.5 + (0.03 * np.arange(group_num))[self.candidate_group_matrix[i].nonzero()[0]],
            }[magnitude]
            correct_single_q = 0.3 + (0.03 * np.arange(group_num))[self.candidate_group_matrix[i].nonzero()[0]]
            correct_pr = np.zeros((assigned_group,))
            wrong_pr = np.zeros((assigned_group,))
            for j in range(assigned_group):
                correct_pr[j] = np.clip(correct_single_q[j] + noise_maker(), 1e-4, 1 - 1e-4)
                wrong_pr[j] = np.clip(pr[j] + noise_maker(), 1e-4, 1 - 1e-4)
            return correct_pr, wrong_pr

        self.correct_pr_assignment = np.zeros((candidate_num, assigned_group))
        self.wrong_pr_assignment = np.zeros((candidate_num, assigned_group))
        for c in range(self.candidate_num):
            correct_pr, wrong_pr = pr_assigner(c)
            self.correct_pr_assignment[c] = correct_pr
            self.wrong_pr_assignment[c] = wrong_pr

        self.build_eligibility_matrix(self.candidate_group_matrix, self.slot_list)
        self.R_train_samples = np.stack([self.eligibility_matrix] * train_n)
        self.R_test_samples = np.stack([self.eligibility_matrix] * test_n)
        self.group_list = np.sum(self.candidate_group_matrix, axis=0)

        for i in range(train_n):
            for c, pr in enumerate(self.wrong_pr_assignment):
                for g, g_slotlist in self.group_slot_idx_map.items(): 
                    connected_edges = self.R_train_samples[i, c, g_slotlist] != 0
                    if np.sum(connected_edges) == 0: continue
                        # with probability equal to 1 - (a's qualification level), drop the edge
                    g_idx = (self.candidate_group_matrix[c].nonzero()[0] == g).nonzero()[0]
                    self.R_train_samples[i, c, g_slotlist] = np.random.binomial(n=1, p=pr[g_idx])

        for i in range(test_n):
            for c, pr in enumerate(self.correct_pr_assignment):
                for g, g_slotlist in self.group_slot_idx_map.items(): 
                    connected_edges = self.R_test_samples[i, c, g_slotlist] != 0
                    if np.sum(connected_edges) == 0: continue
                        # with probability equal to 1 - (a's qualification level), drop the edge
                    g_idx = (self.candidate_group_matrix[c].nonzero()[0] == g).nonzero()[0]
                    self.R_test_samples[i, c, g_slotlist] = np.random.binomial(n=1, p=pr[g_idx])


exp = SyntheticExp_Misspecification(candidate_num, slot_list, total_group,
                   assigned_group, magnitude, train_n, test_n)


match_ranker    = MatchRankRanker(exp.R_train_samples)
match_ranking   = match_ranker.rank_lazy(k=args.greedy_size)
match_rank_name = 'MatchRank'


def compute_test_matching_stats(ranking):
    test_r = len(exp.R_test_samples)
    C = len(ranking)
    matching_scores = np.zeros((test_r, C))
    for i, U in enumerate(exp.R_test_samples):
        sparse_U = scipy.sparse.csr_array(U)
        for j in range(C):
            maximum_matching = scipy.sparse.csgraph.maximum_bipartite_matching(
                sparse_U[ranking[:j + 1], :], perm_type='row')
            score = np.sum(maximum_matching != -1)
            # compute the percentage
            matching_scores[i, j] = score
            if score == slot_list.sum():
                matching_scores[i, j:] = score
                break
    return matching_scores


def compute_avg_min_shortlist(test_score):
    ret = []
    total_slots = slot_list.sum()
    failed = 0
    for S in test_score:
        threshold = np.nonzero(S == total_slots)[0]
        if len(threshold) > 0:
            ret.append((threshold[0] + 1) / total_slots)
        else:
            ret.append(10000)
            failed += 1
    ret = np.array(ret)
    score = str(np.mean(ret)) + ', ' + str(np.std(ret)) + ", " + str(failed)
    return score


match_rank_test_score = compute_test_matching_stats(match_ranking)
print(f'{match_rank_name} {compute_avg_min_shortlist(match_rank_test_score)}')
