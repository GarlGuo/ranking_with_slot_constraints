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


parser = argparse.ArgumentParser(description="ranking with synthetic dataset")
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
parser.add_argument(
    "--other_k",
    type=int,
    default=10000,
)
parser.add_argument(
    "--ranker",
    type=str,
    default=None,
    choices=[
        'matchrank',
        'prand',
        'pror',
        'tr',
        'ntr',
        'random'
    ]
)
args = parser.parse_args()
print(vars(args))

seed_everything(0)

assigned_group = 2
total_group = 10
slots_per_group = 50
student_num = 10000
magnitude = args.magnitude
train_r = 200
test_r = 1000
slot_list = np.array([slots_per_group] * total_group, dtype=np.int32)

basic_name = f"ou-{args.magnitude}"

def save_pickle_file(obj, name):
    with open(f'algo-data{os.sep}synthetic{os.sep}{basic_name}-{name}.pickle', 'wb') as f:
        pickle.dump(obj, f)


def load_pickle_file(name):
    with open(f'algo-data{os.sep}synthetic{os.sep}{basic_name}-{name}.pickle', 'rb') as f:
        return pickle.load(f)


class SyntheticExp_Misspecification(GeneralExperiment):
    def __init__(self, candidate_num, slot_list, group_num, assigned_group, magnitude, train_r, test_r) -> None:
        super().__init__()
        self.slot_total_number = slot_list.sum()
        self.slot_list = slot_list

        group_assignment_obj = BinomialGroupAssignment(candidate_num, group_num, assigned_group)
        self.group_assignment = group_assignment_obj.assignment
        self.applicant_list = group_assignment_obj.candidate_list
        self.candidate_num = group_assignment_obj.student_num
        self.group_num = group_assignment_obj.group_num
        self.student_group_matrix = group_assignment_obj.student_group_matrix

        def q_assigner(i):
            noise_maker = lambda: 0.1 * np.random.randn()
            q = {
                'xs' : 0.1 + (0.03 * np.arange(group_num))[self.student_group_matrix[i].nonzero()[0]],
                's' : 0.2 + (0.03 * np.arange(group_num))[self.student_group_matrix[i].nonzero()[0]],
                'l' : 0.4 + (0.03 * np.arange(group_num))[self.student_group_matrix[i].nonzero()[0]],
                'xl' : 0.5 + (0.03 * np.arange(group_num))[self.student_group_matrix[i].nonzero()[0]],
            }[magnitude]
            correct_single_q = 0.3 + (0.03 * np.arange(group_num))[self.student_group_matrix[i].nonzero()[0]]
            correct_q = np.zeros((assigned_group,))
            wrong_q = np.zeros((assigned_group,))
            for j in range(assigned_group):
                correct_q[j] = np.clip(correct_single_q[j] + noise_maker(), 1e-4, 1 - 1e-4)
                wrong_q[j] = np.clip(q[j] + noise_maker(), 1e-4, 1 - 1e-4)
            return correct_q, wrong_q

        self.correct_q_assignment = np.zeros((candidate_num, assigned_group))
        self.wrong_q_assignment = np.zeros((candidate_num, assigned_group))
        for a in range(self.candidate_num):
            correct_q, wrong_q = q_assigner(a)
            self.correct_q_assignment[a] = correct_q
            self.wrong_q_assignment[a] = wrong_q

        self.build_eligibility_matrix(self.student_group_matrix, self.slot_list)
        self.R_train_samples = np.stack([self.eligibility_matrix] * train_r)
        self.R_test_samples = np.stack([self.eligibility_matrix] * test_r)
        self.group_list = np.sum(self.student_group_matrix, axis=0)

        for i in range(train_r):
            for a, q in enumerate(self.wrong_q_assignment):
                for g, g_slotlist in self.group_slot_idx_map.items(): 
                    connected_edges = self.R_train_samples[i, a, g_slotlist] != 0
                    if np.sum(connected_edges) == 0: continue
                        # with probability equal to 1 - (a's qualification level), drop the edge
                    g_idx = (self.student_group_matrix[a].nonzero()[0] == g).nonzero()[0]
                    self.R_train_samples[i, a, g_slotlist] = np.random.binomial(n=1, p=q[g_idx])

        for i in range(test_r):
            for a, q in enumerate(self.correct_q_assignment):
                for g, g_slotlist in self.group_slot_idx_map.items(): 
                    connected_edges = self.R_test_samples[i, a, g_slotlist] != 0
                    if np.sum(connected_edges) == 0: continue
                        # with probability equal to 1 - (a's qualification level), drop the edge
                    g_idx = (self.student_group_matrix[a].nonzero()[0] == g).nonzero()[0]
                    self.R_test_samples[i, a, g_slotlist] = np.random.binomial(n=1, p=q[g_idx])


exp = SyntheticExp_Misspecification(student_num, slot_list, total_group,
                   assigned_group, magnitude, train_r, test_r)


match_ranker = MatchRankRanker(exp.R_train_samples)
random_ranker = RandomRanker(exp.R_train_samples)
if assigned_group == 1:
    PR_ranker = RelevanceRanker(exp.wrong_q_assignment)
    PR_and_ranker = PR_ranker
    PR_or_ranker = PR_ranker
else:
    PR_and_ranker = RelevanceRanker(
        np.sum(np.log(exp.wrong_q_assignment), axis=1))
    PR_or_ranker = RelevanceRanker(
        np.sum(-np.log(1 - exp.wrong_q_assignment), axis=1))
    PR_ranker = PR_and_ranker
TR_ranker = TR_Ranker(exp.R_train_samples)
NTR_ranker = NTR_Ranker(exp.R_train_samples)

TR_ranking = TR_ranker.rank()
NTR_ranking = NTR_ranker.rank()
random_ranking = random_ranker.rank()
if assigned_group == 1:
    PR_ranking = PR_ranker.rank()
    PR_and_ranking = PR_ranking
    PR_or_ranking = PR_ranking
else:
    PR_and_ranking = PR_and_ranker.rank()
    PR_or_ranking = PR_or_ranker.rank()
    PR_ranking = PR_and_ranking
match_ranking = match_ranker.rank_lazy(k=args.greedy_size)


greedy_k = args.greedy_size
other_k = args.other_k


match_rank_name = 'MatchRank'
TR_rank_name = 'Total Relevance'
NTR_rank_name = 'Normalized Total Relevance'
PR_rank_name = 'P(R)'
PR_and_rank_name = 'P(R) - AND'
PR_or_rank_name = 'P(R) - OR'
random_rank_name = 'Random'


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


match_rank_test_score = compute_test_matching_stats(match_ranking)
if assigned_group == 1:
    PR_test_score = compute_test_matching_stats(PR_ranking)
    PR_and_test_score = PR_test_score
    PR_or_test_score = PR_test_score
else:
    PR_and_test_score = compute_test_matching_stats(PR_and_ranking)
    PR_or_test_score = compute_test_matching_stats(PR_or_ranking)
    PR_test_score = PR_and_test_score

TR_test_score = compute_test_matching_stats(TR_ranking)
NTR_test_score = compute_test_matching_stats(NTR_ranking)
random_test_score = compute_test_matching_stats(random_ranking)

save_pickle_file(match_rank_test_score,  'match_rank_test_score')
if assigned_group == 1:
    save_pickle_file(PR_test_score,  'PR_test_score')
else:
    save_pickle_file(PR_and_test_score,  'PR_and_test_score')
    save_pickle_file(PR_or_test_score,  'PR_or_test_score')

save_pickle_file(TR_test_score,  'TR_test_score')
save_pickle_file(NTR_test_score,  'NTR_test_score')
save_pickle_file(random_test_score,  'random_test_score')


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


print(f'{match_rank_name} {compute_avg_min_shortlist(match_rank_test_score)}')
if assigned_group == 1:
    print(f'{PR_rank_name} {compute_avg_min_shortlist(PR_test_score)}')
else:
    print(f'{PR_and_rank_name} {compute_avg_min_shortlist(PR_and_test_score)}')
    print(f'{PR_or_rank_name} {compute_avg_min_shortlist(PR_or_test_score)}')
print(f'{TR_rank_name} {compute_avg_min_shortlist(TR_test_score)}')
print(f'{NTR_rank_name} {compute_avg_min_shortlist(NTR_test_score)}')
print(f'{random_rank_name} {compute_avg_min_shortlist(random_test_score)}')
