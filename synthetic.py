import numpy as np
import sklearn
from basic_algorithm import *
from greedy_algorithm import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
from general_experiment import seed_everything, BinomialGroupAssignment, GeneralExperiment
import argparse


parser = argparse.ArgumentParser(description="ranking with synthetic dataset")
parser.add_argument(
    "--groups",
    type=int,
    default=10,
)
parser.add_argument(
    "--slots",
    type=int,
    default=50,
)
parser.add_argument(
    "--magnitude",
    type=str,
    default='medium',
    choices=['small', 'medium', 'large']
)
parser.add_argument(
    "--eligibility",
    type=int,
    default=2,
)
parser.add_argument(
    "--n",
    type=int,
    default=200,
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
args = parser.parse_args()
print(vars(args))

seed_everything(0)

assigned_group = args.eligibility
total_group = args.groups
slots_per_group = args.slots
candidate_num = 10000
magnitude = args.magnitude
train_n = args.n
test_n = 1000
slot_list = np.array([slots_per_group] * total_group, dtype=np.int32)

basic_name = f"elibility-{args.eligibility}-groups-{args.groups}-slots-{args.slots}-n-{args.n}-magnitude-{args.magnitude}"


class SyntheticExp(GeneralExperiment):
    def __init__(self, candidate_num, slot_list, group_num, assigned_group, magnitude, train_n, test_n) -> None:
        super().__init__()
        self.n = train_n + test_n
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
                'small' : 0.2 + (0.03 * np.arange(group_num))[self.candidate_group_matrix[i].nonzero()[0]],
                'medium': 0.3 + (0.03 * np.arange(group_num))[self.candidate_group_matrix[i].nonzero()[0]],
                'large' : 0.4 + (0.03 * np.arange(group_num))[self.candidate_group_matrix[i].nonzero()[0]],
            }[magnitude]
            if assigned_group != 1:
                ret_pr = np.zeros((assigned_group,))
                for j in range(assigned_group):
                    ret_pr[j] = np.clip(pr[j] + noise_maker(), 1e-4, 1 - 1e-4)
                return ret_pr
            else:
                return np.clip(pr + noise_maker(), 1e-4, 1 - 1e-4)

        self.pr_assignment = np.zeros((candidate_num,)) if assigned_group == 1 else np.zeros((candidate_num, assigned_group))
        for c in range(self.candidate_num):
            self.pr_assignment[c] = pr_assigner(c)

        self.build_eligibility_matrix(self.candidate_group_matrix, self.slot_list)
        R_samples = np.stack([self.eligibility_matrix] * self.n)
        self.group_list = np.sum(self.candidate_group_matrix, axis=0)

        for i in range(self.n):
            if assigned_group == 1:
                for c, pr in enumerate(self.pr_assignment):
                    connected_edges = (R_samples[i, c, :] != 0)
                    # with probability equal to 1 - (a's qualification level), drop all of a's edge
                    R_samples[i, c, :][connected_edges] = np.random.binomial(n=1, p=pr) # only one coin
            else:
                for c, pr in enumerate(self.pr_assignment):
                    for g, g_slotlist in self.group_slot_idx_map.items(): 
                        connected_edges = R_samples[i, c, g_slotlist] != 0
                        if np.sum(connected_edges) == 0: continue
                            # with probability equal to 1 - (a's qualification level), drop the edge
                        g_idx = (self.candidate_group_matrix[c].nonzero()[0] == g).nonzero()[0]
                        R_samples[i, c, g_slotlist] = np.random.binomial(n=1, p=pr[g_idx])

        R_train, R_test = sklearn.model_selection.train_test_split(R_samples, train_size=train_n)
        self.R_train_samples = R_train
        self.R_test_samples = R_test

exp = SyntheticExp(candidate_num, slot_list, total_group, assigned_group, magnitude, train_n, test_n)


match_ranker             = MatchRankRanker(exp.R_train_samples)
random_ranker            = RandomRanker(exp.R_train_samples)
if assigned_group == 1:
    PR_ranker            = RelevanceRanker(exp.pr_assignment)
    PR_and_ranker        = PR_ranker
    PR_or_ranker         = PR_ranker
else:
    PR_and_ranker        = RelevanceRanker(np.sum(np.log(exp.pr_assignment), axis=1))
    PR_or_ranker         = RelevanceRanker(np.sum(-np.log(1 - exp.pr_assignment), axis=1))
    PR_ranker            = PR_and_ranker
TR_ranker                = TR_Ranker(exp.R_train_samples)
NTR_ranker               = NTR_Ranker(exp.R_train_samples)

TR_ranking               = TR_ranker.rank()
NTR_ranking              = NTR_ranker.rank()
random_ranking           = random_ranker.rank()
if assigned_group == 1:
    PR_ranking           = PR_ranker.rank()
    PR_and_ranking       = PR_ranking
    PR_or_ranking        = PR_ranking
else:
    PR_and_ranking       = PR_and_ranker.rank()
    PR_or_ranking        = PR_or_ranker.rank()
    PR_ranking           = PR_and_ranking
match_ranking            = match_ranker.rank_lazy(k=args.greedy_size)


greedy_k = args.greedy_size
other_k = args.other_k


match_rank_name            = 'MatchRank'
TR_rank_name               = 'Total Relevance'
NTR_rank_name              = 'Normalized Total Relevance'
PR_rank_name               = 'P(R)'
PR_and_rank_name           = 'P(R) - AND'
PR_or_rank_name            = 'P(R) - OR'
random_rank_name           = 'Random'


def compute_test_matching_stats(ranking):
    test_r = len(exp.R_test_samples)
    C = len(ranking)
    matching_scores = np.zeros((test_r, C))
    for i, R in enumerate(exp.R_test_samples):
        sparse_R = scipy.sparse.csr_array(R)
        for j in range(C):
            maximum_matching = scipy.sparse.csgraph.maximum_bipartite_matching(sparse_R[ranking[:j + 1], :], perm_type='row')
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
    score =  str(np.mean(ret)) + ', ' + str(np.std(ret)) + ", " + str(failed)
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
