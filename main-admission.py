import numpy as np
from basic_algorithm import *
from greedy_algorithm import *
import numpy as np
from admission_data import *
import argparse


parser = argparse.ArgumentParser(description="ranking with college admission dataset")
parser.add_argument(
    "--slots_count",
    type=int,
    default=50,
)
parser.add_argument(
    "--n",
    type=int,
    default=100,
)
parser.add_argument(
    "--slot_num_low_limit_ratio",
    type=float,
    default=0.7,
)
parser.add_argument(
    "--R_max_clip",
    type=float,
    default=0.3,
)
parser.add_argument(
    "--greedy_size",
    type=int,
    default=1000,
)
args = parser.parse_args()
print(vars(args))


n = args.n
slots = args.slots_count
exp = AdmissionExp(n, slot_cnt=slots, slot_num_low_limit_ratio=args.slot_num_low_limit_ratio, R_max_clip=args.R_max_clip)

greedy_ranker              = MatchRankRanker(exp.R_samples)
TR_ranker                  = TR_Ranker(exp.R_samples)
NTR_ranker                 = NTR_Ranker(exp.R_samples)
random_ranker              = RandomRanker(exp.R_samples)
PR_ranker                  = RelevanceRanker(exp.pr_assignment)

slots_for_candidates = exp.slot_list[exp.candidate_major_matrix.nonzero()[1]]


TR_ranking                 = TR_ranker.rank()
NTR_ranking                = NTR_ranker.rank()
random_ranking             = random_ranker.rank()
PR_ranking                 = PR_ranker.rank()
greedy_ranking             = greedy_ranker.rank_lazy(k=args.greedy_size)


basic_name = f"admission-slots-{slots}-slot_num_low_limit_ratio-{args.slot_num_low_limit_ratio}-R_max_clip-{args.R_max_clip}"


match_rank_test_score  = exp.compute_test_matching_score_and_groups(greedy_ranking)[0]
TR_test_score          = exp.compute_test_matching_score_and_groups(TR_ranking)[0]
NTB_test_score         = exp.compute_test_matching_score_and_groups(NTR_ranking)[0]
PR_test_score          = exp.compute_test_matching_score_and_groups(PR_ranking)[0]
random_test_score      = exp.compute_test_matching_score_and_groups(random_ranking)[0]

match_rank_name        = 'MatchRank'
TB_rank_name           = 'Total Relevance'
NTB_rank_name          = 'Normalized Total Relevance'
PQ_rank_name           = 'P(R)'
random_rank_name       = 'Random'


opt = exp.num_slots
def compute_avg_min_shortlist(test_score):
    return (np.nonzero(test_score == exp.slot_list.sum())[0][0] + 1) / exp.slot_list.sum()


print(f'match rank {compute_avg_min_shortlist(match_rank_test_score)}')
print(f'TR {compute_avg_min_shortlist(TR_test_score)}')
print(f'NTR {compute_avg_min_shortlist(NTB_test_score)}')
print(f'P(R) {compute_avg_min_shortlist(PR_test_score)}')
print(f'random {compute_avg_min_shortlist(random_test_score)}')
