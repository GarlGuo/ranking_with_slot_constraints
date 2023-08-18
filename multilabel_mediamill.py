import numpy as np
from multilabel_data import *
from basic_algorithm import *
from greedy_algorithm import *
import numpy as np
import matplotlib.pyplot as plt
import scipy.sparse
import random
import os
import warnings
import argparse


parser = argparse.ArgumentParser(description="ranking with bookmarks dataset")
parser.add_argument(
    "--slots_per_group",
    type=int,
    default=30
)
args = parser.parse_args()


random.seed(0)
np.random.seed(0)

warnings.filterwarnings('ignore')
np.set_printoptions(precision=3)

graph_dir = f'graph{os.sep}multilabel-graph{os.sep}mediamill'

dataset = Mediamill_Dataset(slots_per_label=args.slots_per_group)


test_samples = dataset.test_samples
print(f"candidate_num, group_num: {dataset.test_Y_selected.shape}")
print(
    f"total attributes count: {dataset.attr_cnt}, total labels count: {len(dataset.labels)}")
print()

print(f"selected_label indices: {dataset.selected_label_indices}")
print(
    f"selected_label names: {[dataset.labels[i][0] for i in dataset.selected_label_indices]}")
print()

competition = np.sum(dataset.test_Y_selected, axis=0) / dataset.test_slots
print(f"ground truth competition for each label: {competition}")
empirical_competition = np.zeros(
    (dataset.r, dataset.selected_label_indices_cnt))

for i, U in enumerate(dataset.test_R_samples):
    predicted_pos = np.sum(U, axis=0)
    for slot, label in dataset.ground_truth_slotidx_label_map.items():
        empirical_competition[i, label] = max(
            predicted_pos[slot], empirical_competition[i, label])

avg_empirical_competition = np.mean(
    empirical_competition / dataset.test_slots, axis=0)
print(
    f"avg sampled U matrices competition for each label: {avg_empirical_competition}")
print()

E = np.sum(dataset.test_R_samples, axis=(1, 2))
avg_density_R_sample = np.mean(
    E / (np.sum(dataset.test_slots) * dataset.test_samples))
print(f"avg density of U samples: {avg_density_R_sample:.3f}")  # |E| / |V|
print()

print(f"slots for each label: {dataset.test_slots}")
print(f"ground truth labels count: {np.sum(dataset.test_Y_selected, axis=0)}")
print(
    f"ground truth labels percentage: {100 * np.sum(dataset.test_Y_selected, axis=0) / dataset.test_samples}")

match_ranker         = MatchRankRanker(dataset.test_R_samples)
TR_ranker            = TR_Ranker(dataset.test_R_samples)
NTR_ranker           = NTR_Ranker(dataset.test_R_samples)
relevance_and_ranker = RelevanceRanker(dataset.test_relevance_and)
relevance_or_ranker  = RelevanceRanker(dataset.test_relevance_or)
random_ranker        = RandomRanker(dataset.test_R_samples)

match_ranking        = match_ranker.rank_lazy(k=1200)
TR_ranking           = TR_ranker.rank()
NTR_ranking          = NTR_ranker.rank()
random_ranking       = random_ranker.rank()
PR_and_ranking       = relevance_and_ranker.rank()
PR_or_ranking        = relevance_or_ranker.rank()


match_rank_name      = 'MatchRank'
TR_rank_name         = 'Total Relevance'
NTR_rank_name        = 'Normalized Total Relevance'
PR_AND_rank_name     = 'P(R) - AND'
PR_OR_rank_name      = 'P(R) - OR'
random_rank_name     = 'Random'


match_rank_test_score, match_rank_match_group_comp = dataset.compute_ground_truth_matching_score_and_matched_group_number(match_ranking[:5000])
TR_test_score, TR_match_group_comp                 = dataset.compute_ground_truth_matching_score_and_matched_group_number(TR_ranking[:5000])
NTR_test_score, NTR_match_group_comp               = dataset.compute_ground_truth_matching_score_and_matched_group_number(NTR_ranking[:5000])
PQ_and_test_score, PR_and_match_group_comp         = dataset.compute_ground_truth_matching_score_and_matched_group_number(PR_and_ranking[:5000])
PQ_or_test_score, PR_or_match_group_comp           = dataset.compute_ground_truth_matching_score_and_matched_group_number(PR_or_ranking[:5000])
random_test_score, random_match_group_comp         = dataset.compute_ground_truth_matching_score_and_matched_group_number(random_ranking[:5000])


requirements = dataset.test_slots.copy()
used_set = set()
available_candidates = set(range(dataset.test_samples))
while np.sum(requirements) > 0:
    for i in available_candidates:
        if len(np.where(dataset.test_Y_selected[i, :] == 1)[0]) > 0:
            added = False
            for g in np.where(dataset.test_Y_selected[i, :] == 1)[0]:
                if requirements[g] > 0:
                    requirements[g] -= 1
                    used_set.add(i)
                    available_candidates.remove(i)
                    added = True
                    break
            if added:
                break
print(f"opt = {len(used_set)}")
opt = len(used_set)


def threshold_that_satisfies_slot_constraint(abs_num):
    lowest_num = -np.inf
    for i, g in enumerate(abs_num):
        first_idx = np.where(g >= dataset.test_slots[i])
        if len(first_idx[0]) == 0:
            lowest_num = np.inf  # does not satisfy constraints
            break
        lowest_num = max(lowest_num, first_idx[0][0])
    return lowest_num.astype(np.int32)


def plot_slot_progress(axes, match_group_comp, ranking_name):
    match_abs_num = match_group_comp.T * \
        dataset.test_slots.reshape((len(dataset.test_slots), 1)) / 100
    for g, g_line in enumerate(match_abs_num):
        axes.plot(np.arange(1, len(g_line) + 1), 100 * g_line /
                  dataset.test_slots[g], label=f"g{g + 1}", alpha=0.7, linewidth=1)
    threshold = threshold_that_satisfies_slot_constraint(match_abs_num) + 1
    if threshold != np.inf:
        axes.axvline(x=threshold, color='black', linestyle='dashdot')
        axes.set_xlabel(
            f"min |C|={threshold} ({(threshold / np.sum(dataset.test_slots)).item():.2f} |slots|)", fontsize=8)
        print(f'{ranking_name} {(threshold / np.sum(dataset.test_slots)).item():.2f}')
    else:
        axes.set_xlabel(f"does not fill slots", fontsize=8)
    axes.set_ylim(0, 100)
    axes.set_xlim(1, 3000)
    axes.set_xticks([10, 100, 1000])
    axes.tick_params(axis='both', labelsize=7)
    axes.set_xscale('log')
    axes.set_title(ranking_name, fontsize=10)
    return threshold, match_abs_num[:, threshold]


fig, axes = plt.subplots(1, 6, sharex=True, sharey=True, figsize=(
    14, 4), dpi=400, gridspec_kw=dict(wspace=0.2, left=0.05))
plot_slot_progress(axes[0], match_rank_match_group_comp, match_rank_name)
plot_slot_progress(axes[1], PR_and_match_group_comp, PR_AND_rank_name)
plot_slot_progress(axes[2], PR_or_match_group_comp, PR_OR_rank_name)
plot_slot_progress(axes[3], TR_match_group_comp, TR_rank_name)
plot_slot_progress(axes[4], NTR_match_group_comp, NTR_rank_name)
plot_slot_progress(axes[5], random_match_group_comp, random_rank_name)
axes[5].legend()
fig.supylabel("Percentage of Filled Slots", fontsize=12)
fig.suptitle(
    f"Mediamill Dataset ({args.slots_per_group} Slots Per Label)", fontsize=14)
fig.tight_layout()
fig.savefig(f"{graph_dir}{os.sep}group-ratio-{args.slots_per_group}.png")
