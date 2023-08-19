import numpy as np
import scipy.sparse
import os
import pandas as pd
from collections import Counter
from general_experiment import GeneralExperiment


class AdmissionExp(GeneralExperiment):
    def __init__(self, n, slot_cnt=30, slot_num_low_limit_ratio=0.3, R_max_clip=0.3) -> None:
        super().__init__()
        np.random.seed(0)
        df = pd.read_csv(f'data{os.sep}matching_dataset.csv')

        outcome = df["Bin_final"]
        admitted_outcome = np.zeros((len(outcome),), dtype=np.int32)
        admitted_outcome[(outcome == 'Admit') | (outcome == 'Admit - Conditional')] = 1

        major_map = { m : i for i, m in enumerate(pd.unique(df['Major']))}

        admit_counter = Counter(df['Major'][admitted_outcome == 1])
        slot_map = {m : min(int(slot_num_low_limit_ratio * admit_counter[m]), slot_cnt) for m in major_map.keys()}

        self.n = n
        self.major_map = major_map
        self.slot_map = slot_map
      
        self.num_major = len(self.major_map)
        self.num_slots = sum(self.slot_map.values())
        self.candidate_num = len(df)

        PR = np.array(df["Model 1 pred"])
        PR = np.clip(PR, 1e-3, R_max_clip)

        self.candidate_major_matrix = np.zeros((self.candidate_num, self.num_major), dtype=np.int32)
        self.pr_assignment          = np.zeros((self.candidate_num,))
        self.test_Y                 = np.zeros((self.candidate_num,), dtype=np.int32)
        for i, line in df.iterrows():
            # if line['Major'] in excluded_major_list: continue
            self.candidate_major_matrix[i, self.major_map[line['Major']]] = 1
            self.pr_assignment[i] = PR[i]
            self.test_Y[i]       = admitted_outcome[i]
                
        self.slot_list = np.zeros((self.num_major,), dtype=np.int32)
        for m, idx in self.major_map.items():
            self.slot_list[idx] = self.slot_map[m]

        self.build_eligibility_matrix(self.candidate_major_matrix, self.slot_list)
        self.R_samples = []

        for i in range(self.n):
            single_R_sample = np.zeros_like(self.eligibility_matrix)
            for c, pr in enumerate(self.pr_assignment):
                connected_edges = (self.eligibility_matrix[c, :] != 0)
                # with probability equal to 1 - (c's qualification level), drop all of c's edge
                single_R_sample[c, :][connected_edges] = np.random.binomial(n=1, p=pr) # only one coin
            single_R_sample = scipy.sparse.csr_matrix(single_R_sample)
            self.R_samples.append(single_R_sample)

        self.test_R = self.eligibility_matrix.copy()
        self.test_R[admitted_outcome == 0, :] = 0
        
    def compute_test_matching_score_and_groups(self, ranking):
        slot_idx_group_map = dict()
        for g, g_slotlist in self.group_slot_idx_map.items():
            for s in g_slotlist:
                slot_idx_group_map[s] = g

        C = len(ranking)
        num_major = len(self.major_map)
        matching_scores = np.zeros((C))
        matched_groups = np.zeros((C))
        matched_groups.fill(-1)
        matched_groups_count = np.zeros((C, num_major))
        matched_groups_composition = np.zeros((C, num_major))
        matched_groups_R = np.zeros((C, num_major))

        previous_match_slots = set()
        sparse_R = scipy.sparse.csr_array(self.test_R)
        for j in range(C):
            maximum_matching = scipy.sparse.csgraph.maximum_bipartite_matching(sparse_R[ranking[:j + 1], :], perm_type='row')
            matched_slots = np.where(maximum_matching != -1)[0]
            cur_match_slots = set()
            for m_idx in matched_slots:
                if m_idx == -1: continue
                matched_groups_count[j, slot_idx_group_map[m_idx]] += 1
                cur_match_slots.add(m_idx)
            if len(previous_match_slots) < len(cur_match_slots):
                g = slot_idx_group_map[list(cur_match_slots - previous_match_slots)[0]]
                matched_groups[j] = g
                matched_groups_R[j, g] = self.pr_assignment[ranking[j]]
            previous_match_slots = cur_match_slots
            # compute the percentage
            matched_groups_composition[j] = 100 * matched_groups_count[j] / self.slot_list
            matching_scores[j] = np.sum(maximum_matching != -1)
        return matching_scores, matched_groups, matched_groups_composition, matched_groups_R

    def get_train_matching_scores(self, ranking):
        R_samples_biadj_list = [scipy.sparse.csr_matrix(R) for R in self.R_samples]
        scores = np.zeros_like(ranking, dtype=np.float32)
        for i, _ in enumerate(ranking):
            for R_j in R_samples_biadj_list:
                matching_result = scipy.sparse.csgraph.maximum_bipartite_matching(R_j[ranking[:i + 1]], perm_type='row')
                scores[i] += np.sum(matching_result != -1)
            scores[i] /= len(R_samples_biadj_list)
        return scores