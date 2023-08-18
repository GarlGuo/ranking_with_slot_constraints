import math
import random
from typing import List, Dict
import numpy as np
import abc
import scipy.sparse
import sklearn.model_selection
import scipy.sparse.csgraph


def seed_everything(seed):
    np.random.seed(seed)
    random.seed(seed)


class GroupAssignment(abc.ABC):
    def __init__(self, candidate_num, group_num) -> None:
        super().__init__()
        self.student_num = candidate_num
        self.group_num = group_num
        self.candidate_list = np.arange(candidate_num)
        self.assignment = None

    @property
    def student_group_matrix(self):
        matrix = np.zeros((self.student_num, self.group_num))
        for s, groups in enumerate(self.assignment):
            if len(groups) > 0: matrix[s, np.array(groups)] = 1
        return matrix


class BinomialGroupAssignment(GroupAssignment):
    def __init__(self, student_count, total_group, assign_group) -> None:
        super().__init__(student_count, total_group)
        # sorted use for orderedness in display
        self.assignment = [np.random.RandomState(i).choice(np.arange(total_group), size=assign_group, replace=False) for i in range(student_count)] # the last one is an empty array
        np.random.RandomState(0).shuffle(self.assignment)


# multiple group structure: major (cs group, chem group) + extracur (orchestra, hockey team)
# objective func. Meta group settings (cs major + hockey team) (cs + chem)
class GeneralExperiment(abc.ABC):
    def __init__(self) -> None:
        super().__init__()
        self.R_samples_biadj_list = []
    
    def build_eligibility_matrix(self, candidate_group_matrix, slot_list):
        self.eligibility_matrix = np.zeros((self.candidate_num, np.sum(slot_list)))
        self.group_slot_idx_map = dict()
        for slot_idx, slot_cnt in enumerate(slot_list):
            start = np.sum(slot_list[:slot_idx])
            slot_idx_list = np.arange(start, start + slot_cnt)
            self.eligibility_matrix[..., slot_idx_list] = np.repeat(candidate_group_matrix[..., slot_idx], slot_cnt) \
                .reshape(self.eligibility_matrix[..., slot_idx_list].shape)
            self.group_slot_idx_map[slot_idx] = slot_idx_list

    def sample_edges_given_community(self, p=0):
        # repeat the referenced student slot matrix for r times
        self.U_samples = np.stack([self.eligibility_matrix] * self.r)

        for a in self.applicant_list:
            group_for_a = np.nonzero(self.student_group_matrix[a, :])
            for g, g_slotlist in self.group_slot_idx_map.items():
                if g in group_for_a: continue
                # otherwise, connect the unaffiliated group with an edge with prob p
                self.U_samples[:, a, g_slotlist] = np.random.binomial(n=1, p=p, size=(self.r, len(g_slotlist)))

    def sample_edges_given_community_group_based(self, p_dict=0):
        # repeat the referenced student slot matrix for r times
        self.U_samples = np.stack([self.eligibility_matrix] * self.r)

        for a in self.applicant_list:
            group_for_a = np.nonzero(self.student_group_matrix[a, :])
            for g, g_slotlist in self.group_slot_idx_map.items():
                if g in group_for_a: continue
                self.U_samples[:, a, g_slotlist] = np.random.binomial(n=1, p=p_dict[g], size=(self.r, len(g_slotlist)))

    def apply_per_group_qualification_filter(self):
        for i in range(self.r):
            for a, q in enumerate(self.q_assignment):
                for g, g_slotlist in self.group_slot_idx_map.items(): 
                    connected_edges = self.U_samples[i, a, g_slotlist] != 0
                    if np.sum(connected_edges) == 0: continue
                        # with probability equal to 1 - (a's qualification level), drop the edge
                    self.U_samples[i, a, g_slotlist] = np.random.binomial(n=1, p=q[g])

    def apply_per_applicant_qualification_filter(self):
        for i in range(self.r):
            for a, q in enumerate(self.q_assignment):
                connected_edges = (self.U_samples[i, a, :] != 0)
                if np.sum(connected_edges) == 0: continue
                # with probability equal to 1 - (a's qualification level), drop all of a's edge
                self.U_samples[i, a, :] = np.random.binomial(n=1, p=q) # only one coin

    def train_test_split(self, train_size=0.8):
        U_train, U_test = sklearn.model_selection.train_test_split(self.U_samples, train_size=train_size)
        self.Q_train_samples = U_train
        self.U_test_samples = U_test
