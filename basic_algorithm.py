import numpy as np
from abc import ABC, abstractmethod
import scipy.sparse


class Ranker(ABC):
    def __init__(self, R_samples) -> None:
        super().__init__()
        self.R_samples = R_samples 
        if isinstance(R_samples, list):
            self.n = len(R_samples)
            self.candidate_num, self.slot_num = R_samples[0].shape
        else:
            self.n, self.candidate_num, self.slot_num = R_samples.shape 


    @abstractmethod
    def rank(self) -> np.ndarray:
        pass


# Total relevance ranking
class TR_Ranker(Ranker):
    def __init__(self, R_samples) -> None:
        super().__init__(R_samples)
    
    def rank(self, k=None) -> np.ndarray:
        if k is None: k = self.candidate_num
        if isinstance(self.R_samples, np.ndarray):
            candidate_scores = np.sum(self.R_samples, axis=(0, 2))
        elif isinstance(self.R_samples, list) and isinstance(self.R_samples[0], scipy.sparse.csr_matrix):
            candidate_scores = np.zeros((self.candidate_num,))
            for U in self.R_samples:
                candidate_scores += np.array(U.sum(axis=1)).reshape(-1)
        else:
            raise NotImplementedError()

        return np.argsort(-candidate_scores)[:k]

# normalized total relevance
class NTR_Ranker(Ranker):
    def __init__(self, R_samples) -> None:
        super().__init__(R_samples)
    
    def compute_normalized_total_scores(self, R) -> np.ndarray: # candidate_num, slot_num
        if isinstance(R, np.ndarray):
            num_relevant_candidate_for_each_slot = np.sum(R, axis=0)
            slot_competition_factors = np.zeros_like(num_relevant_candidate_for_each_slot)
            slot_competition_factors[np.nonzero(num_relevant_candidate_for_each_slot)] = 1 / num_relevant_candidate_for_each_slot[np.nonzero(num_relevant_candidate_for_each_slot)]
            return np.sum((slot_competition_factors * R), axis=1)
        elif isinstance(R, scipy.sparse.csr_matrix):
            num_relevant_candidate_for_each_slot = np.array(R.sum(axis=0)).reshape(-1)
            slot_competition_factors = np.zeros((R.shape[1], ))
            slot_competition_factors[np.nonzero(num_relevant_candidate_for_each_slot)] = 1 / num_relevant_candidate_for_each_slot[np.nonzero(num_relevant_candidate_for_each_slot)]
            slot_competition_factors = scipy.sparse.csr_matrix(slot_competition_factors)
            return np.array(np.sum(slot_competition_factors.multiply(R), axis=1)).reshape(-1)
        else:
            raise NotImplementedError()

    def rank(self, k=None) -> np.ndarray:
        if k is None: k = self.candidate_num
        candidates_normalized_scores = np.zeros((self.candidate_num, ))
        for R in self.R_samples: 
            candidates_normalized_scores += self.compute_normalized_total_scores(R)
        return np.argsort(-candidates_normalized_scores)[:k]


# random
class RandomRanker(Ranker):
    def __init__(self, R_samples) -> None:
        super().__init__(R_samples)
    
    def rank(self, k=None) -> np.ndarray:
        if k is None: k = self.candidate_num
        return np.random.RandomState(0).permutation(self.candidate_num)[:k]


# P(R) ranker
class RelevanceRanker(Ranker):
    def __init__(self, q_list) -> None:
        self.r_list = q_list

    def rank(self, k=None) -> np.ndarray:
        if k is None: k = len(self.r_list)
        return np.argsort(-self.r_list)[:k]
