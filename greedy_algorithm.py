from typing import List
import numpy as np
from tqdm import tqdm
from basic_algorithm import Ranker
import scipy.optimize
import scipy.sparse
import scipy.sparse.csgraph
import heapq
import copy
import math
import random


class MaxPriorityQueue:

    def __init__(self) -> None:
        self.pq = []

    def add(self, v, idx):
        heapq.heappush(self.pq, (-v, idx))
    
    def pop_max(self):
        neg_v, idx = heapq.heappop(self.pq)
        return (-neg_v, idx)

    def peek_max(self):
        neg_v, idx = self.pq[0]
        return (-neg_v, idx)

    def __len__(self):
        return len(self.pq)


class MatchRankRanker(Ranker):
    def __init__(self, R_samples) -> None:
        super().__init__(R_samples)
        self.R_sparse_list = [scipy.sparse.csr_matrix(Ri) for Ri in R_samples]
        # greedy algo takes a long time and the progress counter will display current progress

    def matching_size(self, sparse_biadj_graph):
        matched_slots = scipy.sparse.csgraph.maximum_bipartite_matching(sparse_biadj_graph, perm_type='row')
        return np.sum(matched_slots != -1)

    def evaluate_matching_size(self, app_list):
        matching_score = 0
        for i in range(self.n): 
            matching_score += self.matching_size(self.R_sparse_list[i][app_list, :])
        matching_score /= self.n
        return matching_score

    def rank(self):
        raise NotImplementedError()

    def new_short_list(self, existing_ranking, cand):
        new_list = copy.deepcopy(existing_ranking)
        new_list.append(cand)
        return new_list

    def rank_lazy(self, k=None) -> np.ndarray:
        ranking = []
        pq = MaxPriorityQueue()
        
        if k is None: k = self.candidate_num
        counter = tqdm(range(k), mininterval=5)

        for j in range(self.candidate_num): # loop over all candidates
            pq.add(self.evaluate_matching_size([j]), j)
        
        first_best_score, first_best_candidate = pq.pop_max()
        counter.update(1)
        ranking.append(first_best_candidate)
        cur_ranking_score = first_best_score

        while len(ranking) < k and len(pq) > 0:
            _, prev_top_stale_candidate = pq.pop_max() # pop the top stale candidate

            new_shortlist_contain_top_stale_candidate = self.new_short_list(ranking, prev_top_stale_candidate)
            ranking_include_top_stale_candidate_score = self.evaluate_matching_size(new_shortlist_contain_top_stale_candidate)
            top_fresh_gain = ranking_include_top_stale_candidate_score - cur_ranking_score
            
            pq.add(top_fresh_gain, prev_top_stale_candidate) # reinsert the top stale candidate back with its fresh gain
            _, cur_stale_candidate = pq.peek_max() 

            if cur_stale_candidate == prev_top_stale_candidate:  # the top stale candidate remains on the top
                ranking = new_shortlist_contain_top_stale_candidate
                cur_ranking_score = ranking_include_top_stale_candidate_score
                pq.pop_max()
                counter.update(1)

        return np.array(ranking)


# adapt from stochastic greedy algorithm introduced in https://arxiv.org/pdf/1409.7938.pdf
# we will use eps=0.1 and 0.9
class StochasticGreedyRanker(Ranker):
    # 0.1, 0.3, 0.5
    def __init__(self, R_samples, eps=0.1) -> None:
        super().__init__(R_samples)
        self.eps = eps
        if isinstance(R_samples, np.ndarray):
            self.R_sparse_list = [scipy.sparse.csr_matrix(Ri) for Ri in R_samples]
        else:
            self.R_sparse_list = R_samples

    def matching_size(self, sparse_biadj_graph):
        matched_slots = scipy.sparse.csgraph.maximum_bipartite_matching(sparse_biadj_graph, perm_type='row')
        return np.sum(matched_slots != -1)

    def evaluate_matching_size(self, candidate_list):
        matching_score = 0
        for i in range(self.n): 
            matching_score += self.matching_size(self.R_sparse_list[i][candidate_list, :])
        matching_score /= self.n
        return matching_score

    def get_stochastic_sample(self, k, available_candidates):
        n = self.candidate_num
        sample_num = min(int(math.ceil((n / k) * math.log(1 / self.eps))), len(available_candidates)) # n/k * log(1/epsilon)
        return np.random.choice(np.array(list(available_candidates)), size=sample_num, replace=False).tolist()

    def rank(self, k=None) -> np.ndarray:
        random.seed(0)
        np.random.seed(0)

        ranking = []
        available_candidate_set = set(range(self.candidate_num))

        if k is None: k = self.candidate_num
        counter = tqdm(range(k))

        while len(ranking) < k and len(available_candidate_set) > 0:
            sampled_ground_truth = self.get_stochastic_sample(k, available_candidate_set)

            best_score_sofar = -1
            best_candidate_sofar = None
            for c in sampled_ground_truth: 
                candidate_score = self.evaluate_matching_size(self.union(ranking, c))
                if best_score_sofar < candidate_score:
                    best_candidate_sofar = c 
                    best_score_sofar = candidate_score
                    best_score_sofar = candidate_score

            ranking.append(best_candidate_sofar)
            available_candidate_set.remove(best_candidate_sofar)
            counter.update(1)
        return np.array(ranking)

    def union(self, existing_ranking, cand):
        new_list = copy.deepcopy(existing_ranking)
        new_list.append(cand)
        return new_list


# select a subset and apply lazy greedy, from http://web.cs.ucla.edu/~baharan/papers/mirzasoleiman15lazier.pdf
class SampleGreedyRanker(Ranker):
    # we will use p=0.1, p=0.3, p=0.5
    def __init__(self, R_samples, p=0.1) -> None:
        super().__init__(R_samples)
        self.p = p
        if isinstance(R_samples, np.ndarray):
            self.R_sparse_list = [scipy.sparse.csr_matrix(Ri) for Ri in R_samples]
        else:
            self.R_sparse_list = R_samples

    def matching_size(self, sparse_biadj_graph):
        matched_slots = scipy.sparse.csgraph.maximum_bipartite_matching(sparse_biadj_graph, perm_type='row')
        return np.sum(matched_slots != -1)

    def evaluate_matching_size(self, candidate_list):
        matching_score = 0
        for i in range(self.n): 
            matching_score += self.matching_size(self.R_sparse_list[i][candidate_list, :])
        matching_score /= self.n
        return matching_score

    def new_short_list(self, existing_ranking, cand):
        new_list = copy.deepcopy(existing_ranking)
        new_list.append(cand)
        return new_list

    def rank(self, k=None) -> np.ndarray:
        ranking = []
        random.seed(0)
        np.random.seed(0)
        
        subset = np.random.permutation(self.candidate_num)[:int(self.p * self.candidate_num)]

        pq = MaxPriorityQueue()
        
        if k is None: k = self.candidate_num

        for j in subset: # lazy greedy on subset
            pq.add(self.evaluate_matching_size([j]), j)
        
        first_best_score, first_best_candidate = pq.pop_max()
        ranking.append(first_best_candidate)
        cur_ranking_score = first_best_score

        while len(ranking) < k and len(pq) > 0:
            _, prev_top_stale_candidate = pq.pop_max()  # pop the top stale candidate

            new_shortlist_contain_top_stale_candidate = self.new_short_list(ranking, prev_top_stale_candidate)
            ranking_include_top_stale_candidate_score = self.evaluate_matching_size(new_shortlist_contain_top_stale_candidate)
            top_fresh_gain = ranking_include_top_stale_candidate_score - cur_ranking_score
            
            pq.add(top_fresh_gain, prev_top_stale_candidate) # reinsert the top stale candidate back with its fresh gain
            _, cur_stale_candidate = pq.peek_max() 

            if cur_stale_candidate == prev_top_stale_candidate: # the top stale candidate remains on the top
                ranking = new_shortlist_contain_top_stale_candidate
                cur_ranking_score = ranking_include_top_stale_candidate_score
                pq.pop_max()           

        return np.array(ranking)


# from https://theory.stanford.edu/~jvondrak/data/submod-fast.pdf
class ThresholdGreedyRanker(Ranker):
    # we will use eps=0.5, eps=0.7, eps=0.9
    def __init__(self, R_samples, eps=0.1) -> None:
        super().__init__(R_samples)
        self.eps = eps
        if isinstance(R_samples, np.ndarray):
            self.R_sparse_list = [scipy.sparse.csr_matrix(Ri) for Ri in R_samples]
        else:
            self.R_sparse_list = R_samples

    def matching_size(self, sparse_biadj_graph):
        matched_slots = scipy.sparse.csgraph.maximum_bipartite_matching(sparse_biadj_graph, perm_type='row')
        return np.sum(matched_slots != -1)

    def evaluate_matching_size(self, candidate_list):
        matching_score = 0
        for i in range(self.n): 
            matching_score += self.matching_size(self.R_sparse_list[i][candidate_list, :])
        matching_score /= self.n
        return matching_score

    def union(self, existing_ranking, c):
        new_list = copy.deepcopy(existing_ranking)
        new_list.append(c)
        return new_list

    def rank(self, k=None) -> np.ndarray:        
        random.seed(0)
        np.random.seed(0)
        
        if k is None: k = self.candidate_num

        counter = tqdm(range(k))
        ranking = []
        available_candidate_set = set(range(self.candidate_num))

        d = max(self.evaluate_matching_size([i]) for i in np.arange(self.candidate_num))
        ranking_score = 0

        w = d
        while w >= self.eps / self.candidate_num * d: # for(w = d; w >= eps / n * d; w = w * (1 - eps))
            if len(ranking) == k or len(available_candidate_set) == 0: break
            for c in np.arange(self.candidate_num):
                if len(ranking) == k: break
                if c not in available_candidate_set: continue
                new_ranking_score = self.evaluate_matching_size(self.union(ranking, c))
                marginal_gain = new_ranking_score - ranking_score
                if marginal_gain >= w:
                    ranking.append(c)
                    available_candidate_set.remove(c)
                    counter.update(1)
                    ranking_score = new_ranking_score
            w = w * (1 - self.eps)

        return np.array(ranking)