import numpy as np
import scipy.sparse
import skmultilearn
from sklearn.linear_model import LogisticRegression
from skmultilearn.dataset import load_dataset
from sklearn import metrics
import sklearn.svm
import sklearn.neural_network
import warnings
import sklearn.calibration
import os
from sklearn.model_selection import train_test_split
import pickle
import sklearn.datasets
from tqdm.auto import tqdm


warnings.filterwarnings('always')


class MultiLabelDataset:
    def __init__(self, dataset_name, selected_labels=np.arange(10), n=100,
                 binary_model='binary_LR', allocated_slots=None, prob_caliber='sigmoid', 
                 data_source=None, has_fitted_models=False, pos_mask=True, pos_mask_p=0.4) -> None:
        super().__init__()
        self.dataset_name = dataset_name
        self.n = n
        self.selected_label_indices = selected_labels
        self.selected_label_indices_cnt = len(selected_labels)

        if data_source == None:
            np.random.RandomState(0).shuffle(self.selected_label_indices)
            print(f"label indices: {self.selected_label_indices}")
            self.train_X, self.train_Y, self.attributes, self.labels = load_dataset(
                self.dataset_name, 'train')
            self.test_X, self.test_Y, _, _ = load_dataset(
                self.dataset_name, 'test')

            self.train_X, self.train_Y = self.train_X.toarray(), self.train_Y.toarray()
            self.test_X, self.test_Y = self.test_X.toarray(), self.test_Y.toarray()
        else:
            self.train_X, self.train_Y, self.test_X, self.test_Y = data_source

        self.train_samples, self.attr_cnt = self.train_X.shape
        self.test_samples, _ = self.test_X.shape

        if pos_mask:
            print(f"has labels' pos masked, p={pos_mask_p}")
            self.mask_positive_occurrence(pos_mask_p=pos_mask_p)

        self.train_Y_selected = self.train_Y[:, self.selected_label_indices]
        self.test_Y_selected = self.test_Y[:, self.selected_label_indices]

        self.fit_binary_model_each_freq_labels(
            binary_model=binary_model, prob_caliber=prob_caliber, has_fitted_models=has_fitted_models)

        self.test_relevance_or = self.compute_relevance_or(
            self.test_pos_probs)
        self.test_relevance_and = self.compute_relevance_and(
            self.test_pos_probs)

        self.test_slots = np.array(list(allocated_slots[idx] for idx in self.selected_label_indices), dtype=np.int32)
        self.ground_truth_R_reference, self.prob_R, label_slot_idx_map, self.ground_truth_slot_idx_label_map = \
            self.build_eligibility_matrix(
                self.test_Y_selected, self.test_slots)
        self.test_R_samples = self.make_R_samples(
            self.ground_truth_R_reference, label_slot_idx_map, self.test_slots, self.test_pos_probs, self.test_Y_selected)

    def mask_positive_occurrence(self, pos_mask_p=0.4):
        print(f"trainset nnz before masking {self.train_Y.sum()}")
        print(f"testset nnz before masking {self.test_Y.sum()}")
        train_Y_pos = self.train_Y.nonzero()
        train_Y_choices = np.random.RandomState(0).choice(np.array([False, True]), p=[
            1-pos_mask_p, pos_mask_p], size=len(train_Y_pos[0]))
        self.train_Y[(train_Y_pos[0][train_Y_choices],
                      train_Y_pos[1][train_Y_choices])] = 0

        test_Y_pos = self.test_Y.nonzero()
        test_Y_choices = np.random.RandomState(0).choice(np.array([False, True]), p=[
            1-pos_mask_p, pos_mask_p], size=len(test_Y_pos[0]))
        self.test_Y[(test_Y_pos[0][test_Y_choices],
                     test_Y_pos[1][test_Y_choices])] = 0

        print(f"trainset nnz after masking {self.train_Y.sum()}")
        print(f"testset nnz after masking {self.test_Y.sum()}")

    def random_feature_drop(self, drop_p=0.3):
        keep_loc = np.random.RandomState(0).choice(np.array([True, False]), p=[
            1-drop_p, drop_p], size=self.train_X.shape[1])
        self.train_X = self.train_X[:, keep_loc]
        self.features = self.features[keep_loc]
        self.test_X = self.test_X[:, keep_loc]

    # 1 - prod_j (1 - P(ij != 1)), equivalent impl
    def compute_relevance_or(self, probs):
        probs[probs > (1 - 1e-10)] = 1 - 1e-10
        return -np.sum(np.log1p(-probs), axis=1)

    # prod_j P(ij = 1), equivalent impl
    def compute_relevance_and(self, probs):
        probs[probs < 1e-20] = 1e-20
        return np.sum(np.log10(probs), axis=1)

    def compute_avg_matching_size(self, algo_matching):
        matching_score = []
        sparse_R = [scipy.sparse.csr_array(R) for R in self.test_R_samples]
        for i in range(len(algo_matching)):
            scores = 0
            for R in sparse_R:
                scores += self.get_maximum_bipartite_matching_size(
                    R[algo_matching[:i + 1], :])
            matching_score.append(scores / len(sparse_R))
        return matching_score

    # compute the percentage of ranked candidates in each group
    def compute_ground_truth_matching_score_and_matched_group_number(self, algo_matching):
        matching_score = []
        matched_group_composition = np.zeros(
            (len(algo_matching), self.selected_label_indices_cnt))
        sparse_U_ref = scipy.sparse.csr_array(self.ground_truth_R_reference)
        for i in range(len(algo_matching)):
            maximum_matching = scipy.sparse.csgraph.maximum_bipartite_matching(
                sparse_U_ref[algo_matching[:i + 1], :], perm_type='col')
            matched_slots = np.where(maximum_matching != -1)[0]
            for m_idx in matched_slots:
                matched_group_composition[i,
                                          self.ground_truth_slot_idx_label_map[m_idx]] += 1
            # compute the percentage
            matched_group_composition[i] = 100 * \
                matched_group_composition[i] / self.test_slots
            matching_score.append(len(matched_slots))
        return matching_score, matched_group_composition

    # compute the absolute number of ranked candidate in each group
    # if one candidate belongs to multiple groups, the membership will be evenly divided
    def compute_group_truth_group_number_with_fraction(self, algo_matching):
        return np.cumsum(self.test_Y_selected_fraction[algo_matching], axis=0)

    def build_eligibility_matrix(self, Y, slot_list):
        eligibility_Y = np.zeros((Y.shape[0], np.sum(slot_list)))
        prob_R = np.zeros((Y.shape[0], np.sum(slot_list)))
        label_slot_idx_map = dict()
        slot_idx_label_map = dict()
        for slot_idx, slot_cnt in enumerate(slot_list):
            start = np.sum(slot_list[:slot_idx])
            slot_idx_list = np.arange(start, start + slot_cnt)
            if isinstance(Y, np.ndarray):
                eligibility_Y[..., slot_idx_list] = np.repeat(
                    Y[..., slot_idx], slot_cnt).reshape(eligibility_Y[..., slot_idx_list].shape)
            else:
                eligibility_Y[..., slot_idx_list] = np.repeat(Y[..., slot_idx].toarray(), slot_cnt)\
                    .reshape(eligibility_Y[..., slot_idx_list].shape)
            prob_R[..., slot_idx_list] = np.repeat(
                self.test_pos_probs[..., slot_idx], slot_cnt).reshape(prob_R[..., slot_idx_list].shape)
            label_slot_idx_map[slot_idx] = slot_idx_list
            for s in slot_idx_list:
                slot_idx_label_map[s] = slot_idx
        return eligibility_Y, prob_R, label_slot_idx_map, slot_idx_label_map

    def get_positive_examples_of_reference(self, Y):
        return np.sum(Y, axis=0)

    def get_positive_examples_of_predicted(self, X):
        labels = np.zeros((self.selected_label_indices_cnt,), dtype=np.int32)
        for i in range(self.selected_label_indices_cnt):
            labels[i] = np.sum(self.fitted_binary_model[i].predict(X))
        return labels

    def make_R_samples(self, eligibility_mat, label_slot_idx_map, slot_list, probs, Y):
        if isinstance(Y, np.ndarray):
            is_sparse = False
            R_samples = np.zeros((self.n, *eligibility_mat.shape))
        else:
            is_sparse = True
            R_samples = []
        if not is_sparse:
            for i in range(Y.shape[0]):
                for j in range(self.selected_label_indices_cnt):
                    slot_idx = label_slot_idx_map[j]
                    coins = np.random.binomial(1, p=probs[i, j], size=self.n)
                    slots = np.zeros((self.n, len(slot_idx)))
                    slots[:, np.arange(slots.shape[1])] = coins.reshape(len(coins), 1)
                    R_samples[:, i, slot_idx] = slots
        else:
            for _ in range(self.n):
                R = np.zeros((*eligibility_mat.shape,))
                R_coined_flip = np.random.binomial(1, p=probs)
                for j in range(self.selected_label_indices_cnt):
                    R[:, label_slot_idx_map[j]] = R_coined_flip[:, j]\
                        .reshape((len(R_coined_flip[:, j]), 1))
                R_samples.append(scipy.sparse.csr_matrix(R))
        return R_samples

    def empty_binary_MLP(self):
        return sklearn.neural_network.MLPClassifier(random_state=0, max_iter=1000, hidden_layer_sizes=(200, 200))

    def empty_binary_LR(self):
        return LogisticRegression(penalty='l2', C=5, max_iter=1000, random_state=0)

    def empty_binary_LR_large(self):
        return LogisticRegression(penalty='l2', C=5, max_iter=100, random_state=0, solver='sag')

    def empty_binary_SVC(self):
        return sklearn.svm.SVC(C=1, random_state=0, probability=True)

    def get_binary_model(self, binary_model):
        return {
            'binary_LR_large': lambda: self.empty_binary_LR_large(),
            'binary_LR': lambda: self.empty_binary_LR(),
            'binary_SVC': lambda: self.empty_binary_SVC(),
            'binary_MLP': lambda: self.empty_binary_MLP()
        }[binary_model]()

    def fit_binary_model_each_freq_labels(self, binary_model='binary_LR', prob_caliber='sigmoid', has_fitted_models=False):
        if not has_fitted_models:
            self.fitted_binary_model = [sklearn.calibration.CalibratedClassifierCV(self.get_binary_model(
                binary_model), cv=5, method=prob_caliber, ensemble=False) for _ in range(self.selected_label_indices_cnt)]
        self.train_pos_probs = np.zeros(
            (*self.train_Y_selected.shape,), dtype=np.float64)
        self.test_pos_probs = np.zeros(
            (*self.test_Y_selected.shape,), dtype=np.float64)

        train_scores = [None] * self.selected_label_indices_cnt
        test_scores = [None] * self.selected_label_indices_cnt

        for i in range(self.selected_label_indices_cnt):
            if not has_fitted_models:
                model = self.fitted_binary_model[i]
                if isinstance(self.train_Y_selected[:, i], np.ndarray):
                    model.fit(self.train_X, self.train_Y_selected[:, i])
                else:
                    model.fit(
                        self.train_X, self.train_Y_selected[:, i].toarray().reshape(-1))
            else:
                model = self.fitted_binary_model[self.selected_label_indices[i]]

            train_preds = model.predict(self.train_X)
            if isinstance(self.train_Y_selected[:, i], np.ndarray):
                train_scores[i] = metrics.precision_recall_fscore_support(
                    self.train_Y_selected[:, i], train_preds, average='binary')
            else:
                train_ref = np.lib.stride_tricks.as_strided(
                    self.train_Y_selected[:, i].toarray(), shape=(self.train_samples,))
                train_scores[i] = metrics.precision_recall_fscore_support(
                    train_ref, train_preds, average='binary')

            test_preds = model.predict(self.test_X)
            if isinstance(self.train_Y_selected[:, i], np.ndarray):
                test_scores[i] = metrics.precision_recall_fscore_support(
                    self.test_Y_selected[:, i], test_preds, average='binary')
            else:
                test_ref = np.lib.stride_tricks.as_strided(
                    self.test_Y_selected[:, i].toarray(), shape=(self.test_samples,))
                test_scores[i] = metrics.precision_recall_fscore_support(
                    test_ref, test_preds, average='binary')

            # [:, 1] means the proba for positive class
            self.train_pos_probs[:, i] = model.predict_proba(self.train_X)[
                :, 1]
            self.test_pos_probs[:, i] = model.predict_proba(self.test_X)[:, 1]

        print(f"train precision : {[round(p[0], 3) for p in train_scores]}")
        print(f"test precision : {[round(p[0], 3) for p in test_scores]}")

        print(f"train recall : {[round(p[1], 3) for p in train_scores]}")
        print(f"test recall : {[round(p[1], 3) for p in test_scores]}")

        print(f"train F-score : {[round(p[2], 3) for p in train_scores]}")
        print(f"test F-score : {[round(p[2], 3) for p in test_scores]}")

    def get_maximum_bipartite_matching_size(self, sparse_R_ref):
        return np.sum(scipy.sparse.csgraph.maximum_bipartite_matching(sparse_R_ref, perm_type='row') != -1)

    def columnwise_mask_(self, Y, mask_p=0.2):
        Y_nnz = Y.nonzero()
        Y_choices = np.random.choice(np.array([False, True]), p=[
                                     1-mask_p, mask_p], size=len(Y_nnz[0]))
        Y_nnz_copy = Y[Y_nnz].copy()
        Y_nnz_copy[Y_choices] = 0
        Y[Y_nnz] = Y_nnz_copy
        return Y


class TMC2007_Dataset(MultiLabelDataset):
    def __init__(self, selected_labels=[11, 18, 13, 12, 21, 17, 4, 1, 7, 5], r=100, slots_per_label=30) -> None:
        allocated_slots={k : slots_per_label for k in selected_labels}
        super().__init__('tmc2007_500', selected_labels=selected_labels, n=r,
                         allocated_slots=allocated_slots, prob_caliber='isotonic', pos_mask=True, pos_mask_p=0.2)


class MedicalDataset(MultiLabelDataset):
    def __init__(self, n=100, slots_per_label=5) -> None:
        selected_labels = [4, 32, 0, 9, 41, 31, 24, 31, 23, 44]
        allocated_slots = {k: slots_per_label for k in selected_labels}
        self.dataset_name = 'medical'
        self.n = n
        self.selected_label_indices = selected_labels
        self.selected_label_indices_cnt = len(selected_labels)

        np.random.RandomState(0).shuffle(self.selected_label_indices)
        print(f"label indices: {self.selected_label_indices}")
        self.train_X, self.train_Y, self.attributes, self.labels = load_dataset(
            self.dataset_name, 'train')
        self.test_X, self.test_Y, _, _ = load_dataset(
            self.dataset_name, 'test')

        self.train_Y, self.test_Y = self.train_Y.toarray(), self.test_Y.toarray()

        self.train_samples, self.attr_cnt = self.train_X.shape
        self.test_samples, _ = self.test_X.shape

        print(f"trainset nnz before masking {self.train_Y.sum()}")
        print(f"testset nnz before masking {self.test_Y.sum()}")
        self.train_Y_selected = self.train_Y[:, self.selected_label_indices]
        self.test_Y_selected = self.test_Y[:, self.selected_label_indices]

        np.random.seed(0)
        for i in range(self.selected_label_indices_cnt):
            self.train_Y_selected[:, i] = self.columnwise_mask_(
                self.train_Y_selected[:, i], mask_p=0.2)
            self.test_Y_selected[:, i] = self.columnwise_mask_(
                self.test_Y_selected[:, i], mask_p=0.2)

        self.fit_binary_model_each_freq_labels(
            binary_model='binary_LR', prob_caliber='sigmoid', has_fitted_models=False)

        self.test_relevance_or = self.compute_relevance_or(
            self.test_pos_probs)
        self.test_relevance_and = self.compute_relevance_and(
            self.test_pos_probs)

        self.test_slots = np.array(
            list(allocated_slots[i] for i in self.selected_label_indices), dtype=np.int32)
        self.ground_truth_R_reference, self.prob_R, label_slot_idx_map, self.ground_truth_slot_idx_label_map = \
            self.build_eligibility_matrix(
                self.test_Y_selected, self.test_slots)
        self.test_R_samples = self.make_R_samples(
            self.ground_truth_R_reference, label_slot_idx_map, self.test_slots, self.test_pos_probs, self.test_Y_selected)


class BibtexDataset(MultiLabelDataset):
    def __init__(self, pos_mask_p=0.2, n=100, pos_mask=True, slots_per_label=10) -> None:
        # selected_label=[134, 14, 9, 44, 16, 13, 101, 49, 35, 62],
        # allocated_slots = {16: 10, 14: 10, 44: 10, 52: 10,
        #                    63: 10, 83: 10, 117: 10, 131: 10, 134: 30, 13: 10}
        selected_label = [117, 14, 44, 52, 63, 83, 104, 131, 134, 10]
        allocated_slots = {k: slots_per_label for k in selected_label}
        super().__init__('bibtex', selected_labels=selected_label, n=n,
                         pos_mask=pos_mask, pos_mask_p=pos_mask_p, allocated_slots=allocated_slots)


class DeliciousDataset(MultiLabelDataset):
    # 99, 540, 700, 365, 204, 666, 727, 756, 906, 726
    def __init__(self, r=100, pos_mask=True, pos_mask_p=0.2, slots_per_label=30) -> None:
        selected_labels = [946, 941, 924, 897, 809, 700, 733, 540, 452, 99]
        allocated_slots = {k: slots_per_label for k in selected_labels}
        super().__init__('delicious', selected_labels=selected_labels, n=r,
                         pos_mask=pos_mask, pos_mask_p=pos_mask_p, allocated_slots=allocated_slots)


class Mediamill_Dataset(MultiLabelDataset):
    def __init__(self, selected_labels=[67, 65, 78, 2, 84, 66, 96, 51, 24, 94], n=100, slots_per_label=30) -> None:
        allocated_slots = {k : slots_per_label for k in selected_labels}
        self.dataset_name = 'mediamill'
        self.n = n
        self.selected_label_indices = selected_labels
        self.selected_label_indices_cnt = len(selected_labels)

        np.random.RandomState(0).shuffle(self.selected_label_indices)
        print(f"label indices: {self.selected_label_indices}")
        self.train_X, self.train_Y, self.attributes, self.labels = load_dataset(
            self.dataset_name, 'train')
        self.test_X, self.test_Y, _, _ = load_dataset(
            self.dataset_name, 'test')

        self.train_Y, self.test_Y = self.train_Y.toarray(), self.test_Y.toarray()

        self.train_samples, self.attr_cnt = self.train_X.shape
        self.test_samples, _ = self.test_X.shape

        print(f"trainset nnz before masking {self.train_Y.sum()}")
        print(f"testset nnz before masking {self.test_Y.sum()}")
        self.train_Y_selected = self.train_Y[:, self.selected_label_indices]
        self.test_Y_selected = self.test_Y[:, self.selected_label_indices]

        np.random.seed(0)
        for i in range(self.selected_label_indices_cnt):
            self.train_Y_selected[:, i] = self.columnwise_mask_(
                self.train_Y_selected[:, i], mask_p=0.2)
            self.test_Y_selected[:, i] = self.columnwise_mask_(
                self.test_Y_selected[:, i], mask_p=0.2)

        self.fit_binary_model_each_freq_labels(
            binary_model='binary_LR', prob_caliber='sigmoid', has_fitted_models=False)

        self.test_relevance_or = self.compute_relevance_or(
            self.test_pos_probs)
        self.test_relevance_and = self.compute_relevance_and(
            self.test_pos_probs)

        self.test_slots = np.array(
            list(allocated_slots[i] for i in self.selected_label_indices), dtype=np.int32)
        self.ground_truth_R_reference, self.prob_R, label_slot_idx_map, self.ground_truth_slot_idx_label_map = \
            self.build_eligibility_matrix(
                self.test_Y_selected, self.test_slots)
        self.test_R_samples = self.make_R_samples(
            self.ground_truth_R_reference, label_slot_idx_map, self.test_slots, self.test_pos_probs, self.test_Y_selected)


class BookmarksDataset(MultiLabelDataset):
    def __init__(self, selected_labels=[20, 163, 151, 144, 145, 109, 57, 89, 92, 87], n=100, slots_per_label=30) -> None:
        # allocated_slots = {20: 40, 163: 20, 151: 30, 144: 20,
        #                    145: 20, 109: 25, 57: 20, 89: 25, 92: 25, 87: 25}
        allocated_slots = {k: slots_per_label for k in selected_labels}
        with open(f'data{os.sep}bookmarks{os.sep}attributes.pickle', 'rb') as f:
            self.attributes = pickle.load(f)
        with open(f'data{os.sep}bookmarks{os.sep}labels.pickle', 'rb') as f:
            self.labels = pickle.load(f)
        # X, Y = load_from_arff(f'bookmarks{os.sep}bookmarks.arff', label_count=208)
        # train_indices, test_indices = train_test_split(np.arange(X.shape[0]), random_state=0)
        # train_X, train_Y = X[train_indices, :].tocsr(), Y[train_indices, :].toarray()
        # test_X, test_Y = X[test_indices, :].tocsr(), Y[test_indices, :].toarray()
        train_X = scipy.sparse.load_npz(
            f'data{os.sep}bookmarks{os.sep}train_X.npz')
        train_Y = np.load(f'data{os.sep}bookmarks{os.sep}train_Y.npy')
        test_X = scipy.sparse.load_npz(
            f'data{os.sep}bookmarks{os.sep}test_X.npz')
        test_Y = np.load(f'data{os.sep}bookmarks{os.sep}test_Y.npy')
        super().__init__('bookmarks', selected_labels=selected_labels, n=n, allocated_slots=allocated_slots, data_source=[
            train_X, train_Y, test_X, test_Y], prob_caliber='sigmoid', binary_model='binary_LR_large', pos_mask=True, pos_mask_p=0.2)
