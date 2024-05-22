import numpy as np
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import time


class DecisionTree:
    def __init__(self, max_depth=None, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features

    def fit(self, X, y):
        self.n_classes = len(set(y))
        self.features = X.shape[1]
        self.tree = self._grow_tree(X, y)

    def predict(self, X):
        return np.array([self._predict(inputs) for inputs in X])

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        if (
            n_samples < self.min_samples_split
            or depth >= self.max_depth
            or len(set(y)) == 1
        ):
            return self._most_common_label(y)

        feat_idxs = np.random.choice(n_features, self.n_features, replace=False)
        best_feat, best_thresh = self._best_criteria(X, y, feat_idxs)

        if best_feat is None:
            return self._most_common_label(y)

        left_idxs, right_idxs = self._split(X[:, best_feat], best_thresh)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return self._most_common_label(y)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return (best_feat, best_thresh, left, right)

    def _best_criteria(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_thresh = None, None
        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)
            for threshold in thresholds:
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_thresh = threshold
        return split_idx, split_thresh

    def _information_gain(self, y, X_column, split_thresh):
        parent_entropy = self._entropy(y)
        left_idxs, right_idxs = self._split(X_column, split_thresh)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        n, n_left, n_right = len(y), len(left_idxs), len(right_idxs)
        e_left, e_right = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_left / n) * e_left + (n_right / n) * e_right

        ig = parent_entropy - child_entropy
        return ig

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log2(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        counter = np.bincount(y)
        return np.argmax(counter)

    def _predict(self, inputs):
        node = self.tree
        while isinstance(node, tuple):
            if inputs[node[0]] <= node[1]:
                node = node[2]
            else:
                node = node[3]
        return node


class RandomForest:
    def __init__(self):
        self.n_trees = None
        self.max_depth = None
        self.min_samples_split = None
        self.max_features = None
        self.n_jobs = None
        self.trees = []

    def set_params( self, n_trees=100,  max_depth=None,min_samples_split=2,max_features="sqrt",n_jobs=1,):
        self.n_trees = int(n_trees)
        self.max_depth = int(max_depth)
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features
        self.n_jobs = int(n_jobs)

    def train(self, X, y):
        self.trees = []
        self.n_features = X.shape[1]
        if self.max_features == "sqrt":
            self.n_features = int(np.sqrt(self.n_features))
        elif self.max_features == "log2":
            self.n_features = int(np.log2(self.n_features))
        elif isinstance(self.max_features, int):
            self.n_features = self.max_features

        with Pool(processes=self.n_jobs) as pool:
            self.trees = pool.map(self._fit_tree, [(X, y) for _ in range(self.n_trees)])

    def _fit_tree(self, data):
        X, y = data
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_features=self.n_features,
        )
        X_sample, y_sample = self._bootstrap_sample(X, y)
        tree.fit(X_sample, y_sample)
        return tree

    def predict(self, X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        tree_preds = np.swapaxes(tree_preds, 0, 1)
        y_pred = [np.bincount(tree_pred).argmax() for tree_pred in tree_preds]
        return np.array(y_pred)

    def _bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        idxs = np.random.choice(n_samples, n_samples, replace=True)
        return X[idxs], y[idxs]
