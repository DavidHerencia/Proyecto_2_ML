import numpy as np
from itertools import combinations

class SVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=2000, kernel='linear', gamma=0.01):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel_type = kernel
        self.gamma = gamma
        self.alphas = None
        self.b = None
        self.X = None
        self.y = None
        self.val_accuracy_history = []

    def fit(self, X, y, X_val=None, y_val=None):
        self.X = X
        self.y = y
        n_samples, n_features = X.shape
        self.alphas = np.zeros(n_samples)
        self.b = 0

        # Create a kernel matrix
        K = self.kernel_matrix(X)

        for iteration in range(self.n_iters):
            condition = y * (np.dot(K, self.alphas * y) - self.b) >= 1
            self.alphas -= self.lr * (2 * self.lambda_param * self.alphas)
            self.alphas[~condition] -= self.lr * (2 * self.lambda_param * self.alphas[~condition] - 1)
            self.b -= self.lr * np.sum(y[~condition])

            # Check validation accuracy every 100 iterations
            if X_val is not None and y_val is not None and iteration % 100 == 0:
                val_accuracy = self.evaluate(X_val, y_val)
                self.val_accuracy_history.append(val_accuracy)
                print(f"Iteration {iteration}: Validation Accuracy = {val_accuracy}")

    def predict(self, X):
        K = self.kernel_matrix(X, self.X)
        approx = np.dot(K, self.alphas * self.y) - self.b
        return np.sign(approx)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)

    def kernel_matrix(self, X, X_prime=None):
        if X_prime is None:
            X_prime = self.X

        if self.kernel_type == 'linear':
            return np.dot(X, X_prime.T)
        elif self.kernel_type == 'rbf':
            sq_dists = np.sum(X ** 2, axis=1).reshape(-1, 1) + np.sum(X_prime ** 2, axis=1) - 2 * np.dot(X, X_prime.T)
            return np.exp(-self.gamma * sq_dists)


class OVRSVM:
    def __init__(self):
        self.lr = None
        self.lambda_param = None
        self.n_iters = None
        self.kernel = None
        self.gamma = None
        self.svm_models = []

    def set_params( self, learning_rate=0.001, lambda_param=0.01, n_iters=2000, kernel='linear', gamma=0.01):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.gamma = gamma

    def train(self, X, y, X_val=None, y_val=None):
        self.classes = np.unique(y)
        self.svm_models = []
        self.val_accuracy_history = []

        for c in self.classes:
            print("OVR", c)
            binary_y = np.where(y == c, 1, -1)
            binary_y_val = np.where(y_val == c, 1, -1) if y_val is not None else None
            svm = SVM(self.lr, self.lambda_param, self.n_iters, self.kernel, self.gamma)
            svm.fit(X, binary_y, X_val, binary_y_val)
            self.svm_models.append(svm)

            if binary_y_val is not None:
                self.val_accuracy_history.append(svm.val_accuracy_history)

    def predict(self, X):
        predictions = np.array([svm.predict(X) for svm in self.svm_models]).T
        return np.array([self.classes[np.argmax(pred)] for pred in predictions])


class OVOsvm:
    def __init__(self):
        self.lr = None
        self.lambda_param = None
        self.n_iters = None
        self.kernel = None
        self.gamma = None
        self.classifiers = []

    def set_params( self,learning_rate=0.001, lambda_param=0.01, n_iters=2000, kernel='linear', gamma=0.01):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.kernel = kernel
        self.gamma = gamma

    def train(self, X, y, X_val=None, y_val=None):
        self.classifiers = []
        self.val_accuracy_history = []
        classes = np.unique(y)

        for class_i, class_j in combinations(classes, 2):
            # print("OVO", class_j)
            X_pair = X[(y == class_i) | (y == class_j)]
            y_pair = y[(y == class_i) | (y == class_j)]
            y_pair = np.where(y_pair == class_i, 1, -1)
            binary_y_val = np.where(y_val == class_i, 1, -1) if y_val is not None else None
            svm = SVM(self.lr, self.lambda_param, self.n_iters, self.kernel, self.gamma)
            svm.fit(X_pair, y_pair, X_val, binary_y_val)
            self.classifiers.append((svm, class_i, class_j))

            if binary_y_val is not None:
                self.val_accuracy_history.append(svm.val_accuracy_history)

    def predict(self, X):
        predictions = []
        for svm, class_i, class_j in self.classifiers:
            pred = svm.predict(X)
            pred = np.where(pred == 1, class_i, class_j)
            predictions.append(pred)
        predictions = np.array(predictions)
        return np.array([max(set(pred), key=list(pred).count) for pred in predictions.T])
