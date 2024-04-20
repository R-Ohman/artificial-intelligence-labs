from collections import defaultdict
import numpy as np
from decision_tree import DecisionTree

class RandomForest:
    def __init__(self, params: dict):
        """
        :param params: dictionary with the following keys and values:
            - ntrees: number of trees in the forest
            - feature_subset: number of features to consider for each split
            - depth: maximum depth of each tree
        """
        self.forest = []
        self.params = defaultdict(lambda: None, params)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        for _ in range(self.params["ntrees"]):
            X_bagging, y_bagging = self.bagging(X,y)
            tree = DecisionTree(self.params)
            tree.train(X_bagging, y_bagging)
            self.forest.append(tree)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> None:
        predicted = self.predict(X)
        predicted = [round(p) for p in predicted]
        print(f"Accuracy: {round(np.mean(predicted==y),2)}")

    def predict(self, X: np.ndarray) -> list[float]:
        tree_predictions = []
        for tree in self.forest:
            tree_predictions.append(tree.predict(X))
        forest_predictions = list(map(lambda x: sum(x)/len(x), zip(*tree_predictions)))
        return forest_predictions

    @staticmethod
    def bagging(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
        X_selected, y_selected = X[idx], y[idx]
        return X_selected, y_selected
