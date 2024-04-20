from collections import defaultdict
import numpy as np
from node import Node


class DecisionTree:
    def __init__(self, params: dict):
        """
        :param params: dictionary with the following keys and values:
            - depth: maximum depth of the tree
        """
        self.root_node = Node()
        self.params = defaultdict(lambda: None, params)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.root_node.train(X, y, self.params)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> None:
        predicted = self.predict(X)
        predicted = [round(p) for p in predicted]
        print(f"Accuracy: {round(np.mean(predicted == y), 2)}")

    def predict(self, X: np.ndarray) -> list[float]:
        prediction = []
        for x in X:
            prediction.append(self.root_node.predict(x))
        return prediction
