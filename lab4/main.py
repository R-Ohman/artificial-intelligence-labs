import numpy as np

from decision_tree import DecisionTree
from random_forest import RandomForest
from load_data import generate_data, load_titanic


def main():
    np.random.seed(123)

    train_data, test_data = load_titanic()

    print("Decision Tree")
    dt = DecisionTree({"depth": 14})
    dt.train(*train_data)
    print("[train]", end="\t")
    dt.evaluate(*train_data)
    print("[test]", end="\t")
    dt.evaluate(*test_data)

    print("Random Forest")
    rf = RandomForest({"ntrees": 10, "feature_subset": 2, "depth": 14})
    rf.train(*train_data)
    print("[train]", end="\t")
    rf.evaluate(*train_data)
    print("[test]", end="\t")
    rf.evaluate(*test_data)


if __name__=="__main__":
    main()
