import numpy as np
import pandas as pd


def load_datasets(csv_path='./data/harmbench_behaviors.csv', number_of_behaviors=10):
    """
    Load and filter dataset:
        csv_path: Load the dataset from the given CSV file path and filter the dataset to include only behaviors with FunctionalCategory as "standard".
        number_of_behaviors: the number of behaviors to sample from the filtered dataset. If None, returns all behaviors. (Note that the maximum number of behaviors is 159.)
    Returns a dataframe with the following structure:
        | Behavior | FunctionalCategory | SemanticCategory | Tags | ContextString | BehaviorID |
    """
    np.random.seed(42)
    df = pd.read_csv(csv_path)
    filtered_df = df[df["FunctionalCategory"] == "standard"]
    if number_of_behaviors is None:
        return filtered_df
    return filtered_df[:number_of_behaviors]

if __name__ == "__main__":
    dataset = load_datasets("./data/harmbench_behaviors.csv")
    print(dataset.head())
