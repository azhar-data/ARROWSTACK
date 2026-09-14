"""Run both projects offline using the included, checksummed source datasets."""
from task1_eda.analysis import run
from task2_regression.train import train


if __name__ == "__main__":
    print("Task 1 — Palmer Penguins EDA")
    print(run())
    print("Task 2 — Auto MPG regression")
    result = train()
    print(result["selected_model"])
    print(result["test_metrics"])
