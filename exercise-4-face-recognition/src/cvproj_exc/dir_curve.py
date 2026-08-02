import matplotlib.pyplot as plt
import numpy as np
import os

from dotenv import load_dotenv
load_dotenv()

from classifier import NearestNeighborClassifier
from config import Config
from evaluation import OpenSetEvaluation


def main():
    # The range of the false alarm rate in logarithmic space to draw DIR curves.
    false_alarm_rate_range = np.logspace(-3.0, 0, 1000, endpoint=False)

    # Pickle files containing embeddings and corresponding class labels for the
    # training and the test dataset.
    train_data_file = Config.EVAL_TRAIN_DATA
    test_data_file = Config.EVAL_TEST_DATA

    # Use nearest neighbor classifier
    classifier = NearestNeighborClassifier()

    # Initialize open-set evaluator
    evaluation = OpenSetEvaluation(
        classifier=classifier,
        false_alarm_rate_range=false_alarm_rate_range
    )
    evaluation.prepare_input_data(train_data_file, test_data_file)

    # Run evaluation
    results = evaluation.run()

    # Plot DIR curve
    plt.figure(figsize=(8, 6))
    plt.semilogx(
        results["false_alarm_rates"],
        results["identification_rates"],
        markeredgewidth=1,
        linewidth=2.5,
        linestyle="--",
        color="blue",
        label="DIR Curve"
    )
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.axis([
        false_alarm_rate_range[0],
        false_alarm_rate_range[-1],
        0, 1
    ])
    plt.xlabel("False alarm rate")
    plt.ylabel("Identification rate")
    plt.title("Open-Set Face Identification - DIR Curve")
    plt.legend()

    # Save in current directory
    output_path = "dir_curve.png"
    plt.savefig(output_path, dpi=300)
    print(f"[INFO] DIR curve saved to: {output_path}")
    plt.show()


if __name__ == "__main__":
    main()
