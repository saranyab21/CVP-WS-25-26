import pickle

import numpy as np

from classifier import NearestNeighborClassifier

# Class label for unknown subjects in test and training data.
UNKNOWN_LABEL = -1


# Evaluation of open-set face identification.
class OpenSetEvaluation:

    def __init__(
        self,
        classifier=NearestNeighborClassifier(),
        false_alarm_rate_range=np.logspace(-3, 0, 1000, endpoint=True),
    ):
        # The false alarm rates.
        self.false_alarm_rate_range = false_alarm_rate_range

        # Datasets (embeddings + labels) used for training and testing.
        self.train_embeddings = []
        self.train_labels = []
        self.test_embeddings = []
        self.test_labels = []

        # The evaluated classifier (see classifier.py)
        self.classifier = classifier

    # Prepare the evaluation by reading training and test data from file.
    def prepare_input_data(self, train_data_file, test_data_file):
        with open(train_data_file, "rb") as f:
            (self.train_embeddings, self.train_labels) = pickle.load(f, encoding="bytes")
        with open(test_data_file, "rb") as f:
            (self.test_embeddings, self.test_labels) = pickle.load(f, encoding="bytes")

    # Run the evaluation and find performance measure (identification rates) at different
    # similarity thresholds.
    def run(self):
        self.classifier.fit(self.train_embeddings, self.train_labels)
        pred_labels, similarities = self.classifier.predict_labels_and_similarities(self.test_embeddings)

        thresholds = []
        identification_rates = []

        for far in self.false_alarm_rate_range:
            threshold = self.select_similarity_threshold(similarities, far)
            thresholds.append(threshold)

            open_set_preds = [
                label if sim >= threshold else UNKNOWN_LABEL
                for label, sim in zip(pred_labels, similarities)
            ]

            id_rate = self.calc_identification_rate(np.array(open_set_preds))
            identification_rates.append(id_rate)

        return {
                "similarity_thresholds": thresholds,
                "false_alarm_rates": self.false_alarm_rate_range,
                "identification_rates": identification_rates,
        }

    def select_similarity_threshold(self, similarities, false_alarm_rate):
        # Use similarity scores for test samples with UNKNOWN label
        unknown_similarities = np.array([
            sim for sim, label in zip(similarities, self.test_labels)
            if label == UNKNOWN_LABEL
        ])

        if len(unknown_similarities) == 0:
            return float("-inf")  # No unknown samples, accept everything

        percentile = 100 * (1.0 - false_alarm_rate)
        return np.percentile(unknown_similarities, percentile)

    def calc_identification_rate(self, prediction_labels):
        # Compute identification rate (excluding unknown GTs)
        known_mask = self.test_labels != UNKNOWN_LABEL
        if np.sum(known_mask) == 0:
            return 0.0
        correct = np.sum(prediction_labels[known_mask] == np.array(self.test_labels)[known_mask])
        return correct / np.sum(known_mask)
