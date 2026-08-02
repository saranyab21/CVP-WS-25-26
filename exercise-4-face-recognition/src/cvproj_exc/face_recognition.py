"""
face_recognition.py

1) FaceNet (OpenCV DNN + ONNX):
   - Takes an aligned 224x224 face crop and outputs a 128-D embedding.
   - Embeddings are L2-normalized so distances are meaningful.

2) FaceRecognizer (supervised identification + open-set):
   - Builds a gallery of known identities by storing embeddings + labels (saved to a pickle).
   - For prediction, it computes the embedding for the input face and compares it to all gallery embeddings.
   - Uses k-nearest neighbors (kNN) majority vote to decide identity.
   - Uses an open-set rule to reject unknowns:
       reject if (min_dist > max_distance) OR (posterior < min_prob)

   Why we do color + grayscale:
   - Color and illumination can change in videos; grayscale can be slightly more stable.
   - We store both embeddings for each identity and average both at test time for a more robust embedding.

3) FaceClustering (unsupervised k-means):
   - Stores embeddings for faces and clusters them into num_clusters groups.
   - Tracks the k-means objective per iteration (J) to analyze convergence.
   - num_clusters is tuned separately (elbow curve + seed stability); we use k=3 as a good trade-off.
"""

import os
import pickle

import cv2
import numpy as np

from dotenv import load_dotenv
load_dotenv()

from config import Config


# ----------------------------
# FaceNet: embedding extractor
# ----------------------------
class FaceNet:
    """
    Lightweight wrapper around a pre-trained FaceNet-like model (ONNX).
    Input: aligned face (224x224 BGR)
    Output: 128-D L2-normalized embedding
    """

    def __init__(self):
        # Loads the ONNX model once (expensive) and reuses it for all frames.
        self.facenet = cv2.dnn.readNetFromONNX(str(Config.RESNET50))

    def predict(self, face):
        """
        Extract a 128-D embedding from an aligned face crop.

        Notes:
        - We convert BGR -> RGB because many face models are trained in RGB.
        - We subtract a fixed mean (model-specific preprocessing).
        - Output is normalized so cosine/euclidean comparisons behave well.
        """
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB) - (131.0912, 103.8827, 91.4953)

        # OpenCV DNN expects NCHW tensor shape.
        reshaped = np.moveaxis(face, 2, 0)
        reshaped = np.expand_dims(reshaped, axis=0)

        self.facenet.setInput(reshaped)
        embedding = np.squeeze(self.facenet.forward())

        # Normalize embedding length so distances are comparable.
        return embedding / np.linalg.norm(embedding)

    @classmethod
    @property
    def get_embedding_dimensionality(cls):
        """Embedding dimensionality produced by the network."""
        return 128


# --------------------------------------------
# FaceRecognizer: supervised ID + open-set gate
# --------------------------------------------
class FaceRecognizer:
    """
    Supervised face identification:
    - Gallery = (embeddings, labels) stored in a pickle
    - Prediction = kNN in embedding space + open-set rejection
    """

    # Final parameters chosen from tuning (recog_tuning.csv)
    # k=3 gives stable kNN decision, max_distance/min_prob enforce open-set safety.
    def __init__(self, num_neighbours=3, max_distance=1.0, min_prob=0.5):
        self.facenet = FaceNet()

        # kNN parameters (tuned)
        self.num_neighbours = num_neighbours
        self.max_distance = max_distance
        self.min_prob = min_prob

        # Gallery storage
        self.labels = []
        self.embeddings = np.empty((0, FaceNet.get_embedding_dimensionality))

        # Auto-load gallery if it already exists (so you can test without re-registering).
        if os.path.exists(Config.REC_GALLERY):
            self.load()

    def save(self):
        """Persist gallery to disk."""
        print("FaceRecognizer saving: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "wb") as f:
            pickle.dump((self.labels, self.embeddings), f)

    def load(self):
        """Load gallery from disk."""
        print("FaceRecognizer loading: {}".format(Config.REC_GALLERY))
        with open(Config.REC_GALLERY, "rb") as f:
            (self.labels, self.embeddings) = pickle.load(f)

    def partial_fit(self, face, label):
        """
        Register one identity sample into the gallery.

        We store two embeddings per image:
        - color embedding
        - grayscale embedding (converted back to 3-channel for the network)

        Reason:
        - Illumination / color shifts can happen in videos.
        - Keeping both variants gives a slightly more robust representation.
        """
        # Color embedding
        embedding_color = self.facenet.predict(face)

        # Grayscale embedding (still feed 3-channel to model)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        embedding_gray = self.facenet.predict(gray_3ch)

        # Store both in gallery; both map to the same label
        self.embeddings = np.vstack((self.embeddings, embedding_color, embedding_gray))
        self.labels.extend([label, label])

    '''def predict(self, face):
        """
        Identify the person in the given face crop (or return "unknown").

        Steps:
        1) Compute embedding for input face (robust: average color + gray embedding)
        2) Compute distances to all gallery embeddings
        3) Select k nearest samples
        4) Majority vote -> predicted label + posterior
        5) Open-set gate:
           - if nearest distance is too large OR posterior too low -> reject as unknown
        """
        if len(self.labels) == 0 or self.embeddings.shape[0] == 0:
            return "unknown", 0.0, float("inf")

        # Robust embedding: average of color and grayscale variants
        embedding_color = self.facenet.predict(face)
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        embedding_gray = self.facenet.predict(gray_3ch)
        embedding = (embedding_color + embedding_gray) / 2

        # Distances to all stored embeddings
        dists = np.linalg.norm(self.embeddings - embedding, axis=1)

        # k nearest neighbors
        k = min(self.num_neighbours, len(dists))
        knn_indices = np.argsort(dists)[:k]
        knn_labels = [self.labels[i] for i in knn_indices]
        knn_dists = [dists[i] for i in knn_indices]

        # Majority vote
        unique_labels, counts = np.unique(knn_labels, return_counts=True)
        pred_idx = np.argmax(counts)
        pred_label = unique_labels[pred_idx]
        votes_for_pred = counts[pred_idx]

        # Posterior = fraction of k neighbors supporting the winner
        posterior = votes_for_pred / k

        # Distance to the predicted class (take the closest neighbor among those votes)
        pred_dists = [dist for dist, lbl in zip(knn_dists, knn_labels) if lbl == pred_label]
        min_dist = float(np.min(pred_dists))

        # Open-set decision:
        # - too far away => likely not in gallery
        # - too uncertain => likely ambiguous / noisy
        if min_dist > self.max_distance or posterior < self.min_prob:
            return "unknown", float(posterior), float(min_dist)

        return pred_label, float(posterior), float(min_dist)'''

    def predict(self, face):
        """
        Identify the person in the given face crop (or return "unknown").
    
        This version follows the assignment wording more strictly:
        - compute TWO embeddings for the query (color + grayscale)
        - run kNN on BOTH (two separate neighbor sets)
        - fuse the decisions to get:
            * predicted label
            * posterior probability
            * distance to predicted class
        """
        if len(self.labels) == 0 or self.embeddings.shape[0] == 0:
            return "unknown", 0.0, float("inf")
    
        # --- 1) Query embeddings (color + grayscale) ---
        emb_rgb = self.facenet.predict(face)
    
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        emb_gray = self.facenet.predict(gray_3ch)
    
        # Helper: run one kNN pass and return (labels, dists) of k nearest
        def knn_for_embedding(query_emb):
            dists = np.linalg.norm(self.embeddings - query_emb, axis=1)
            k = min(self.num_neighbours, len(dists))
            idx = np.argsort(dists)[:k]
            nn_labels = [self.labels[i] for i in idx]
            nn_dists = [float(dists[i]) for i in idx]
            return nn_labels, nn_dists, k
    
        # --- 2) kNN on BOTH embeddings ---
        labels_rgb, dists_rgb, k_rgb = knn_for_embedding(emb_rgb)
        labels_gray, dists_gray, k_gray = knn_for_embedding(emb_gray)
    
        # Use the same k for both (they should match, but keep it safe)
        k = min(k_rgb, k_gray)
        labels_rgb, dists_rgb = labels_rgb[:k], dists_rgb[:k]
        labels_gray, dists_gray = labels_gray[:k], dists_gray[:k]
    
        # --- 3) Fuse votes (posterior uses BOTH branches) ---
        # Count votes per class from RGB neighbors and Gray neighbors
        vote_counts = {}
        for lbl in labels_rgb:
            vote_counts[lbl] = vote_counts.get(lbl, 0) + 1
        for lbl in labels_gray:
            vote_counts[lbl] = vote_counts.get(lbl, 0) + 1
    
        # Winner = class with most total votes across both lists
        pred_label = max(vote_counts.items(), key=lambda x: x[1])[0]
        votes_for_pred = vote_counts[pred_label]
    
        # Posterior: total votes for pred over total votes (2*k)
        posterior = votes_for_pred / (2 * k)
    
        # --- 4) Distance to predicted class (use BOTH branches) ---
        # d(C_i|x) = min distance among neighbors that vote for predicted class
        pred_dists_rgb = [d for d, lbl in zip(dists_rgb, labels_rgb) if lbl == pred_label]
        pred_dists_gray = [d for d, lbl in zip(dists_gray, labels_gray) if lbl == pred_label]
    
        # If class never appeared in one branch, that list is empty -> ignore it safely
        candidates = []
        if pred_dists_rgb:
            candidates.append(min(pred_dists_rgb))
        if pred_dists_gray:
            candidates.append(min(pred_dists_gray))
    
        min_dist = min(candidates) if candidates else float("inf")
    
        # --- 5) Open-set decision (same idea, now with fused posterior+distance) ---
        if min_dist > self.max_distance or posterior < self.min_prob:
            return "unknown", float(posterior), float(min_dist)
    
        return pred_label, float(posterior), float(min_dist)



# --------------------------------------------
# FaceClustering: unsupervised clustering (k-means)
# --------------------------------------------
class FaceClustering:
    """
    Unsupervised clustering of embeddings (k-means).

    num_clusters meaning:
    - how many groups we force k-means to create in embedding space

    Why num_clusters=3:
    - based on elbow curve + seed stability from misc.py evidence:
      largest improvement from 2->3, very stable objective across seeds;
      higher k gives diminishing returns and tends to over-segment.
    """

    def __init__(self, num_clusters=3, max_iter=25):
        self.facenet = FaceNet()
        self.embeddings = np.empty((0, FaceNet.get_embedding_dimensionality))

        # k-means parameters
        self.num_clusters = num_clusters
        self.max_iter = max_iter

        # Learned outputs
        self.cluster_center = np.empty((num_clusters, FaceNet.get_embedding_dimensionality))
        self.cluster_membership = []

        # Stored per-iteration objective (for convergence analysis)
        self.objective_history = []

        # Load clustering model if present
        if os.path.exists(Config.CLUSTER_GALLERY):
            self.load()

    def save(self):
        """Persist clustering state to disk."""
        print("FaceClustering saving: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "wb") as f:
            pickle.dump(
                (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership),
                f,
            )

    def load(self):
        """Load clustering state from disk."""
        print("FaceClustering loading: {}".format(Config.CLUSTER_GALLERY))
        with open(Config.CLUSTER_GALLERY, "rb") as f:
            (self.embeddings, self.num_clusters, self.cluster_center, self.cluster_membership) = (
                pickle.load(f)
            )

    def partial_fit(self, face):
        """
        Collect embeddings for clustering.
        This does not assign clusters; it only stores points.
        """
        embedding = self.facenet.predict(face)
        self.embeddings = np.vstack((self.embeddings, embedding))

    def fit(self, visualize: bool = False, random_seed=None):
        """
        Run k-means on stored embeddings and track objective values.

        The training script calls:
          fit(visualize=True, random_seed=42)

        Objective definition:
          J = sum_i ||x_i - c_{cluster(i)}||^2

        We store J each iteration so we can:
          - plot convergence curves
          - compare stability across different seeds
        """
        if self.embeddings is None or len(self.embeddings) == 0:
            print("No embeddings found. Run partial_fit() first.")
            return

        X = np.array(self.embeddings)
        if X.shape[0] < self.num_clusters:
            print("Not enough samples to form clusters.")
            return

        if random_seed is not None:
            np.random.seed(random_seed)

        # Initialize centers randomly from the data points
        indices = np.random.choice(X.shape[0], self.num_clusters, replace=False)
        self.cluster_center = X[indices].copy()

        self.objective_history = []

        for it in range(self.max_iter):
            # Assignment step: assign each point to nearest center
            distances = np.linalg.norm(X[:, None] - self.cluster_center[None, :], axis=2)  # (N, K)
            self.cluster_membership = np.argmin(distances, axis=1)

            # Compute objective J for this iteration
            diffs = X - self.cluster_center[self.cluster_membership]
            J = float(np.sum(np.sum(diffs ** 2, axis=1)))
            self.objective_history.append(J)

            # Update step: recompute each center as mean of its assigned points
            for k in range(self.num_clusters):
                cluster_points = X[self.cluster_membership == k]
                if len(cluster_points) > 0:
                    self.cluster_center[k] = np.mean(cluster_points, axis=0)

        if visualize:
            try:
                import matplotlib.pyplot as plt
                plt.figure()
                plt.plot(self.objective_history, marker="o")
                plt.xlabel("Iteration")
                plt.ylabel("k-means objective")
                plt.title("k-means objective convergence")
                plt.grid(True)
                plt.show()
            except Exception as e:
                print(f"Visualization failed: {e}")

    def predict(self, face):
        """
        Assign a face to the closest cluster center (after fit()).

        Returns:
          (cluster_id, dists_to_all_clusters)

        """
        embedding = self.facenet.predict(face)

        if self.cluster_center is None or self.cluster_center.shape[0] == 0:
            return "unknown", np.array([float("inf")])

        dists = np.linalg.norm(self.cluster_center - embedding, axis=1)

        # Debug print is useful during development; can be removed for submission if needed.
        print("All distances to clusters:", dists)

        cluster_id = int(np.argmin(dists))
        return cluster_id, dists


