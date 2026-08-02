# skeleton.py -- Writer Identification: Baseline + Bonus
# Author: Saranya Bhattacharjee
# Implements bonus tasks (e, f, g): SIFT+Hellinger, GMP, Multi-VLAD+PCA

import numpy as np
import cv2
import os
import pickle
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import Ridge
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score

# ----------- USER FLAGS -----------
USE_SIFT_HELLINGER = True     # (e) Bonus: Use SIFT + Hellinger descriptors
USE_GMP_VLAD = False            # (f) Bonus: Use Generalized Max Pooling (GMP)
USE_MULTI_VLAD_PCA = False     # (g) Bonus: Use Multi-VLAD + PCA whitening

ZENODO_IMG_DIR_TRAIN = r"C:\Users\admin\OneDrive\Desktop\SEM-2\Computer Vision project\Exercise 2\icdar17-historicalwi-training-color\icdar2017-training-color"
ZENODO_IMG_DIR_TEST  = r"C:\Users\admin\OneDrive\Desktop\SEM-2\Computer Vision project\Exercise 2\icdar17-historicalwi-training-color\icdar2017-training-color"
N_CLUSTERS = 32
N_MULTI_VLAD = 5
#PCA_DIM = 1000  #Should use this for full dataset
PCA_DIM = 5  #Used this since N_SMALL=5

# --------- Utility Functions ----------

def load_image_gray(filename_base, img_dir):
    for ext in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        fname = os.path.join(img_dir, filename_base)
        if not fname.lower().endswith(ext):
            fname_ext = fname + ext
        else:
            fname_ext = fname
        if os.path.isfile(fname_ext):
            img = cv2.imread(fname_ext, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                return img
    fname = os.path.join(img_dir, filename_base)
    if os.path.isfile(fname):
        img = cv2.imread(fname, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            return img
    raise FileNotFoundError(f"Image not found (tried .png/.jpg): {filename_base} in {img_dir}")

def computeDescs(filename_base, img_dir):
    img = load_image_gray(filename_base, img_dir)
    sift = cv2.SIFT_create()
    kps, descs = sift.detectAndCompute(img, None)
    if descs is None or len(descs) == 0:
        return np.zeros((0, 128), np.float32)
    for kp in kps:
        kp.angle = 0
    descs0 = []
    for kp in kps:
        _, desc = sift.compute(img, [kp])
        if desc is not None:
            descs0.append(desc[0])
    descs = np.array(descs0, dtype=np.float32)
    descs_l1 = descs / (np.linalg.norm(descs, ord=1, axis=1, keepdims=True) + 1e-8)
    descs_hell = np.sign(descs_l1) * np.sqrt(np.abs(descs_l1))
    return descs_hell.astype(np.float32)

def save_descriptors(descs_list, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(descs_list, f)

def load_descriptors(in_file):
    with open(in_file, 'rb') as f:
        return pickle.load(f)

def loadRandomDescriptors_from_list(descs_list, n_desc=10000):
    descs_all = []
    count = 0
    for desc in descs_list:
        if desc.shape[0] > 0:
            descs_all.append(desc)
            count += desc.shape[0]
        if count >= n_desc:
            break
    if len(descs_all) == 0:
        raise RuntimeError("No descriptors loaded!")
    return np.vstack(descs_all)[:n_desc]

def get_assignments(descs, codebook):
    return np.argmin(np.linalg.norm(descs[:,None,:] - codebook[None,:,:], axis=2), axis=1)

def vlad_encode(descs, codebook):
    assignments = get_assignments(descs, codebook)
    vlad = np.zeros((codebook.shape[0], codebook.shape[1]), dtype=np.float32)
    for k in range(codebook.shape[0]):
        if np.sum(assignments == k) > 0:
            vlad[k] = np.sum(descs[assignments == k] - codebook[k], axis=0)
    vlad = vlad.flatten()
    vlad = vlad / (np.linalg.norm(vlad) + 1e-12)
    return vlad

def vlad_gmp(descs, codebook):
    assignments = get_assignments(descs, codebook)
    vlad = np.zeros((codebook.shape[0], codebook.shape[1]), dtype=np.float32)
    for k in range(codebook.shape[0]):
        cluster_res = descs[assignments == k] - codebook[k]
        if cluster_res.shape[0] == 0:
            continue
        gmp = []
        for d in range(cluster_res.shape[1]):
            ridge = Ridge(alpha=1.0, fit_intercept=False, solver='sparse_cg', max_iter=500)
            y = np.ones(cluster_res.shape[0])
            X = cluster_res[:, [d]]
            try:
                ridge.fit(X, y)
                gmp.append(ridge.coef_[0])
            except Exception:
                gmp.append(0.0)
        vlad[k] = np.array(gmp)
    vlad = vlad.flatten()
    vlad = vlad / (np.linalg.norm(vlad) + 1e-12)
    return vlad

def multi_vlad_encode(descs, codebooks):
    vlad_list = []
    for codebook in codebooks:
        vlad = vlad_encode(descs, codebook)
        vlad_list.append(vlad)
    return np.concatenate(vlad_list)

def progress_bar(current, total, bar_length=40):
    percent = float(current) / total
    arrow = '-' * int(round(percent * bar_length) - 1) + '>'
    spaces = ' ' * (bar_length - len(arrow))
    print(f'\rProgress: [{arrow}{spaces}] {int(percent * 100)}%', end='')

def extract_and_save_all_sift(train_list, test_list, train_file, test_file):
    print("Extracting SIFT+Hellinger descriptors for TRAIN set")
    train_descs = []
    for i, fn in enumerate(train_list):
        train_descs.append(computeDescs(fn, ZENODO_IMG_DIR_TRAIN))
        if (i + 1) % 10 == 0 or i == len(train_list) - 1:
            progress_bar(i + 1, len(train_list))
    print("\nSaving train descriptors")
    save_descriptors(train_descs, train_file)
    print(f"Train descriptors saved to {train_file}")

    print("Extracting SIFT+Hellinger descriptors for TEST set")
    test_descs = []
    for i, fn in enumerate(test_list):
        test_descs.append(computeDescs(fn, ZENODO_IMG_DIR_TEST))
        if (i + 1) % 10 == 0 or i == len(test_list) - 1:
            progress_bar(i + 1, len(test_list))
    print("\nSaving test descriptors")
    save_descriptors(test_descs, test_file)
    print(f"Test descriptors saved to {test_file}")

# Main pipeline
def run_experiment(train_list, test_list, labels_train, labels_test,
                   train_descs_file='train_sift_hellinger.pkl',
                   test_descs_file='test_sift_hellinger.pkl'):
    print("Train labels:", np.unique(labels_train))
    print("Test labels:", np.unique(labels_test))
    print("Common labels:", set(labels_train).intersection(set(labels_test)))

    # Descriptor extraction or loading
    if USE_SIFT_HELLINGER:
        if os.path.exists(train_descs_file) and os.path.exists(test_descs_file):
            print("Loading saved SIFT+Hellinger descriptors from previous run")
            train_descs = load_descriptors(train_descs_file)
            test_descs = load_descriptors(test_descs_file)
        else:
            print("First extracting all SIFT+Hellinger descriptors")
            extract_and_save_all_sift(train_list, test_list, train_descs_file, test_descs_file)
            train_descs = load_descriptors(train_descs_file)
            test_descs = load_descriptors(test_descs_file)
    else:
        if not (os.path.exists(train_descs_file) and os.path.exists(test_descs_file)):
            raise RuntimeError("Please run once with USE_SIFT_HELLINGER=True to save SIFT+Hellinger descriptors")
        print("Loading saved SIFT+Hellinger descriptors from previous run")
        train_descs = load_descriptors(train_descs_file)
        test_descs = load_descriptors(test_descs_file)

    # Sanity log!
    print(f"Loaded train descriptors: {len(train_descs)} | sample shapes: {[d.shape for d in train_descs[:3]]}")
    print(f"Loaded test descriptors: {len(test_descs)} | sample shapes: {[d.shape for d in test_descs[:3]]}")

    # --------- Build codebooks -------------
    if USE_MULTI_VLAD_PCA:
        print("Building multi-VLAD codebooks")
        codebooks = []
        for i in range(N_MULTI_VLAD):
            np.random.seed(i+42)
            descs_sample = loadRandomDescriptors_from_list(train_descs, n_desc=10000)
            kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=i+42, batch_size=1000, max_iter=100)
            codebook = kmeans.fit(descs_sample).cluster_centers_
            codebooks.append(codebook)
    else:
        print("Building single VLAD codebook")
        descs_sample = loadRandomDescriptors_from_list(train_descs, n_desc=10000)
        kmeans = MiniBatchKMeans(n_clusters=N_CLUSTERS, random_state=0, batch_size=1000, max_iter=100)
        codebook = kmeans.fit(descs_sample).cluster_centers_

    # --------- VLAD encoding ---------------
    print("Encoding VLAD vectors")
    if USE_MULTI_VLAD_PCA:
        X_train = [multi_vlad_encode(desc, codebooks) for desc in train_descs]
        X_test = [multi_vlad_encode(desc, codebooks) for desc in test_descs]
    elif USE_GMP_VLAD:
        X_train = [vlad_gmp(desc, codebook) for desc in train_descs]
        X_test = [vlad_gmp(desc, codebook) for desc in test_descs]
    else:
        X_train = [vlad_encode(desc, codebook) for desc in train_descs]
        X_test = [vlad_encode(desc, codebook) for desc in test_descs]
    X_train = np.stack(X_train)
    X_test = np.stack(X_test)

    # --------- PCA Whitening (bonus g) -----
    if USE_MULTI_VLAD_PCA:
        print(f"Applying PCA whitening to {X_train.shape[1]}-D VLAD features...")
        pca = PCA(n_components=PCA_DIM, whiten=True, random_state=0)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    # --------- Classification (cosine NN) ----
    print("Computing cosine similarity for retrieval")
    X_train_n = X_train / (np.linalg.norm(X_train, axis=1, keepdims=True) + 1e-12)
    X_test_n = X_test / (np.linalg.norm(X_test, axis=1, keepdims=True) + 1e-12)
    sims = X_test_n @ X_train_n.T

    aps = []
    for i in range(X_test.shape[0]):
        y_true = (labels_train == labels_test[i]).astype(int)
        y_score = sims[i]
        if np.sum(y_true) == 0:
            continue
        aps.append(average_precision_score(y_true, y_score))
    if aps:
        mAP = np.mean(aps)
    else:
        print("Warning: No valid queries with positive labels for AP computation")
        mAP = 0.0
    print(f"mAP: {mAP:.4f}")
    return mAP

if __name__ == '__main__':
    train_split_file = 'color_labels_train.txt'
    test_split_file = 'color_labels_test.txt'
    train_list = [l.strip().split()[0] for l in open(train_split_file)]
    test_list = [l.strip().split()[0] for l in open(test_split_file)]
    labels_train = np.array([l.strip().split()[1] for l in open(train_split_file)])
    labels_test = np.array([l.strip().split()[1] for l in open(test_split_file)])

    #N_SMALL = None   # Use None for full dataset
    N_SMALL = 5     # Using this for now
    if N_SMALL:
        train_list = train_list[:N_SMALL]
        test_list = test_list[:N_SMALL]
        labels_train = labels_train[:N_SMALL]
        labels_test = labels_test[:N_SMALL]

    mAP = run_experiment(
        train_list, test_list, labels_train, labels_test,
        train_descs_file='train_sift_hellinger.pkl',
        test_descs_file='test_sift_hellinger.pkl'
    )

    print("Done.")
