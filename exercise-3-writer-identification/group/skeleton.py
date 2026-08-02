import os
import shlex
import argparse
from tqdm import tqdm
import _pickle as cPickle
import gzip
from sklearn.cluster import MiniBatchKMeans
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
import numpy as np
import cv2
import multiprocessing
from joblib import Parallel, delayed


def parseArgs(parser):
    parser.add_argument('--labels_test', default="Exercise 2/icdar17_local_features/icdar17_labels_test.txt",
                        help='contains test images/descriptors to load + labels')
    parser.add_argument('--labels_train', default="Exercise 2/icdar17_local_features/icdar17_labels_train.txt",
                        help='contains training images/descriptors to load + labels')
    parser.add_argument('-s', '--suffix', default='_SIFT_patch_pr.pkl.gz',
                        help='only chose those images with a specific suffix')
    parser.add_argument('--in_test', default="Exercise 2/icdar17_local_features/test",
                        help='the input folder of the test images / features')
    parser.add_argument('--in_train', default="Exercise 2/icdar17_local_features/train",
                        help='the input folder of the training images / features')
    parser.add_argument('--overwrite', action='store_true',
                        help='do not load pre-computed encodings')
    parser.add_argument('--powernorm', action='store_true',
                        help='use powernorm')
    parser.add_argument('--gmp', action='store_true',
                        help='use generalized max pooling')
    parser.add_argument('--gamma', default=1, type=float,
                        help='regularization parameter of GMP')
    parser.add_argument('--C', default=1000, type=float, 
                        help='C parameter of the SVM')
    parser.add_argument('--verbose', action='store_true',
                        help='print verbose output')
    return parser

def getFiles(folder, pattern, labelfile):
    with open(labelfile, 'r') as f:
        all_lines = f.readlines()
    all_files, labels = [], []
    for line in all_lines:
        splits = shlex.split(line)
        file_name = splits[0]
        class_id = splits[1]
        for p in ['.pkl.gz', '.txt', '.png', '.jpg', '.tif', '.ocvmb', '.csv']:
            if file_name.endswith(p):
                file_name = file_name.replace(p, '')
        true_file_name = os.path.join(folder, file_name + pattern)
        all_files.append(true_file_name)
        labels.append(class_id)
    return all_files, labels

def loadRandomDescriptors(files, max_descriptors):
    max_files = min(100, len(files))
    indices = np.random.permutation(len(files))[:max_files]
    files = np.array(files)[indices]
    max_descs_per_file = int(max_descriptors / len(files))
    descriptors = []
    for i in tqdm(range(len(files)), desc="Loading descriptors"):
        with gzip.open(files[i], 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')
        idx = np.random.choice(len(desc), min(len(desc), max_descs_per_file), replace=False)
        descriptors.append(desc[idx])
    descriptors = np.concatenate(descriptors, axis=0)
    return descriptors

def dictionary(descriptors, n_clusters):
    kmeans = MiniBatchKMeans(n_clusters=n_clusters, random_state=42, batch_size=1024)
    kmeans.fit(descriptors)
    return kmeans.cluster_centers_

def assignments(descriptors, clusters):
    distances = np.linalg.norm(descriptors[:, None] - clusters, axis=2)
    nearest_clusters = np.argmin(distances, axis=1)
    assignment = np.zeros((len(descriptors), len(clusters)))
    assignment[np.arange(len(descriptors)), nearest_clusters] = 1
    return assignment

def vlad(files, mus, powernorm, gmp=False, gamma=1000, verbose=False):
    K = mus.shape[0]
    encodings = []
    for f in tqdm(files, desc="VLAD encoding"):
        with gzip.open(f, 'rb') as ff:
            desc = cPickle.load(ff, encoding='latin1')
        a = assignments(desc, mus)
        T, D = desc.shape
        f_enc = np.zeros((K, D), dtype=np.float32)
        for k in range(K):
            descriptors_k = desc[a[:, k] == 1] - mus[k]
            if gmp:
                f_enc[k] = np.sum(np.tanh(gamma * descriptors_k), axis=0)
            else:
                f_enc[k] = np.sum(descriptors_k, axis=0)
        f_enc = f_enc.flatten()
        if powernorm:
            f_enc = np.sign(f_enc) * np.sqrt(np.abs(f_enc))
        f_enc = normalize(f_enc.reshape(1, -1), norm='l2')
        encodings.append(f_enc.flatten())
        if verbose:
            print(f"Encoded file: {f}")
    return np.array(encodings)

def esvm(encs_test, encs_train, C=1000):
    def loop(i):
        y = np.zeros(len(encs_train) + 1)
        y[0] = 1
        X = np.vstack([encs_test[i], encs_train])
        svm = LinearSVC(C=C, dual=False, max_iter=5000)
        svm.fit(X, y)
        return svm.coef_.reshape(1, -1)

    n_jobs = multiprocessing.cpu_count()
    results = Parallel(n_jobs=n_jobs)(
        delayed(loop)(i) for i in tqdm(range(len(encs_test)), desc="E-SVM")
    )
    new_encs = np.concatenate(results, axis=0)
    return new_encs

def distances(encs):
    encs = normalize(encs, axis=1, norm='l2')
    dist_matrix = 1 - np.dot(encs, encs.T)
    np.fill_diagonal(dist_matrix, np.inf)
    return dist_matrix

def evaluate(encs, labels, verbose=False):
    dist_matrix = distances(encs)
    indices = dist_matrix.argsort()
    n_encs = len(encs)
    mAP = []
    correct = 0
    for r in range(n_encs):
        precisions = []
        rel = 0
        for k in range(n_encs-1):
            if labels[indices[r, k]] == labels[r]:
                rel += 1
                precisions.append(rel / float(k+1))
                if k == 0:
                    correct += 1
        avg_precision = np.mean(precisions) if precisions else 0
        mAP.append(avg_precision)
    mAP = np.mean(mAP)
    print('Top-1 accuracy: {:.4f} - mAP: {:.4f}'.format(float(correct) / n_encs, mAP))

if __name__ == '__main__':
    parser = argparse.ArgumentParser('retrieval')
    parser = parseArgs(parser)
    args = parser.parse_args()
    np.random.seed(42)

    files_train, labels_train = getFiles(args.in_train, args.suffix, args.labels_train)
    print(f'#train: {len(files_train)}')
    if not os.path.exists('mus.pkl.gz'):
        descriptors = loadRandomDescriptors(files_train, 100000)
        mus = dictionary(descriptors, 64)
        print('> loaded {} descriptors'.format(len(descriptors)))
        print('> compute dictionary')
        with gzip.open('mus.pkl.gz', 'wb') as fOut:
            cPickle.dump(mus, fOut, -1)
    else:
        with gzip.open('mus.pkl.gz', 'rb') as f:
            mus = cPickle.load(f)
        print('Loaded dictionary from mus.pkl.gz')

    files_test, labels_test = getFiles(args.in_test, args.suffix, args.labels_test)
    print(f'#test: {len(files_test)}')
    fname = 'enc_test_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_test.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        enc_test = vlad(files_test, mus, args.powernorm, args.gmp, args.gamma, args.verbose)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_test, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_test = cPickle.load(f)
        print(f'Loaded test encodings from {fname}')

    print('> evaluate')
    evaluate(enc_test, labels_test, args.verbose)

    print('> compute VLAD for train (for E-SVM)')
    fname = 'enc_train_gmp{}.pkl.gz'.format(args.gamma) if args.gmp else 'enc_train.pkl.gz'
    if not os.path.exists(fname) or args.overwrite:
        enc_train = vlad(files_train, mus, args.powernorm, args.gmp, args.gamma, args.verbose)
        with gzip.open(fname, 'wb') as fOut:
            cPickle.dump(enc_train, fOut, -1)
    else:
        with gzip.open(fname, 'rb') as f:
            enc_train = cPickle.load(f)
        print(f'Loaded train encodings from {fname}')

    print('> esvm computation')
    enc_test_esvm = esvm(enc_test, enc_train, args.C)
    print('> evaluate E-SVM')
    evaluate(enc_test_esvm, labels_test, args.verbose)
