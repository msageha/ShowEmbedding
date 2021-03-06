import os
import argparse
import gensim
import umap
from scipy.sparse.csgraph import connected_components #need for umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
# from sklearn.manifold import TSNE
#データサイズによっては，sklearn のsingle core TSNEが遅いため，multi coreのものを使う．
from MulticoreTSNE import MulticoreTSNE as TSNE
import numpy as np
from sklearn.preprocessing import LabelEncoder
import time

def load(path, domains=['OC', 'OW', 'OY', 'PB', 'PM', 'PN']):
    print('start loading', flush=True)
    model = gensim.models.KeyedVectors.load(path)
    _labels = list(model.docvecs.doctags)
    labels = []
    for _label in _labels:
        if np.array([domain in _label for domain in domains]).sum():
            labels.append(_label)
    embeddings = [model[label] for label in labels]

    return embeddings, labels

def checkLabel(label):
    domains = ['LB', 'OB', 'OC', 'OL', 'OM', 'OP', 'OT', 'OV', 'OW', 'OY', 'PB', 'PM', 'PN']
    for domain in domains:
        if domain in label:
            return domain

def show(weights, labels, path):
    _labels = [checkLabel(label) for label in labels]
    le = LabelEncoder()
    le.fit(_labels)
    _labels = le.transform(_labels)
    fig = plt.figure(figsize=(10,10), dpi=200)
    ax=plt.subplot()
    if len(le.classes_) > 10:
        cmap = cm.tab20
    else:
        cmap = cm.tab10
    for domain in le.classes_:
        _label = le.transform([domain])[0]
        numpy_filter = _labels == _label
        ax.scatter(weights[numpy_filter][:,0], weights[numpy_filter][:,1], c=cmap(_label), label=domain, alpha=0.5, s=5)
    #範囲のセット
    # ax.set_xlim([-40, 40])
    # ax.set_ylim([-40, 40])
    plt.legend()
    plt.savefig(path)
    plt.clf()

def main():
    parser = argparse.ArgumentParser(description='main function parser')
    parser.add_argument('--path', type=str, help='load file path', required=True)
    parser.add_argument('--dump_dir', type=str, help='dump directory', default=None)
    parser.add_argument('--size', type=int, default=1000, help='embedding vector size')
    args = parser.parse_args()

    embeddings, labels = load(args.path)
    embeddings = np.array(embeddings)

    output = args.path.split('/')[-1]
    # # UMAP
    n_neighbors = [15] #, 35, 55, 75]
    min_dists = [0.1] #0.001, 0.01, 0.1]
    for min_dist in min_dists:
        for n_neighbor in n_neighbors:
            start = time.time()
            weights = umap.UMAP(n_neighbors=n_neighbor, min_dist=min_dist).fit_transform(embeddings)
            finish = time.time()
            print(f'time: {finish-start} s', flush=True)
            os.makedirs(f'graph/umap/{output}', exist_ok=True)
            show(weights, labels, f'graph/umap/{output}/min_dist:{min_dist}_neighbor:{n_neighbor}.svg')

    # t-SNE
    perplexities = [30] #10, 20, 30, 40, 50]
    for perplexity in perplexities:
        start = time.time()
        tsne_model = TSNE(n_components=2, perplexity=perplexity, n_jobs=10)
        weights = tsne_model.fit_transform(embeddings)
        finish = time.time()
        print(f'time: {finish-start} s', flush=True)
        os.makedirs(f'graph/tsne/{output}', exist_ok=True)
        show(weights, labels, f'graph/tsne/{output}/perplexity:{perplexity}.svg')

if __name__=='__main__':
    main()