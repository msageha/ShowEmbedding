import argparse
import gensim
import umap
from scipy.sparse.csgraph import connected_components #need for umap
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.manifold import TSNE
import time


def load(path, size=1000):
    model = gensim.models.KeyedVectors.load(path)
    embeddings = model.wv.vectors.copy()
    labels = model.wv.index2word.copy()
    embeddings = embeddings[:size]
    labels = labels[:size]

    return embeddings, labels

def show(weights, labels, path):
    fig = plt.figure(figsize=(10,10),dpi=200)
    ax=plt.subplot()
    for weight, label in zip(weights, labels):
        ax.annotate(label, (weight[0], weight[1]), fontsize=5)
    ax.scatter(weights[:, 0], weights[:, 1], alpha=0.5, s=5)
    #範囲のセット
    ax.set_xlim([-40, 40])
    ax.set_ylim([-40, 40])
    plt.savefig(path)
    plt.clf()

def main():
    parser = argparse.ArgumentParser(description='main function parser')
    parser.add_argument('--path', type=str, help='load file path', required=True)
    parser.add_argument('--dump_dir', type=str, help='dump directory', default=None)
    parser.add_argument('--size', type=int, default=1000, help='embedding vector size')
    args = parser.parse_args()

    embeddings, labels = load(args.path, args.size)

    # UMAP
    weights = umap.UMAP().fit_transform(embeddings)
    show(weights, labels, 'umap.svg')

    # t-SNE
    tsne_model = TSNE(n_components=2)
    weights = tsne_model.fit_transform(embeddings)
    show(weights, labels, 'tsne.svg')

if __name__=='__main__':
    main()