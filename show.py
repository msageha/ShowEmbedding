# coding:utf-8
import argparse

import gensim
import torch
from tensorboardX import SummaryWriter


def create(path, dump_dir=None, size=10000):
    # tensorboard MAX 100,000 vectors https://github.com/tensorflow/tensorboard/issues/773
    # vec_path = "/Users/sango.m.ab/Desktop/research2/SentenceClasification/.vector_cache/wiki.ja.vec"
    # vec_path = '/Users/sango.m.ab/Desktop/research/data/entity_vector/entity_vector.model.txt'
    comment = path.split('/')[-1]
    writer = SummaryWriter(log_dir=dump_dir, comment=comment)
    model = gensim.models.KeyedVectors.load(path)
    weights = model.wv.vectors.copy()
    labels = model.wv.index2word.copy()
    size = 1000
    # tensorboard '\u3000'(全角スペース)はだめ.\xa0もだめなため注意．
    # 学習済みfastTextには，46と9027番目に含まれている．
    weights = weights[:size]
    labels = labels[:size]

    for index, label in enumerate(labels):
        if '\u3000' == label:
            labels[index] = '空白'

    writer.add_embedding(torch.FloatTensor(weights), metadata=labels)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='main function parser')
    parser.add_argument('--path', type=str, help='load file path', required=True)
    parser.add_argument('--dump_dir', type=str, help='dump directory', default=None)
    parser.add_argument('--size', type=int, default=10000, help='embedding vector size')
    args = parser.parse_args()

    create(args.path, size=args.size, dump_dir=args.dump_dir)