# importing metadata
print('enter the program')
import os, sys
import numpy as np
import argparse
print('loaded parser library')

# training library
import torch
import itertools
print('loaded training library')

# clustering library
import umap
print('loaded sklearn library')

# plotting library
import matplotlib.pyplot as plt

print('finish loading package')

from utils.import_data import import_metafile
from Model.Transformer import training_bert
from Model.word2vec import run_model as training_w2v
print('finish importing models')


def add_args(parser):
    parser = argparse.ArgumentParser(description='get filament embedding using the word2vec')
    parser.add_argument('-i', '--in_parts', action='store', required=True,
                        help='path to the particle.star meta files' )
    parser.add_argument('--o', action='store', required=True,
                        help='folder to save the particle file' )
    parser.add_argument('--datatype', action='store', required=False, default=0,
                        help='0 is relion 3.1, 1 is relion 3, 2 is cryosparc' )
    parser.add_argument('--model', action='store', required=False, default='word2vec',
                        help='model to calculate the 2D class embedding, using word2vec or bert' )
    parser.add_argument('-v','--verbose',action='store_true',help='Increaes verbosity')
    parser.add_argument('--j', action='store', help='number of thread')
    parser.add_argument('--max', action='store', default=0,type=int,
                        help='max filament size (in term of number of segments in a filament), 0 means max')
    parser.add_argument('--display', action='store_true', help='display the result')

    group = parser.add_argument_group('word2vec model parameter and filament averaging')
    group.add_argument('--window', action='store', required=False, default=1,
                       help='the window size to consider the neighboring segments')
    group.add_argument('--negative', action='store', required=False, default=4,
                       help = 'number of the negative sample')
    group.add_argument('--emb_size', action='store', required=False, default=100,
                       help = 'the size of word embedding')
    group.add_argument('--batch', action='store', required=False, default=1000,
                       help = 'the batch size')
    group.add_argument('--min_gain', action='store', required=False, default=1,
                       help = 'min gain of the loss')
    group.add_argument('--loss', action='store', required=False, default='cross_entropy',
                       help = 'loss function')
    group.add_argument('--optim', action='store', required=False, default='adam',
                       help = 'loss function')
    group.add_argument('--avg_method', action='store', required=False, default=0,
                       help = 'the method to do filament average, 0 means simple average, 1 means weight average')

    group = parser.add_argument_group('dimension reduction and cluster diameter')
    group.add_argument('--n_neighbors', action='store', required=False, default=15,
                       help = 'n_neighboers for the umap')
    group.add_argument('--min_dist', action='store', required=False, default=0.1,
                       help = 'min_dis in umap')
    group.add_argument('--pca_dim', action='store', required=False, default=3,
                       help = 'PCA dimension')


    return parser

def device_name():
    if torch.cuda.is_available():
        dev = "cuda:0"
    else:
        dev = "cpu"
    device = torch.device(dev)
    return device
# main program
def main(args):

    device=device_name()

    #check the input and output
    file_path=args.in_parts
    datatype=args.datatype
    file_name = os.path.basename(file_path)
    output_path = args.o
    if os.path.isdir(output_path) is False:
        os.mkdir(output_path)

    #import the files
    filament_corpus, max_length= import_metafile(file_path, datatype)
    if args.max>0:
        max_length=args.max
    else:
        max_length=max_length

    vocabulary = set(itertools.chain.from_iterable(filament_corpus))
    vocabulary_size = len(vocabulary)
    print('number of 2D classes: %d' % vocabulary_size)

    word_to_index = {w: idx for (idx, w) in enumerate(vocabulary)}


    if args.model == 'word2vec':
        filament_score = training_w2v(filament_corpus,device,embedding_size=args.emb_size,w=args.window)
    elif args.model == 'bert':
        # remember this is the cut corpus one
        output_path = output_path[:-1]
        filament_score = training_bert(filament_corpus, block_size=max_length, output_path=output_path)
        output_path = output_path+'/'
    else:
        raise AssertionError('Such model has not been implemented')

    all_data = filament_score[:]
    np.save(output_path + '/filament_score.npy', np.array(filament_score))

    n_neighbors = args.n_neighbors
    min_dist = args.min_dist
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    umap_2D = reducer.fit_transform(all_data)
    filament_umap = umap_2D[:]

    plt.figure(figsize=(20, 20))
    plt.scatter(filament_umap[:, 0], filament_umap[:, 1], alpha=0.6, color='blue')
    plt.savefig(output_path + '/' + os.path.splitext(file_name)[0] + "_umap_blue.png", bbox_inches='tight',
                pad_inches=0.01)

    np.save(output_path + '/umap_2D.npy', umap_2D)

    if args.display:
        dm_path=output_path + '/umap_2D.npy'
        meta_path= file_path
        print(type(dm_path),type(dm_path))
        os.system('streamlit run /net/jiang/home/li3221/scratch/Github/2Dclass2vec/web_app.py -- --dm_path %s --meta_path %s' % (dm_path, meta_path))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    args = add_args(parser).parse_args()
    print(args)
    main(args)
