# importing metadata
print('enter the program')
import os, sys
import numpy as np
import pandas as pd
from numpy.random import multinomial
import argparse
import EMdata
print('load parser library')

# clustering library
import umap
import sklearn
from sklearn.cluster import KMeans,SpectralClustering,MeanShift, estimate_bandwidth,AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from scipy.spatial.distance import cdist
from scipy.stats import multivariate_normal
from sklearn.decomposition import PCA
print('load sklearn library')

# models
from Model import word2vec, Transformer

# plotting library
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import time

print('finish loading package')


def add_args(parser):
    parser = argparse.ArgumentParser(description='get filament embedding using the word2vec')
    parser.add_argument('-i', '--in_parts', action='store', required=True,
                        help='path to the particle.star meta files' )
    parser.add_argument('--o', action='store', required=True,
                        help='folder to save the particle file' )
    parser.add_argument('--datatype', action='store', required=False, default=0,
                        help='0 is relion 3.1, 1 is relion 3, 2 is cryosparc' )
    parser.add_argument('--model', action='store', required=False, default='word2vec',
                        help='model to calculate the 2D class embedding' )
    parser.add_argument('-v','--verbose',action='store_true',help='Increaes verbosity')
    parser.add_argument('--j', action='store', help='number of thread')
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

    group = parser.add_argument_group('dimension reduction and cluster diameter')
    group.add_argument('--n_neighbors', action='store', required=False, default=15,
                       help = 'n_neighboers for the umap')
    group.add_argument('--min_dist', action='store', required=False, default=0.1,
                       help = 'min_dis in umap')
    group.add_argument('--pca_dim', action='store', required=False, default=3,
                       help = 'PCA dimension')


    return parser

def import_metafile(file_path, datatype):
    
    file_info=EMdata.read_data_df(file_path)
    dataframe=file_info.star2dataframe()
    metadata=dataframe.columns
    corpus_information=EMdata.process_helical_df(dataframe).extract_helical_select()
    corpus_dic, helix_name = corpus_information
    corpus = list(corpus_dic.values())
    corpus_backup = corpus[:]
    corpus_ignore = []
    for i in range(len(corpus)):
        corpus_row = []
        lst = corpus[i]
        count = lst[0][1]
        for j in range(len(lst)):
            particle = lst[j]
            if count == int(particle[1]):
                corpus_row.append(particle[0])
                count += 1
            else:
                while 1:
                    if count == int(lst[j][1]):
                        corpus_row.append(particle[0])
                        count += 1
                        break
                    corpus_row += [0]
                    count += 1
        corpus_ignore.append(corpus_row)

    corpus_length_histogram = []
    for i in range(len(corpus_ignore)):
        corpus_length_histogram.append(len(corpus_ignore[i]))
    fig, ax = plt.subplots(2)
    ax[0].hist(corpus_length_histogram, list(range(0, max(corpus_length_histogram) + 10, 1)))
    ax[1].bar(list(range(0, len(corpus_backup))), corpus_length_histogram)

    return corpus_ignore

def cut_corpus(corpus,cut_length):
    new_corpus=[]
    cut_length=cut_length
    print(len(corpus))
    for i in range(len(corpus)):
        lst=corpus[i]
        n=len(lst)
        if n<cut_length:
            new_corpus.append(lst)
            continue
        cut_amount=int((n-n%cut_length)/cut_length)
        for j in range(cut_amount-1):
            new_corpus.append(lst[j*cut_length:(j+1)*cut_length])
        new_corpus.append(lst[(cut_amount-1)*cut_length:])
    print(len(new_corpus))
    return new_corpus

def get_batches(word_to_index, context_tuple_list, device, batch_size=100):
    random.shuffle(context_tuple_list)
    batches = []
    batch_target, batch_context, batch_negative = [], [], []
    for i in range(len(context_tuple_list)):
        batch_target.append(word_to_index[context_tuple_list[i][0]])
        batch_context.append(word_to_index[context_tuple_list[i][1]])
        batch_negative.append([word_to_index[w] for w in context_tuple_list[i][2]])
        if (i+1) % batch_size == 0 or i == len(context_tuple_list)-1:
            tensor_target = torch.from_numpy(np.array(batch_target)).long().to(device)
            tensor_context = torch.from_numpy(np.array(batch_context)).long().to(device)
            tensor_negative = torch.from_numpy(np.array(batch_negative)).long().to(device)
            batches.append((tensor_target, tensor_context, tensor_negative))
            batch_target, batch_context, batch_negative = [], [], []
    return batches

def sample_negative(corpus_ignore, sample_size):
    sample_probability = {}
    word_counts = dict(Counter(list(itertools.chain.from_iterable(corpus_ignore))))
    normalizing = sum([v**0.75 for v in word_counts.values()])
    for word in word_counts:
        sample_probability[word] = word_counts[word]**0.75 / normalizing
    words = np.array(list(word_counts.keys()))
    while True:
        word_list = []
        sampled_index = np.array(multinomial(sample_size, list(sample_probability.values())))
        for index, count in enumerate(sampled_index):
            for _ in range(count):
                 word_list.append(words[index])
        yield word_list

class Word2Vec(nn.Module):

    def __init__(self, embedding_size, vocab_size):
        super(Word2Vec, self).__init__()
        self.target = nn.Embedding(vocab_size, embedding_size).cuda()
        self.context = nn.Embedding(vocab_size, embedding_size).cuda()

    def forward(self, target_word, context_word, negative_example):
        emb_target = self.target(target_word)
        emb_context = self.context(context_word)
        emb_product = torch.mul(emb_target, emb_context).cuda()
        emb_product = torch.sum(emb_product, dim=1).cuda()
        out = torch.sum(F.logsigmoid(emb_product)).cuda()
        emb_negative = self.context(negative_example)
        emb_product = torch.bmm(emb_negative, emb_target.unsqueeze(2)).cuda()
        emb_product = torch.sum(emb_product, dim=1).cuda()
        out += torch.sum(F.logsigmoid(-emb_product)).cuda()
        return -out


class EarlyStopping():
    def __init__(self, patience=5, min_percent_gain=0.1):
        self.patience = patience
        self.loss_list = []
        self.min_gain = min_percent_gain / 100.

    def update_loss(self, loss):
        self.loss_list.append(loss)
        if len(self.loss_list) > self.patience:
            del self.loss_list[0]

    def stop(self):
        if len(self.loss_list) == 1:
            return False
        gain = (max(self.loss_list) - min(self.loss_list)) / max(self.loss_list)
        print("Loss gain: {}%".format(gain * 100))
        if gain < self.min_gain:
            return True
        else:
            return False

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

    filament_corpus= import_metafile(file_path, datatype)

    vocabulary = set(itertools.chain.from_iterable(filament_corpus))
    vocabulary_size = len(vocabulary)
    print('number of 2D classes: %d' % vocabulary_size)

    word_to_index = {w: idx for (idx, w) in enumerate(vocabulary)}
    index_to_word = {idx: w for (idx, w) in enumerate(vocabulary)}

    w = args.window
    context_tuple_list = []
    negative_samples = sample_negative(filament_corpus, args.negative)

    for text in filament_corpus:
        for i, word in enumerate(text):
            if word == 0:
                print(type(word))
                continue
            first_context_word_index = max(0, i - w)
            last_context_word_index = min(i + w, len(text))
            for j in range(first_context_word_index, last_context_word_index):
                neighbor = text[j]
                if j == 0:
                    continue
                if neighbor == 0:
                    continue
                if i != j:
                    context_tuple_list.append((word, neighbor, next(negative_samples)))
    print("There are %d pairs of target and context words" % (len(context_tuple_list)))

    embedding_size = args.emb_size
    loss_function = nn.CrossEntropyLoss()
    net = Word2Vec(embedding_size=embedding_size, vocab_size=vocabulary_size)
    optimizer = optim.Adam(net.parameters())
    early_stopping = EarlyStopping(patience=5, min_percent_gain=args.min_gain)

    n = 0
    while True:
        n = n + 1
        print(n)
        losses = []
        context_tuple_batches = get_batches(word_to_index, context_tuple_list, device, batch_size=args.batch)
        for i in range(len(context_tuple_batches)):
            net.zero_grad()
            target_tensor, context_tensor, negative_tensor = context_tuple_batches[i]
            loss = net(target_tensor, context_tensor, negative_tensor)
            loss.backward()
            optimizer.step()
            losses.append(loss.data)
        print("Loss: ", torch.mean(torch.stack(losses)), torch.stack(losses)[0], torch.stack(losses)[-1])
        early_stopping.update_loss(torch.mean(torch.stack(losses)))
        if early_stopping.stop() is True:
            break
        if torch.mean(torch.stack(losses)) < 10:
            break

    EMBEDDINGS = net.target.weight.data.cpu().numpy()
    print('EMBEDDINGS.shape: ', EMBEDDINGS.shape)
    EMBEDDINGS_np = np.array(EMBEDDINGS)

    # average filament embedding
    average_method = 0  # 0 is average, 1 is weight average
    filament_score = []
    all_filament_data = []
    filament_variance = []
    for filament in filament_corpus:
        score = torch.zeros(embedding_size)
        counts = 0
        filament_list = []
        for i in filament:
            if i == 0:
                continue
            counts += 1
            filament_list.append(EMBEDDINGS[word_to_index[i]])
        if len(filament_list) == 0:
            print('no')
        filament_list = np.array(filament_list)
        # if len(filament_list)==1:
        #    filament_variance.append(float(0))
        # else:
        #    pca=PCA(n_components=1).fit(filament_list)
        #    filament_variance.append(pca.singular_values_[0])
        mean = filament_list.mean(axis=0)
        all_filament_data.append(filament_list)
        if average_method == 0:
            filament_score.append(mean)
        elif average_method == 1:
            dim = len(filament_list[0])
            filament_normalized = np.exp(
                -0.5 * ((filament_list - mean) @ (filament_list - mean).T * 0)).diagonal() / np.sqrt(
                np.pi ** dim * 0.05)
            filament_normalized = filament_normalized / filament_normalized.sum()
            score = filament_normalized @ filament_list
            if counts <= 2:
                continue
            filament_score.append(np.array(score))
    all_data = filament_score[:]
    all_data.extend(EMBEDDINGS_np)
    filament_number = len(filament_score)
    print(filament_number)
    np.save(output_path + '/filament_score.npy', np.array(filament_score))


    #pca_sum = PCA(n_components=2).fit_transform(all_data)
    #pca_sum_3D = PCA(n_components=args.pca_dim).fit_transform(all_data)
#
    #print('PCA_finished')
    ## cluster_pca = KMeans(n_clusters=3).fit_predict(pca_sum[0:len(corpus)])
#
    #plt.figure(figsize=(20, 20))
    #plt.scatter(pca_sum[:, 0], pca_sum[:, 1], alpha=0.3, color='blue')
    #for i in range(len(EMBEDDINGS_np)):
    #    plt.scatter(pca_sum[i + filament_number][0], pca_sum[i + filament_number][1], color='black', marker='*')
    #    plt.annotate(index_to_word[i], xy=(pca_sum[i + filament_number][0], pca_sum[i + filament_number][1]),
    #                 ha='right', va='bottom')
    #plt.savefig(output_path + '/' + os.path.splitext(file_name)[0] + "_pca.png", bbox_inches='tight', pad_inches=0.01)

    n_neighbors = args.n_neighbors
    min_dist = args.min_dist
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    umap_2D = reducer.fit_transform(all_data)
    #umap_3D = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=3).fit_transform(pca_sum_3D)
    # filament_umap_ND=umap.UMAP(n_neighbors=n_neighbors,min_dist=min_dist,n_components=50).fit_transform(all_data)[0:filament_number]
    filament_umap = umap_2D[0:filament_number]
    #filament_umap_3D = umap_3D[0:filament_number]

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
