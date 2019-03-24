import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
import re
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(
        "data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(
            min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    # print(len(labels))

    idx_test = test_idx_range.tolist()
    # print(idx_test)
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def load_corpus(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj = tuple(objects)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    features = sp.vstack((allx, tx)).tolil()
    labels = np.vstack((ally, ty))
    print(len(labels))

    train_idx_orig = parse_index_file(
        "data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size


def load_corpus_multimodal(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both training and val docs
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.word_adj => adjacency matrix of word nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.doc_adj => adjacency matrix of doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.doc_word_adj => adjacency matrix for doc and word nodes as scipy.sparse.csr.csr_matrix object;

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'word_adj', 'doc_adj', 'doc_word_adj', 'word_feat']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, word_adj, doc_adj, doc_word_adj, word_feat = tuple(objects)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    labels = np.vstack((ally, ty))
    print(len(labels))

    train_idx_orig = parse_index_file(
        "data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    #word_adj = preprocess_graph(word_adj)
    #doc_adj = preprocess_graph(doc_adj)
    #doc_word_adj = preprocess_graph(doc_word_adj)
    #adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return word_adj, doc_adj, doc_word_adj, word_feat, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size

def load_corpus_kg(dataset_str):
    """
    Loads input corpus from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test docs as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training docs/words
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training docs as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test docs as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.adj => adjacency matrix of word/doc nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.train.index => the indices of training docs in original doc list.
    ind.dataset_str.word_entity_adj => adjacency matrix for word and entity nodes as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.entity_adj_list => adjacency matrix list for knowledge graph triples (one for each relation)
                                       as a list of scipy.sparse.csr.csr_matrix objects;

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'adj', 'word_entity_adj', 'entity_adj_list']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, adj, word_entity_adj, entity_adj_list = tuple(objects)
    print(x.shape, y.shape, tx.shape, ty.shape, allx.shape, ally.shape)

    labels = np.vstack((ally, ty))
    print(len(labels))

    train_idx_orig = parse_index_file(
        "data/{}.train.index".format(dataset_str))
    train_size = len(train_idx_orig)

    val_size = train_size - x.shape[0]
    test_size = tx.shape[0]

    idx_train = range(len(y))
    idx_val = range(len(y), len(y) + val_size)
    idx_test = range(allx.shape[0], allx.shape[0] + test_size)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    print(y_train.shape)

    return adj, word_entity_adj, entity_adj_list, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size

def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx

def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj, symmetric=True):
    if symmetric:
        d = sp.diags(np.power(np.array(adj.sum(1)), -0.5).flatten())
        a_norm = adj.dot(d).transpose().dot(d).tocsr()
    else:
        d = sp.diags(np.power(np.array(adj.sum(1)), -1).flatten())
        a_norm = d.dot(adj).tocsr()
    return a_norm

def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    return sparse_to_tuple(adj_normalized)

def preprocess_graph(adj, symmetric=True):
    # this function has bugs, return none, decagon defines and do this immediatly. here we load from pkl
    adj = sp.coo_matrix(adj)
    if adj.shape[0] == adj.shape[1]:
        if symmetric == True:
            adj_ = adj + sp.eye(adj.shape[0])
            rowsum = np.array(adj_.sum(1))
            degree_inv_sqrt = np.power(rowsum, -0.5).flatten()
            degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
            degree_mat_inv_sqrt = sp.diags(degree_inv_sqrt)
            adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
        else:
            degree_inv_sqrt = np.power(np.array(adj.sum(1)), -1).flatten()
            degree_inv_sqrt[np.isinf(degree_inv_sqrt)] = 0.
            degree_mat_inv_sqrt = sp.diags(degree_inv_sqrt)
            adj_normalized = degree_mat_inv_sqrt.dot(adj).tocsr()
    else:
        rowsum = np.array(adj.sum(1))
        rowdegree_inv = np.power(rowsum, -0.5).flatten()
        rowdegree_inv[np.isinf(rowdegree_inv)] = 0.
        rowdegree_mat_inv = sp.diags(rowdegree_inv)

        colsum = np.array(adj.sum(0))
        coldegree_inv = np.power(colsum, -0.5).flatten()
        coldegree_inv[np.isinf(coldegree_inv)] = 0.
        coldegree_mat_inv = sp.diags(coldegree_inv)

        adj_normalized = rowdegree_mat_inv.dot(adj).dot(coldegree_mat_inv).tocoo()
        return sparse_to_tuple(adj_normalized)

def construct_feed_dict(features, support, labels, labels_mask, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i]
                      for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    return feed_dict

def build_feed_dict(labels, labels_mask, adj, edge_types, feat, placeholders):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({
        placeholders['adj_mats_%d,%d,%d' % (i,j,k)]: adj[i,j][k]
        for i, j in edge_types for k in range(edge_types[i,j])})
    #print(adj[1,1][0])
    feed_dict.update({placeholders['feat_%d' % i]: feat[i] for i, _ in edge_types})
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    return feed_dict

def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (
        2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)


def loadWord2Vec(filename):
    """Read Word Vectors"""
    vocab = []
    embd = []
    word_vector_map = {}
    file = open(filename, 'r')
    for line in file.readlines():
        row = line.strip().split(' ')
        if(len(row) > 2):
            vocab.append(row[0])
            vector = row[1:]
            length = len(vector)
            for i in range(length):
                vector[i] = float(vector[i])
            embd.append(vector)
            word_vector_map[row[0]] = vector
    print('Loaded Word Vectors!')
    file.close()
    return vocab, embd, word_vector_map

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def word_synonyms(word):
    '''
    look up synonyms given a word
    '''
    synonyms = []
    for syn in wn.synsets(word):
	    for l in syn.lemmas():
		    synonyms.append(l.name())
    return synonyms

def synonimize(word, pos=None):
	""" Get synonyms of the word / lemma """ 
	try:
		# map part of speech tags to wordnet
		pos = {'NN': wn.NOUN,'JJ':wn.ADJ,'VB':wn.VERB,'RB':wn.ADV}[pos[:2]]
	except:
		# or just return the original word
		print("OUCH {} {}".format(word, pos))
		return [word]

	synsets = wn.synsets(word, pos)
	synonyms = []
	for synset in synsets:
		for sim in  synset.similar_tos():
			synonyms += sim.lemma_names()
	
	# return list of synonyms or just the original word
	return synonyms or [word]


def wordnet_id_synset_dict():
    '''
    synset to number mapping
    '''

    f = open('data/WN18/wordnet-mlj12-definitions.txt', 'r')
    lines = f.readlines()
    f.close()

    synset_id_dict = {}

    count = 0
    for line in lines:
        temp = line.strip().split('\t')
        #print(temp[0], temp[1])
        # n, v, a, r
        if temp[1].find('_NN_') != -1 or temp[1].find('_JJ_') != -1 or temp[1].find('_VB_') != -1 or temp[1].find('_RB_') != -1:
            count += 1
            wordnet_str = temp[1][2:]
            num_start = wordnet_str.rfind('_')
            num = wordnet_str[num_start + 1:]
            if len(num) == 1:
                num = '0' + num
            # print(num)
            pos_start = wordnet_str[:num_start].rfind('_')
            pos = wordnet_str[:num_start][pos_start + 1:]
            if pos == 'NN':
                pos = 'n'
            elif pos == 'JJ':
                pos = 'a'
            elif pos == 'VB':
                pos = 'v'
            elif pos == 'RB':
                pos = 'r'
            # print(pos)
            name = wordnet_str[:pos_start]
            # print(name)
            new_str = name + '.' + pos + '.' + num
            # print(new_str, temp[0])
            synset_id_dict[new_str] = temp[0]

            # print(wordnet_str)
            # if wordnet_str.find('10') != -1:
            #    print(wordnet_str, num, pos, name)
        else:
            print(temp[1])
    print(count)

    return synset_id_dict


def wordnet_id_num_dict():
    ''' number to id mapping'''

    f = open('data/WN18/entity2id.txt', 'r')
    lines = f.readlines()
    f.close()

    id_num_dict = {}
    for line in lines:
        temp = line.strip().split('\t')
        if len(temp) == 2:
            #print(temp[0], temp[1])
            id_num_dict[temp[0]] = temp[1]
    return id_num_dict

def wordnet_defs():
    ''' id to definitions '''

    f = open('data/WN18/wordnet-mlj12-definitions.txt', 'r')
    lines = f.readlines()
    f.close()
    
    number_def_dict = {}
    for line in lines:
        temp = line.strip().split('\t')

        number_def_dict[temp[0]] = temp[2]
    id_num_dict = wordnet_id_num_dict()

    id_def_dict = {}
    for num in id_num_dict:
        entity_id = id_num_dict[num]
        definition = number_def_dict[num]
        id_def_dict[entity_id] = definition
    
    def_docs = []
    for i in range(len(id_def_dict)):
        def_docs.append(id_def_dict[str(i)])

    tfidf_vec = TfidfVectorizer()
    tfidf_matrix = tfidf_vec.fit_transform(def_docs)
    return tfidf_matrix


def read_triples(file_path):
    '''read train, val or test triples'''
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    
    triple_list = []
    for line in lines:
        line = line.strip()
        temp = line.split()
        if len(temp) == 3:
            #print(temp[0], temp[1], temp[2])
            triple_list.append(line)
    
    return triple_list