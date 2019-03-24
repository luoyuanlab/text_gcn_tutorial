from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from sklearn import metrics
from utils import *
from models import GCN, RGCN
import random
import os
import sys

if len(sys.argv) != 2:
	sys.exit("Use: python train.py <dataset>")

datasets = ['20ng', 'R8', 'R52', 'ohsumed', 'mr', 'aclImdb', 'ag_news', 'dbpedia']
dataset = sys.argv[1]

if dataset not in datasets:
	sys.exit("wrong dataset name")


# Set random seed
seed = random.randint(1, 200)
tf.set_random_seed(seed)

# Settings
os.environ["CUDA_VISIBLE_DEVICES"] = ""

flags = tf.app.flags
FLAGS = flags.FLAGS
# 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('dataset', dataset, 'Dataset string.')
# 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_string('model', 'gcn', 'Model string.')
flags.DEFINE_float('learning_rate', 0.001, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 200, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0,
                   'Weight for L2 loss on embedding matrix.')  # 5e-4
flags.DEFINE_integer('early_stopping', 10,
                     'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

# Load data
word_adj, doc_adj, doc_word_adj, word_feat, y_train, y_val, y_test, train_mask, val_mask, test_mask, train_size, test_size = load_corpus_multimodal(
    FLAGS.dataset)

vocab_size = word_adj.shape[0]
word_feat = sp.identity(vocab_size)
word_nonzero_feat, word_num_feat = word_feat.shape
word_feat = sparse_to_tuple(word_feat.tocoo())

doc_size = doc_adj.shape[0]
doc_feat = sp.identity(doc_size)
doc_nonzero_feat, doc_num_feat = doc_feat.shape
doc_feat = sparse_to_tuple(doc_feat.tocoo())

word_adj = preprocess_adj(word_adj)
doc_adj = preprocess_adj(doc_adj)

word_doc_sdj = doc_word_adj.T
doc_word_adj = preprocess_graph(doc_word_adj)
word_doc_sdj = preprocess_graph(word_doc_sdj)

print(word_adj)
# data representation
adj_mats_orig = {
    (0, 0): [doc_adj], 
    (0, 1): [doc_word_adj], 
    (1, 0): [word_doc_sdj], 
    (1, 1): [word_adj]
}
num_feat = {
    1: word_num_feat,
    0: doc_num_feat,
}
nonzero_feat = {
    1: word_nonzero_feat,
    0: doc_nonzero_feat,
}
feat = {
    1: word_feat,
    0: doc_feat,
}
#edge_type2dim = {k: [adj.shape for adj in adjs] for k, adjs in adj_mats_orig.items()}
edge_types = {k: len(v) for k, v in adj_mats_orig.items()}

print(edge_types)
num_edge_types = sum(edge_types.values())
print("Edge types:", "%d" % num_edge_types)


# Define model evaluation function
def evaluate(labels, mask, adj_mats_orig, edge_types, feat, placeholders):
    t_test = time.time()
    feed_dict_val = build_feed_dict(labels, mask, adj_mats_orig, edge_types, feat, placeholders)
    outs_val = sess.run([model.loss, model.accuracy, model.pred, model.labels], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], outs_val[2], outs_val[3], (time.time() - t_test)

def construct_placeholders(edge_types):
    # Define placeholders
    placeholders = {
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
    }
    placeholders.update({
        'adj_mats_%d,%d,%d' % (i, j, k): tf.sparse_placeholder(tf.float32)
        for i, j in edge_types for k in range(edge_types[i,j])})
    placeholders.update({
        'feat_%d' % i: tf.sparse_placeholder(tf.float32)
        for i, _ in edge_types})
    return placeholders

placeholders = construct_placeholders(edge_types)
# Create model
model = RGCN(placeholders, num_feat, nonzero_feat, edge_types)

# Initialize session
session_conf = tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))
sess = tf.Session(config=session_conf)
# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    t = time.time()
    # Construct feed dictionary
    #print(adj_mats_orig[0,0])
    feed_dict = build_feed_dict(y_train, train_mask, adj_mats_orig, edge_types, feat, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    # to inductive
    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy, model.embeddings], feed_dict=feed_dict)

    doc_embeddings = outs[3][0]
    word_embeddings = outs[3][1]
    print(len(word_embeddings), len(word_embeddings[0]))
    # Validation
    cost, acc, pred, labels, duration = evaluate(y_val, val_mask, adj_mats_orig, edge_types, feat, placeholders)
    cost_val.append(cost)

    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(
              outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))

    if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
        print("Early stopping...")
        break

print("Optimization Finished!")

# Testing
test_cost, test_acc, pred, labels, test_duration = evaluate(y_test, test_mask, adj_mats_orig, edge_types, feat, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

test_pred = []
test_labels = []
print(len(test_mask))
for i in range(len(test_mask)):
    if test_mask[i]:
        test_pred.append(pred[i])
        test_labels.append(labels[i])

#print(test_labels)
print("Test Precision, Recall and F1-Score...")
print(metrics.classification_report(test_labels, test_pred, digits=4))
print("Macro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='macro'))
print("Micro average Test Precision, Recall and F1-Score...")
print(metrics.precision_recall_fscore_support(test_labels, test_pred, average='micro'))


# doc and word embeddings
print('embeddings:')

train_doc_embeddings = doc_embeddings[:train_size]  # include val docs
test_doc_embeddings = doc_embeddings[train_size:]

print(len(word_embeddings), len(train_doc_embeddings),
      len(test_doc_embeddings))
print(word_embeddings)

f = open('data/corpus/' + dataset + '_vocab.txt', 'r')
words = f.readlines()
f.close()

vocab_size = len(words)
word_vectors = []
for i in range(vocab_size):
    word = words[i].strip()
    word_vector = word_embeddings[i]
    word_vector_str = ' '.join([str(x) for x in word_vector])
    word_vectors.append(word + ' ' + word_vector_str)

word_embeddings_str = '\n'.join(word_vectors)
f = open('data/' + dataset + '_word_vectors.txt', 'w')
f.write(word_embeddings_str)
f.close()

doc_vectors = []
doc_id = 0
for i in range(train_size):
    doc_vector = train_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

for i in range(test_size):
    doc_vector = test_doc_embeddings[i]
    doc_vector_str = ' '.join([str(x) for x in doc_vector])
    doc_vectors.append('doc_' + str(doc_id) + ' ' + doc_vector_str)
    doc_id += 1

doc_embeddings_str = '\n'.join(doc_vectors)
f = open('data/' + dataset + '_doc_vectors.txt', 'w')
f.write(doc_embeddings_str)
f.close()
