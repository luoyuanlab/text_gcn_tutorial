from layers import *
from metrics import *
import tensorflow as tf
from collections import defaultdict

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []

        self.inputs = None
        self.outputs = None

        self.loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        """ Wrapper for _build() """
        with tf.variable_scope(self.name):
            self._build()

        # Build sequential layer model
        self.activations.append(self.inputs)
        for layer in self.layers:
            hidden = layer(self.activations[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]

        # Store model variables for easy access
        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

    def save(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = saver.save(sess, "tmp/%s.ckpt" % self.name)
        print("Model saved in file: %s" % save_path)

    def load(self, sess=None):
        if not sess:
            raise AttributeError("TensorFlow session not provided.")
        saver = tf.train.Saver(self.vars)
        save_path = "tmp/%s.ckpt" % self.name
        saver.restore(sess, save_path)
        print("Model restored from file: %s" % save_path)



class GCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(GCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.input_dim = input_dim
        # self.input_dim = self.inputs.get_shape().as_list()[1]  # To be supported in future Tensorflow versions
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.placeholders = placeholders

        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss
        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.outputs, self.placeholders['labels'], self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.outputs, self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.pred = tf.argmax(self.outputs, 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)


    def _build(self):

        self.layers.append(GraphConvolution(input_dim=self.input_dim,
                                            output_dim=FLAGS.hidden1,
                                            placeholders=self.placeholders,
                                            act=tf.nn.relu,
                                            dropout=True,
                                            sparse_inputs=True,
                                            featureless=False,
                                            logging=self.logging))

        self.layers.append(GraphConvolution(input_dim=FLAGS.hidden1,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x, #
                                            dropout=True,
                                            logging=self.logging))
        
        

    def predict(self):
        
        return tf.nn.softmax(self.outputs)


class RGCN(Model):
    def __init__(self, placeholders, num_feat, nonzero_feat, edge_types, **kwargs):
        super(RGCN, self).__init__(**kwargs)
        self.edge_types = edge_types
        self.num_edge_types = sum(self.edge_types.values())
        self.num_obj_types = max([i for i, _ in self.edge_types]) + 1
        self.inputs = {i: placeholders['feat_%d' % i] for i, _ in self.edge_types}
        self.input_dim = num_feat
        self.output_dim = placeholders['labels'].get_shape().as_list()[1]
        self.nonzero_feat = nonzero_feat
        self.placeholders = placeholders
        self.dropout = placeholders['dropout']
        self.adj_mats = {et: [
            placeholders['adj_mats_%d,%d,%d' % (et[0], et[1], k)] for k in range(n)]
            for et, n in self.edge_types.items()}
        print(self.adj_mats)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def _build(self):
        self.hidden1 = defaultdict(list)
        for i, j in self.edge_types:
            self.hidden1[i].append(GraphConvolutionSparseMulti(
                input_dim=self.input_dim, output_dim=FLAGS.hidden1,
                edge_type=(i,j), num_types=self.edge_types[i,j],
                adj_mats=self.adj_mats, nonzero_feat=self.nonzero_feat,
                act=lambda x: x, dropout=self.dropout,
                logging=self.logging)(self.inputs[j]))

        for i, hid1 in self.hidden1.items():
            self.hidden1[i] = tf.nn.relu(tf.add_n(hid1))

        self.embeddings_reltyp = defaultdict(list)
        for i, j in self.edge_types:
            # i, j identity, featureless = True
            self.embeddings_reltyp[i].append(GraphConvolutionMulti(
                input_dim=FLAGS.hidden1, output_dim=self.output_dim,
                edge_type=(i,j), num_types=self.edge_types[i,j],
                adj_mats=self.adj_mats, act=lambda x: x,
                dropout=self.dropout, logging=self.logging)(self.hidden1[j]))

        self.embeddings = [None] * self.num_obj_types
        for i, embeds in self.embeddings_reltyp.items():
            # self.embeddings[i] = tf.nn.relu(tf.add_n(embeds))
            self.embeddings[i] = tf.add_n(embeds)
    
        
    def _loss(self):

        # Cross entropy error
        self.loss += masked_softmax_cross_entropy(self.embeddings[0], self.placeholders['labels'], self.placeholders['labels_mask'])

    def _accuracy(self):
        self.accuracy = masked_accuracy(self.embeddings[0], self.placeholders['labels'],
                                        self.placeholders['labels_mask'])
        self.pred = tf.argmax(self.embeddings[0], 1)
        self.labels = tf.argmax(self.placeholders['labels'], 1)

