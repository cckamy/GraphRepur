# from __future__ import division
# from __future__ import print_function

import os
import time
import tensorflow as tf
import numpy as np
from sklearn import metrics

import json
from networkx.readwrite import json_graph

from graphsage.supervised_models import SupervisedGraphsage
from graphsage.models import SAGEInfo
from graphsage.minibatch import NodeMinibatchIterator
from graphsage.neigh_samplers import UniformNeighborSampler

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"

from optparse import OptionParser

import warnings
warnings.filterwarnings("ignore")

parser = OptionParser()
parser.add_option("-M", "--Model", default='gcn', help="Aggregator")
parser.add_option("-k", "--kw", default='combined_score', help="Drug-drug links keywords")
parser.add_option("-t", "--threshold", default=0.05, help="Differentially expressed genes threshold")
parser.add_option("-L", "--LOSS", default='focal', help="Loss function")
parser.add_option("-F", "--focal_alpha", default=0.75, help="Loss parameter")
parser.add_option("-c", "--checkpt_file", default='pre_trained_Repur/mod.ckpt', help="The checkpt file path")
parser.add_option("-p","--path",default='data/example_data', help="The input data path") #1
parser.add_option("-l","--lr",default=0.01,help="Learning rate")
parser.add_option("-s","--sam",default = (25,10,0),help="Sampling number")
parser.add_option("-D","--Dim",default = (256,256),help="Hidden units")
parser.add_option("-b","--bz",default = 64,help="Batch Size")
parser.add_option("-d","--drop",default = 0.2,help="Dropout")

(opts, args) = parser.parse_args()

Model=opts.Model#graphsage_mean,gcn, graphsage_seq,graphsage_maxpool,graphsage_meanpool
kw,deg_threshold=opts.kw,opts.threshold  #'similarity', 'experimental', 'database', 'textmining', 'combined_score'
LOSS=opts.LOSS # sigmoid, softmax, focal
focal_alpha=opts.focal_alpha

checkpt_file=opts.checkpt_file
data_fn=opts.path

lr,sam,dim,bz,drop=opts.lr,opts.sam,opts.Dim,opts.bz,opts.drop

seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

gpu_num=3
flags = tf.app.flags
FLAGS = flags.FLAGS
max_epoch=1000
patience=50

GPU_MEM_FRACTION = 0.8

def calc_f1(y_true, y_pred):
    if FLAGS.sigmoid != 'sigmoid':
        y_true = np.argmax(y_true, axis=1)
        y_pred = np.argmax(y_pred, axis=1)
    else:
        y_pred[y_pred > 0.5] = 1
        y_pred[y_pred <= 0.5] = 0
    return metrics.f1_score(y_true, y_pred, average="micro"), metrics.f1_score(y_true, y_pred, average="macro")

def calc_roc(y_true, y_pred): 
    acc = metrics.accuracy_score(y_true[:,1],  y_pred.argmax(axis=1)) 
    roc = metrics.roc_auc_score(y_true,  y_pred)
    precision, recall, _ = metrics.precision_recall_curve(y_true[:,1],  y_pred[:,1])
    aupr = metrics.auc(recall, precision)
    return acc,roc,aupr

# Define model evaluation function
def evaluate(sess, model, minibatch_iter, size=None):
    feed_dict_val, labels = minibatch_iter.node_val_feed_dict(size)
    node_outs_val = sess.run([model.preds, model.loss],
                        feed_dict=feed_dict_val)
    mic, mac=0,0
    acc, roc,aupr=calc_roc(labels, node_outs_val[0])
    return node_outs_val[1], node_outs_val[0],mic, mac, acc, roc,aupr,labels

def incremental_evaluate(sess, model, minibatch_iter, size, test=False):
    val_losses = []
    val_preds = []
    labels = []
    iter_num = 0
    finished = False
    while not finished:
        feed_dict_val, batch_labels, finished, _  = minibatch_iter.incremental_node_val_feed_dict(size, iter_num, test=test)
        node_outs_val = sess.run([model.preds, model.loss],
                         feed_dict=feed_dict_val)
        val_preds.append(node_outs_val[0])
        labels.append(batch_labels)
        val_losses.append(node_outs_val[1])
        iter_num += 1
    val_preds = np.vstack(val_preds)
    labels = np.vstack(labels)
    f1_scores=0,0
    acc, roc,aupr=calc_roc(labels, val_preds)
    return np.mean(val_losses), val_preds,f1_scores[0], f1_scores[1],acc, roc,aupr,labels

def construct_placeholders(num_classes):
    # Define placeholders
    placeholders = {
        'labels' : tf.compat.v1.placeholder(tf.float32, shape=(None, num_classes), name='labels'),
        'batch' : tf.compat.v1.placeholder(tf.int32, shape=(None), name='batch1'),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=(), name='dropout'),
        'batch_size' : tf.compat.v1.placeholder(tf.int32, name='batch_size'),
    }
    return placeholders

def load_data(prefix, normalize=False, load_walks=False):
    G_data = json.load(open(prefix +".json"))
    G = json_graph.node_link_graph(G_data)
    feats=np.array([G_data['nodes'][i]['feature'] for i in range(len(G_data['nodes']))])
    class_map={i:G_data['nodes'][i]['label'] for i in range(len(G_data['nodes']))}
    id_map={i:i for i in range(len(G_data['nodes']))}
    walks = []
    return G, feats, id_map, walks, class_map

print("Loading training data..")
train_data = load_data(data_fn)

tf.compat.v1.reset_default_graph()
tf.app.flags.DEFINE_boolean('log_device_placement', False, """Whether to log device placement.""")
#core params..
flags.DEFINE_string('model', Model, 'model names. See README for possible values.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_integer('epochs', max_epoch, 'number of epochs to train.')
flags.DEFINE_float('focal_alpha',focal_alpha, 'alpha of Focal')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
#flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_string('loss', LOSS, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0, 'Set to positive value to use identity embedding features of that dimension. Default 0.')
#logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 128, "how many nodes per validation sample.")
flags.DEFINE_integer('print_every', 20, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10**10, "Maximum total number of iterations")
flags.DEFINE_integer('gpu', gpu_num, "which gpu to use.")
os.environ["CUDA_VISIBLE_DEVICES"]=str(FLAGS.gpu)

flags.DEFINE_float('learning_rate', lr, 'initial learning rate.')
flags.DEFINE_float('dropout', drop, 'dropout rate (1 - keep probability).')
flags.DEFINE_integer('samples_1', sam[0], 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', sam[1], 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', sam[2], 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', dim[0], 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', dim[1], 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('batch_size', bz, 'minibatch size.')

G = train_data[0]
features = train_data[1]
id_map = train_data[2]
class_map  = train_data[4]

if isinstance(list(class_map.values())[0], list):
    num_classes = len(list(class_map.values())[0])
else:
    num_classes = len(set(class_map.values()))

if not features is None:
    features = np.vstack([features, np.zeros((features.shape[1],))])

context_pairs = train_data[3] if FLAGS.random_context else None
placeholders = construct_placeholders(num_classes)
minibatch = NodeMinibatchIterator(G,
        id_map,
        placeholders,
        class_map,
        num_classes,
        batch_size=FLAGS.batch_size,
        max_degree=FLAGS.max_degree,
        context_pairs = context_pairs)
adj_info_ph = tf.compat.v1.placeholder(tf.int32, shape=minibatch.adj.shape)
adj_info = tf.Variable(adj_info_ph, trainable=False, name="adj_info")

if FLAGS.model == 'graphsage_mean':
    # Create model
    sampler = UniformNeighborSampler(adj_info)
    if FLAGS.samples_3 != 0:
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2),
                            SAGEInfo("node", sampler, FLAGS.samples_3, FLAGS.dim_2)]
    elif FLAGS.samples_2 != 0:
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                            SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]
    else:
        layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1)]

    model = SupervisedGraphsage(num_classes, placeholders,
                                 features,
                                 adj_info,
                                 minibatch.deg,
                                 layer_infos,
                                 f_alpha=FLAGS.focal_alpha,
                                 model_size=FLAGS.model_size,
                                 sigmoid_loss = FLAGS.loss,
                                 identity_dim = FLAGS.identity_dim,
                                 logging=True)

elif FLAGS.model == 'gcn':
    # Create model
    sampler = UniformNeighborSampler(adj_info)
    layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, 2*FLAGS.dim_1),
                        SAGEInfo("node", sampler, FLAGS.samples_2, 2*FLAGS.dim_2)]

    model = SupervisedGraphsage(num_classes, placeholders,
                             features,
                             adj_info,
                             minibatch.deg,
                             layer_infos,
                             f_alpha=FLAGS.focal_alpha,
                             model_size=FLAGS.model_size,
                             sigmoid_loss = FLAGS.loss,
                             identity_dim = FLAGS.identity_dim,
                             logging=True)

elif FLAGS.model == 'graphsage_seq':
    sampler = UniformNeighborSampler(adj_info)
    layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                        SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

    model = SupervisedGraphsage(num_classes, placeholders,
                                 features,
                                 adj_info,
                                 minibatch.deg,
                                 layer_infos=layer_infos,
                                 f_alpha=FLAGS.focal_alpha,
                                 aggregator_type="seq",
                                 model_size=FLAGS.model_size,
                                 sigmoid_loss = FLAGS.loss,
                                 identity_dim = FLAGS.identity_dim,
                                 logging=True)

elif FLAGS.model == 'graphsage_maxpool':
    sampler = UniformNeighborSampler(adj_info)
    layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                        SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

    model = SupervisedGraphsage(num_classes, placeholders,
                                features,
                                adj_info,
                                minibatch.deg,
                                 layer_infos=layer_infos,
                                 f_alpha=FLAGS.focal_alpha,
                                 aggregator_type="maxpool",
                                 model_size=FLAGS.model_size,
                                 sigmoid_loss = FLAGS.loss,
                                 identity_dim = FLAGS.identity_dim,
                                 logging=True)

elif FLAGS.model == 'graphsage_meanpool':
    sampler = UniformNeighborSampler(adj_info)
    layer_infos = [SAGEInfo("node", sampler, FLAGS.samples_1, FLAGS.dim_1),
                        SAGEInfo("node", sampler, FLAGS.samples_2, FLAGS.dim_2)]

    model = SupervisedGraphsage(num_classes, placeholders,
                                features,
                                adj_info,
                                minibatch.deg,
                                 layer_infos=layer_infos,
                                 f_alpha=FLAGS.focal_alpha,
                                 aggregator_type="meanpool",
                                 model_size=FLAGS.model_size,
                                 sigmoid_loss = FLAGS.loss,
                                 identity_dim = FLAGS.identity_dim,
                                 logging=True)

else:
    raise Exception('Error: model name unrecognized.')

config = tf.compat.v1.ConfigProto(log_device_placement=FLAGS.log_device_placement,
                        device_count={"CPU": 1})
config.gpu_options.allow_growth = True
config.allow_soft_placement = True

# Initialize session
saver = tf.compat.v1.train.Saver()
sess = tf.compat.v1.Session(config=config)

# Init variables
sess.run(tf.compat.v1.global_variables_initializer(), feed_dict={adj_info_ph: minibatch.adj})

train_adj_info = tf.compat.v1.assign(adj_info, minibatch.adj)
val_adj_info = tf.compat.v1.assign(adj_info, minibatch.test_adj)

saver.restore(sess, checkpt_file)
sess.run(val_adj_info.op)
val_cost, val_pred,val_f1_mic, val_f1_mac, val_acc, val_roc,val_aupr, val_lbl = \
    incremental_evaluate(sess, model, minibatch, FLAGS.batch_size)
print("Full validation stats:",
              "loss=", "{:.5f}".format(val_cost),
              "acc =", "{:.5f}".format(val_acc),
              "roc =", "{:.5f}".format(val_roc),
              "aupr=", "{:.5f}".format(val_aupr))

ts_cost, ts_pred,ts_f1_mic, ts_f1_mac, ts_acc, ts_roc,ts_aupr, ts_lbl = \
    incremental_evaluate(sess, model, minibatch, FLAGS.batch_size, test=True)
print("Full test stats:",
              "loss=", "{:.5f}".format(ts_cost),
              "acc =", "{:.5f}".format(ts_acc),
              "roc =", "{:.5f}".format(ts_roc),
              "aupr=", "{:.5f}".format(ts_aupr))
sess.close()
exit()