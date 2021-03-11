import tensorflow as tf
import numpy as np

import graphsage.models as models
import graphsage.layers as layers
from graphsage.aggregators import MeanAggregator, MaxPoolingAggregator, MeanPoolingAggregator, SeqAggregator, GCNAggregator

flags = tf.app.flags
FLAGS = flags.FLAGS

tf.compat.v1.set_random_seed(0)
import warnings
warnings.filterwarnings("ignore")
class SupervisedGraphsage(models.SampleAndAggregate):
    """Implementation of supervised GraphSAGE."""

    def __init__(self, num_classes,
            placeholders, features, adj,degrees,
            layer_infos,f_alpha, concat=True, aggregator_type="mean",
            model_size="small", sigmoid_loss=False, identity_dim=0,
                **kwargs):
        '''
        Args:
            - placeholders: Stanford TensorFlow placeholder object.
            - features: Numpy array with node features.
            - adj: Numpy array with adjacency lists (padded with random re-samples)
            - degrees: Numpy array with node degrees. 
            - layer_infos: List of SAGEInfo namedtuples that describe the parameters of all 
                   the recursive layers. See SAGEInfo definition above.
            - concat: whether to concatenate during recursive iterations
            - aggregator_type: how to aggregate neighbor information
            - model_size: one of "small" and "big"
            - sigmoid_loss: Set to true if nodes can belong to multiple classes
        '''

        models.GeneralizedModel.__init__(self, **kwargs)

        if aggregator_type == "mean":
            self.aggregator_cls = MeanAggregator
        elif aggregator_type == "seq":
            self.aggregator_cls = SeqAggregator
        elif aggregator_type == "meanpool":
            self.aggregator_cls = MeanPoolingAggregator
        elif aggregator_type == "maxpool":
            self.aggregator_cls = MaxPoolingAggregator
        elif aggregator_type == "gcn":
            self.aggregator_cls = GCNAggregator
        else:
            raise Exception("Unknown aggregator: ", self.aggregator_cls)

        # get info from placeholders...
        self.inputs1 = placeholders["batch"]
        self.model_size = model_size
        self.adj_info = adj
        #self.adj_ext_info = adj_ext
        if identity_dim > 0:
           self.embeds = tf.get_variable("node_embeddings", [adj.get_shape().as_list()[0], identity_dim])
        else:
           self.embeds = None
        if features is None: 
            if identity_dim == 0:
                raise Exception("Must have a positive value for identity feature dimension if no input features given.")
            self.features = self.embeds
            #self.features_ext = self.embeds
        else:
            self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False)
            #self.features_ext = tf.Variable(tf.constant(features_ext, dtype=tf.float32), trainable=False)
            if not self.embeds is None:
                self.features = tf.concat([self.embeds, self.features], axis=1)
                #self.features_ext = tf.concat([self.embeds, self.features_ext], axis=1)
        self.degrees = degrees
        self.concat = concat
        self.f_alpha = f_alpha
        self.num_classes = num_classes
        self.sigmoid_loss = sigmoid_loss
        self.dims = [(0 if features is None else features.shape[1]) + identity_dim]
        self.dims.extend([layer_infos[i].output_dim for i in range(len(layer_infos))])
        self.batch_size = placeholders["batch_size"]
        self.placeholders = placeholders
        self.layer_infos = layer_infos

        self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
        self.build()

    def build(self):
        samples1, support_sizes1 = self.sample(self.inputs1, self.layer_infos)
        num_samples = [layer_info.num_samples for layer_info in self.layer_infos]
        self.outputs1, self.aggregators = self.aggregate(samples1, [self.features], self.dims, num_samples,
                support_sizes1, concat=self.concat, model_size=self.model_size)

        dim_mult = 2 if self.concat else 1

        self.outputs1 = tf.nn.l2_normalize(self.outputs1, 1)

        dim_mult = 2 if self.concat else 1
        self.node_pred = layers.Dense(dim_mult*self.dims[-1], self.num_classes, dropout=self.placeholders['dropout'],act=lambda x : x)
        # TF graph management
        self.node_preds = self.node_pred(self.outputs1)

        self._loss()
        grads_and_vars = self.optimizer.compute_gradients(self.loss)
        clipped_grads_and_vars = [(tf.clip_by_value(grad, -5.0, 5.0) if grad is not None else None, var) 
                for grad, var in grads_and_vars]
        self.grad, _ = clipped_grads_and_vars[0]
        self.opt_op = self.optimizer.apply_gradients(clipped_grads_and_vars)
        self.preds = self.predict()

    def focal_loss(self,logits, labels, focal_alpha, gamma=2, epsilon=1e-6):
    #def focal_loss(logits, labels, alpha=0.25, gamma=2, epsilon=1e-6):
        r"""Compute focal loss for predictions.
            Multi-labels Focal loss formula:
                FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                     ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
        Args:
         pred: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing the predicted logits for each class
         y: A float tensor of shape [batch_size, num_anchors,
            num_classes] representing one-hot encoded classification targets
         alpha: A scalar tensor for focal loss alpha hyper-parameter
         gamma: A scalar tensor for focal loss gamma hyper-parameter
        Returns:
            loss: A (scalar) tensor representing the value of the loss function
        labels= np.array([[1,0],[0,1],[1,0],[0,1]])#
        logits =np.array([[12,3],[3,10],[4,6.5],[3,6]])
        alpha,gamma,epsilon=1,0,1e-6
        alpha,gamma,epsilon=0.25,0,1e-6
        """
        zeros = tf.zeros_like(logits, dtype=logits.dtype) # labels.dtype
        softmax_logits=tf.nn.softmax(logits)
        pos_p_sub = tf.where(labels > zeros, labels - softmax_logits, zeros) # positive sample
        neg_p_sub = tf.where(labels > zeros, zeros, softmax_logits) # negative sample
        f1_loss = - focal_alpha * (pos_p_sub ** gamma) * tf.math.log(tf.clip_by_value(softmax_logits, epsilon, 1.0-epsilon)) \
                              - (1 - focal_alpha) * (neg_p_sub ** gamma) * tf.math.log(tf.clip_by_value(1.0-softmax_logits,epsilon,1.0-epsilon))
        loss=tf.gather_nd(f1_loss,tf.where(labels>zeros))
        return loss#,f1_loss

    def _loss(self):
        # Weight decay loss
        for aggregator in self.aggregators:
            for var in aggregator.vars.values():
                self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
        for var in self.node_pred.vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)
       
        # classification loss
        if self.sigmoid_loss=='sigmoid':
            self.loss += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
        if self.sigmoid_loss=='softmax':
            self.loss += tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=self.node_preds,
                    labels=self.placeholders['labels']))
        if self.sigmoid_loss=='focal':
            self.loss += tf.reduce_mean(self.focal_loss(logits=self.node_preds,\
                                                        labels=self.placeholders['labels'], focal_alpha=self.f_alpha))
        tf.compat.v1.summary.scalar('loss', self.loss)

    def predict(self):
        if self.sigmoid_loss=='sigmoid':
            return tf.nn.sigmoid(self.node_preds)
        if self.sigmoid_loss=='softmax' or self.sigmoid_loss=='focal':
            return tf.nn.softmax(self.node_preds)
