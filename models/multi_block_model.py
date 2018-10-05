import tensorflow as tf
from abc import *
from tools.lazy_decorator import *


class MultiBlockModel(ABC):

    def __init__(self, batch_generator):
        self.batch_generator = batch_generator

        self._create_placeholders()

    @lazy_function
    def _create_placeholders(self):

        self.eval_per_epoch_pl = tf.placeholder(tf.float32,
                                                name='evaluation_pl',
                                                shape=(3, 1))

        self.block_mask_bool = tf.cast(self.batch_generator.mask_pl, tf.bool)  # shape: (BS, 1)
        self.labels = tf.boolean_mask(self.batch_generator.labels_pl, self.block_mask_bool)  # shape: (BS, N)
        self.labels = tf.cast(self.labels, tf.int32)

        num_blocks = tf.reduce_sum(self.batch_generator.mask_pl)  # num of blocks per batch
        self.num_blocks = tf.cast(num_blocks, tf.float32)

        self.is_training_pl = tf.placeholder(tf.bool,
                                             name='is_training_pl',
                                             shape=())

    @lazy_property
    def prediction(self):
        pred = self._prediction_helper
        # Apply mask to prediction and labels
        return tf.boolean_mask(pred, self.block_mask_bool)  # shape: (BS, N, K)

    @lazy_property
    def prediction_sm(self):
        return tf.nn.softmax(self.prediction)

    @lazy_property
    @abstractmethod
    def _prediction_helper(self):
        raise NotImplementedError('Should be defined in subclass')

    @lazy_property
    def loss(self):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.prediction,
                                                              labels=self.labels)
        return tf.reduce_mean(loss)

    @lazy_property
    def correct(self):
        return tf.equal(tf.argmax(self.prediction, 2), tf.to_int64(self.labels))

    @lazy_property
    def accuracy(self):
        return tf.reduce_sum(tf.cast(self.correct, tf.float32)) / \
               tf.cast(self.batch_generator.num_points * self.num_blocks, tf.float32)

    @lazy_function
    def register_summary(self):
        tf.summary.scalar('loss', self.loss)
        tf.summary.scalar('avg_acc', self.eval_per_epoch_pl[0, 0])
        tf.summary.scalar('avg_iou', self.eval_per_epoch_pl[1, 0])
        tf.summary.scalar('avg_loss', self.eval_per_epoch_pl[2, 0])
        tf.summary.scalar('accuracy', self.accuracy)

    @property
    def input_shape(self):
        return self.batch_generator.pointclouds_pl.get_shape().as_list()

    @property
    def labels_shape(self):
        return self.batch_generator.labels_pl.get_shape().as_list()
