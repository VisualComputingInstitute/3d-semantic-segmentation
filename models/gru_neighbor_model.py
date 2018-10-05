from tools import tf_util
from .multi_block_model import *
from batch_generators import *
from typing import Dict


class GruNeighborModel(MultiBlockModel):
    """
    parameterized version of a neighboring model using GRU units as described in the paper
    """

    def __init__(self, batch_generator: BatchGenerator, params: Dict[str, list]):
        super().__init__(batch_generator)

        self._bn_decay = 0.9

    @lazy_property
    def _prediction_helper(self):
        num_point = self.batch_generator.num_points
        batch_size = self.batch_generator.batch_size

        dims = self.batch_generator.input_shape

        cumulated_batch_size = dims[0] * dims[1]
        time = dims[1]

        input_image = tf.reshape(self.batch_generator.pointclouds_pl, (cumulated_batch_size, dims[2], dims[3]))
        input_image = tf.expand_dims(input_image, -1)

        net = input_image

        # CONV
        net = tf_util.conv2d(net, 64, [1, self.batch_generator.dataset.num_features + 3], padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training_pl, scope='conv1', bn_decay=self._bn_decay)
        net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training_pl, scope='conv2', bn_decay=self._bn_decay)
        net = tf_util.conv2d(net, 64, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training_pl, scope='conv3', bn_decay=self._bn_decay)
        net = tf_util.conv2d(net, 128, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training_pl, scope='conv4', bn_decay=self._bn_decay)
        points_feat1 = tf_util.conv2d(net, 1024, [1, 1], padding='VALID', stride=[1, 1],
                                      bn=True, is_training=self.is_training_pl, scope='conv5', bn_decay=self._bn_decay)
        # MAX
        pc_feat1 = tf_util.max_pool2d(points_feat1, [num_point, 1], padding='VALID', scope='maxpool1')

        # FC
        pc_feat2 = tf.reshape(pc_feat1, [cumulated_batch_size, -1])
        pc_feat2 = tf_util.fully_connected(pc_feat2, 256, bn=True, is_training=self.is_training_pl, scope='fc1', bn_decay=self._bn_decay)
        pc_feat2 = tf_util.fully_connected(pc_feat2, 64, bn=True, is_training=self.is_training_pl, scope='fc2', bn_decay=self._bn_decay)
        tf.summary.histogram("pc_feat2_b", pc_feat2[:])
        pc_feat2_ = tf.reshape(pc_feat2, (batch_size, time, 64))

        pc_feat2_ = tf_util.gru_seq(pc_feat2_, 64, batch_size, time, False, scope='gru1')

        pc_feat2_ = tf.reshape(pc_feat2_, (cumulated_batch_size, 64))
        tf.summary.histogram("pc_feat2_a", pc_feat2_[:])

        pc_feat2 = tf.concat([pc_feat2, pc_feat2_], axis=1)
        # CONCAT
        pc_feat2_expand = tf.tile(tf.reshape(pc_feat2, [cumulated_batch_size, 1, 1, -1]), [1, num_point, 1, 1])
        points_feat2_concat = tf.concat(axis=3, values=[points_feat1, pc_feat2_expand])

        # CONV
        net2 = tf_util.conv2d(points_feat2_concat, 512, [1, 1], padding='VALID', stride=[1, 1],
                              bn=True, is_training=self.is_training_pl, scope='conv6')
        net2 = tf_util.conv2d(net2, 256, [1, 1], padding='VALID', stride=[1, 1],
                              bn=True, is_training=self.is_training_pl, scope='conv7')
        net2 = tf_util.dropout(net2, keep_prob=0.7, is_training=self.is_training_pl, scope='dp1')
        net2 = tf_util.conv2d(net2, self.batch_generator.dataset.num_classes, [1, 1], padding='VALID', stride=[1, 1],
                              activation_fn=None, scope='conv8')
        net2 = tf.squeeze(net2, [2])

        return tf.reshape(net2, (batch_size, time, num_point, -1))
