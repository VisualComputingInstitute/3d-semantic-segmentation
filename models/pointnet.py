from tools import tf_util
from .multi_block_model import *
from batch_generators import *
from typing import Dict


class Pointnet(MultiBlockModel):
    """
    Pointnet network architecture from
    PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation
    Author: Qi, et al.
    """

    def __init__(self, batch_generator: BatchGenerator, params: Dict[str, list]):
        super().__init__(batch_generator)
        self._bn_decay = 0.9

    @lazy_property
    def _prediction_helper(self):
        # pointcloud placeholder has the following format: BxSxNxF
        # Batch   B
        # Scale   S
        # Point   P
        # Feature F
        num_points = self.batch_generator.pointclouds_pl.get_shape().as_list()[2]

        image_pl = tf.transpose(self.batch_generator.pointclouds_pl, [0, 2, 1, 3])

        # CONV
        net = tf_util.conv2d(image_pl, 64, [1, 1], padding='VALID', stride=[1, 1],
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
        pc_feat1 = tf.reduce_max(points_feat1, axis=1, keep_dims=True, name="global_features")
        # FC
        pc_feat1 = tf.reshape(pc_feat1, [-1, 1024])
        pc_feat1 = tf_util.fully_connected(pc_feat1, 256, bn=True, is_training=self.is_training_pl, scope='fc1',
                                           bn_decay=self._bn_decay)
        pc_feat1 = tf_util.fully_connected(pc_feat1, 128, bn=True, is_training=self.is_training_pl, scope='fc2',
                                           bn_decay=self._bn_decay)

        # CONCAT
        pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [-1, 1, 1, 128]), [1, num_points, 1, 1])
        points_feat1_concat = tf.concat(axis=3, values=[points_feat1, pc_feat1_expand])

        # CONV
        net = tf_util.conv2d(points_feat1_concat, 512, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training_pl, scope='conv6')
        net = tf_util.conv2d(net, 256, [1, 1], padding='VALID', stride=[1, 1],
                             bn=True, is_training=self.is_training_pl, scope='conv7')
        net = tf_util.dropout(net, keep_prob=0.7, is_training=self.is_training_pl, scope='dp1')
        net = tf_util.conv2d(net, self.batch_generator.dataset.num_classes, [1, 1], padding='VALID', stride=[1, 1],
                             activation_fn=None, scope='conv8')

        net = tf.transpose(net, [0, 2, 1, 3])
        return net
