from tools import tf_util
from .multi_block_model import *
from batch_generators import *
from typing import Dict


class MultiScaleCuModel(MultiBlockModel):
    """
    parameterized version of a multiscale pointnet with consolidation units
    """

    def __init__(self, batch_generator: BatchGenerator, params: Dict[str, list]):
        """
        initialization of multi scale model with consolidation units
        :param batch_generator:
        :param params: contains parameter concerning
            - ilc_sizes (filter sizes for input level context)
            - olc_sizes (consolidation units' sizes)
            - olc_sizes (filter sizes for output level context)
        """
        super().__init__(batch_generator)

        if params is None:
            # standard parameters from the paper
            self._ilc_sizes = [64, 128]
            self._cu_sizes = [256, 1024]
            self._olc_sizes = [512, 128]
        else:
            # load custom parameters
            self._ilc_sizes = params['ilc_sizes']
            self._cu_sizes = params['cu_sizes']
            self._olc_sizes = params['olc_sizes']

        self._bn_decay = 0.9

    @lazy_property
    def _prediction_helper(self):
        # pointcloud placeholder has the following format: BxSxNxF
        # Batch   B
        # Scale   S
        # Point   P
        # Feature F

        # allow an arbitrary number of scales
        scales = [tf.expand_dims(self.batch_generator.pointclouds_pl[:, i, ...], axis=2)
                  for i in range(self.input_shape[1])]

        num_points = self.batch_generator.num_points

        # store reference to original scale for later concatenating
        scale1 = scales[1]

        ''' INPUT-LEVEL CONTEXT '''
        for scale_index in range(len(scales)):
            # build global feature extractor for each scale independently
            for size_index, ilc_size in enumerate(self._ilc_sizes):
                scales[scale_index] = tf_util.conv2d(scales[scale_index], ilc_size, [1, 1], padding='VALID',
                                                     stride=[1, 1],
                                                     bn=True, is_training=self.is_training_pl,
                                                     scope='ilc_conv' + str(size_index) + 'sc' + str(scale_index),
                                                     bn_decay=self._bn_decay)
            # calculate global features for each scale
            scales[scale_index] = tf.reduce_max(scales[scale_index], axis=1,
                                                keep_dims=True, name="gf_sc" + str(scale_index))

        ''' CONCATENATE GLOBAL FEATURES OF ALL SCALES '''
        net = tf.concat(values=scales, axis=3)
        net = tf.tile(net, [1, num_points, 1, 1], name='repeat')
        net = tf.concat(values=[scale1, net], axis=3)

        ''' CONSOLIDATION UNIT SECTION '''
        for index, cu_size in enumerate(self._cu_sizes):
            net = tf_util.consolidation_unit(net, size=cu_size, scope='cu' + str(index), bn=True,
                                             bn_decay=self._bn_decay, is_training=self.is_training_pl)

        ''' OUTPUT-LEVEL CONTEXT '''
        for size_index, olc_size in enumerate(self._olc_sizes):
            net = tf_util.conv2d(net, olc_size, [1, 1], padding='VALID', stride=[1, 1], bn=True,
                                 is_training=self.is_training_pl,
                                 scope='olc_conv' + str(size_index), bn_decay=self._bn_decay)

        net = tf_util.conv2d(net, self.batch_generator.dataset.num_classes, [1, 1], padding='VALID', stride=[1, 1],
                             activation_fn=None, scope='conv_output')

        net = tf.transpose(net, [0, 2, 1, 3])

        return net
