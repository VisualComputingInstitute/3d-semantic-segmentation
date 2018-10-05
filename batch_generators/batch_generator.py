from abc import *
import numpy as np
import tensorflow as tf
import itertools
from tools.lazy_decorator import *


class BatchGenerator(ABC):
    """
    Abstract base class for batch generators providing the code for parallel creation of batches
    """

    def __init__(self, dataset, batch_size, num_points, augmentation):
        """
        :param dataset: dataset object
        :type dataset: Dataset
        :param num_points: number of points in a batch
        :type num_points: int
        """
        self.dataset = dataset
        self._num_points = num_points
        self._batch_size = batch_size
        self._augmentation = augmentation

    @lazy_property
    def handle_pl(self):
        # Handle for datasets
        return tf.placeholder(tf.string, shape=[], name='handle_training_test')

    @lazy_property
    def next_element(self):
        iterator = tf.data.Iterator.from_string_handle(self.handle_pl, self.dataset_train.output_types)
        return iterator.get_next()

    @lazy_property
    def dataset_train(self):
        # Create dataset for training
        dataset_train = tf.data.Dataset.from_generator(self._next_train_index, tf.int64, tf.TensorShape([]))
        dataset_train = dataset_train.map(self._wrapped_generate_train_blob, num_parallel_calls=8)
        dataset_train = dataset_train.batch(self._batch_size)
        return dataset_train.prefetch(buffer_size=self._batch_size * 1)

    @lazy_property
    def iterator_train(self):
        return self.dataset_train.make_one_shot_iterator()

    @lazy_property
    def iterator_test(self):
        # Create dataset for testing
        dataset_test = tf.data.Dataset.from_generator(self._next_test_index, tf.int64, tf.TensorShape([]))
        dataset_test = dataset_test.map(self._wrapped_generate_test_blob, num_parallel_calls=8)
        dataset_test = dataset_test.batch(self._batch_size)
        dataset_test = dataset_test.prefetch(buffer_size=self._batch_size * 1)
        return dataset_test.make_one_shot_iterator()

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def num_points(self):
        return self._num_points
    
    @property
    def input_shape(self):
        return self.pointclouds_pl.get_shape().as_list()

    @lazy_property
    @abstractmethod
    def pointclouds_pl(self):
        raise NotImplementedError('Should be defined in subclass')

    @lazy_property
    @abstractmethod
    def labels_pl(self):
        raise NotImplementedError('Should be defined in subclass')

    @lazy_property
    @abstractmethod
    def mask_pl(self):
        raise NotImplementedError('Should be defined in subclass')

    @lazy_property
    @abstractmethod
    def cloud_ids_pl(self):
        raise NotImplementedError('Should be defined in subclass')

    @lazy_property
    @abstractmethod
    def point_ids_pl(self):
        raise NotImplementedError('Should be defined in subclass')

    def _next_train_index(self):
        """
        get next index in training sample list (e.g. containing [pointcloud_id, center_x, center_y])
        Take care that list is shuffled for each epoch!
        :return: next index for training
        """
        for i in itertools.cycle(range(self.train_sample_idx.shape[0])):
            yield (i)

    def _next_test_index(self):
        """
        get next index in test sample list (e.g. containing [pointcloud_id, center_x, center_y])
        Take care that list is shuffled for each epoch!
        :return: next index for test
        """
        for i in itertools.cycle(range(self.test_sample_idx.shape[0])):
            yield (i)

    def _wrapped_generate_train_blob(self, index):
        return tf.py_func(func=self._generate_blob,
                          #     pos_id,  train, aug_trans
                          inp=[index, True, self._augmentation],
                          #       data        labels   mask
                          Tout=(tf.float32, tf.int8, tf.int8, tf.int32, tf.int64),

                          name='generate_train_blob')

    def _wrapped_generate_test_blob(self, index):
        return tf.py_func(func=self._generate_blob,
                          #     pos_id,  train, aug_trans
                          inp=(index, False, False),
                          #       data        labels   mask     cloud_id  point_id
                          Tout=(tf.float32, tf.int8, tf.int8, tf.int32, tf.int64),
                          name='generate_test_blob')

    @property
    def num_train_batches(self):
        return self.train_sample_idx.shape[0] // self._batch_size

    @property
    def num_test_batches(self):
        return int(np.ceil(self.test_sample_idx.shape[0] // self._batch_size))

    @property
    @abstractmethod
    def test_sample_idx(self):
        """
        :rtype: ndarray
        :return:
        """
        raise NotImplementedError('Should be defined in subclass')

    @property
    @abstractmethod
    def train_sample_idx(self):
        """
        :rtype: ndarray
        :return:
        """
        raise NotImplementedError('Should be defined in subclass')

    @abstractmethod
    def _generate_blob(self, index, train=True, aug_trans=True):
        raise NotImplementedError('Should be defined in subclass')
