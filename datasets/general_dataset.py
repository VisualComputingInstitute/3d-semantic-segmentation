import os
import numpy as np
import datasets.color_constants as cc
from tools.lazy_decorator import *
from typing import Tuple, List, Dict
import logging


class GeneralDataset:
    """
    Class used for reading in datasets for training/testing.
    Parameterized in order to handle different kinds of datasets (e.g. k-fold datasets)
    """

    @property
    def data_path(self) -> str:
        return self._data_path

    @property
    def data(self) -> List[np.ndarray]:
        return self._data

    @property
    def full_sized_data(self) -> Dict[str, np.ndarray]:
        return self._full_sized_data

    @property
    def file_names(self) -> List[str]:
        return self._file_names

    @property
    def train_pc_idx(self) -> List[int]:
        return self._train_pc_idx

    @property
    def test_pc_idx(self) -> List[int]:
        return self._test_pc_idx

    def __init__(self, data_path: str, is_train: bool, test_sets: list,
                 downsample_prefix: str, is_colors: bool, is_laser: bool, n_classes=None):
        self._test_sets = test_sets
        self._downsample_prefix = downsample_prefix
        self._is_colors = is_colors
        self._is_laser = is_laser

        # it is possible that there is no class information given for test sets
        if n_classes is None:
            self._is_class = True
        else:
            self._is_class = False
            self._num_classes = n_classes

        self._data_path = data_path
        self._data, self._file_names, self._full_sized_data = self._load(is_train)

        # log some dataset properties
        logging.debug(f"number of features:         {self.num_features}")
        logging.debug(f"number of classes:          {self.num_classes}")
        logging.debug(f"number of training samples: {len(self.train_pc_idx)}")
        logging.debug(f"number of test samples:     {len(self.test_pc_idx)}")

    @lazy_property
    def num_classes(self) -> int:
        """
        calculate the number of unique class labels if class information is given in npy-file.
        Otherwise, just return the number of classes which have been defined in the constructor
        :return: number of classes for this dataset
        """
        if self._is_class:
            # assuming that labels are in the last column
            # counting unique class labels of all pointclouds
            _num_classes = len(np.unique(np.concatenate([np.unique(pointcloud[:, -1])
                                                 for pointcloud in self.data])))

            if _num_classes > len(self.label_colors()):
                logging.warning(f"There are more classes than label colors for this dataset. "
                                f"If you want to plot your results, this will not work.")

            return _num_classes
        else:
            return self._num_classes

    @lazy_property
    def normalization(self) -> np.ndarray:
        """
        before blob is fed into the neural network some normalization takes place in the batch generator
        normalization factors specific for each dataset have to be provided
        note: this property can be overriden by subclasses if another normalization is needed
        :return: np.ndarray with normalization factors
        """
        _normalizer = np.array([1. for _ in range(self.num_features)])

        if self._is_colors:
            _normalizer[3:6] = 255.     # normalize colors to [0,1]
            if self._is_laser:
                _normalizer[6] = 2048.  # normalize laser [-1, 1]
        elif self._is_laser:
            _normalizer[3] = 2048.      # normalize laser [-1, 1]

        return _normalizer

    @lazy_property
    def num_features(self) -> int:
        return 3 + self._is_colors * 3 + self._is_laser

    @staticmethod
    def label_colors() -> np.ndarray:
        return np.array([cc.colors['brown'].npy,
                         cc.colors['darkgreen'].npy,
                         cc.colors['springgreen'].npy,
                         cc.colors['red1'].npy,
                         cc.colors['darkgray'].npy,
                         cc.colors['gray'].npy,
                         cc.colors['pink'].npy,
                         cc.colors['yellow1'].npy,
                         cc.colors['violet'].npy,
                         cc.colors['hotpink'].npy,
                         cc.colors['blue'].npy,
                         cc.colors['lightblue'].npy,
                         cc.colors['orange'].npy,
                         cc.colors['black'].npy])

    def _load(self, is_train: bool) -> Tuple[List[np.ndarray], List[str], Dict[str, np.ndarray]]:
        """
        Note that we assume a folder hierarchy of DATA_PATH/SET_NO/{full_size, sample_X_Y, ...}/POINTCLOUD.npy
        :param is_train: true iff training mode
        :return: list of pointclouds and list of filenames
        """
        data_training_test = {}
        full_sized_test_data = {}
        names = set()

        train_pc_names = set()
        test_pc_names = set()

        # pick
        pick = [0, 1, 2]

        if self._is_colors:
            pick = pick + [3, 4, 5]

            if self._is_laser:
                pick = pick + [6]

        if self._is_laser:
            pick = pick + [3]

        pick = pick + [-1]

        for dirpath, dirnames, filenames in os.walk(self.data_path):
            for filename in [f for f in filenames if f.endswith(".npy")]:
                is_test_set = os.path.dirname(dirpath).split('/')[-1] in self._test_sets

                if not is_test_set and not is_train:
                    # we do not have to load training examples if we only want to evaluate
                    continue

                name = None
                if os.path.basename(dirpath) == self._downsample_prefix:
                    # dimension of a single npy file: (number of points, number of features + label)
                    pointcloud_data = np.load(os.path.join(dirpath, filename))
                    pointcloud_data = pointcloud_data[:, pick]
                    pointcloud_data = pointcloud_data.astype(np.float32)  # just to be sure!

                    name = filename.replace('.npy', '')
                    data_training_test[name] = pointcloud_data
                elif os.path.basename(dirpath) == 'full_size':
                    if not is_train:
                        # for testing we consider full scale point clouds
                        if is_test_set:
                            # dimension of a single npy file: (number of points, number of features + label)
                            pointcloud_data = np.load(os.path.join(dirpath, filename))
                            pointcloud_data = pointcloud_data[:, pick]
                            pointcloud_data = pointcloud_data.astype(np.float32)  # just to be sure!

                            name = filename.replace('.npy', '')
                            full_sized_test_data[name] = pointcloud_data

                if name is not None:
                    names.add(name)

                    if is_test_set:
                        test_pc_names.add(name)
                    else:
                        train_pc_names.add(name)

        names = sorted(names)

        data_training_test = [data_training_test[key] for key in names]

        self._train_pc_idx = sorted([names.index(name) for name in train_pc_names])
        self._test_pc_idx = sorted([names.index(name) for name in test_pc_names])

        return data_training_test, names, full_sized_test_data


if __name__ == '__main__':
    from tools.tools import setup_logger

    setup_logger()

    dataset = GeneralDataset(data_path='/fastwork/schult/stanford_indoor',
                             is_train=False,
                             test_sets=['area_3', 'area_2'],
                             downsample_prefix='sample_1_1',
                             is_colors=True,
                             is_laser=True)
