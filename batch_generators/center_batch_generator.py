from tqdm import *
from sklearn.neighbors import BallTree
from .batch_generator import *
from .ReadWriteLock import ReadWriteLock


class CenterBatchGenerator(BatchGenerator):
    """
    creates batches where the blobs are centered around a specific point in the grid cell
    """

    def __init__(self, dataset, batch_size, num_points, grid_spacing,
                 augmentation=False, metric='euclidean'):
        self._rw_lock = ReadWriteLock()

        super().__init__(dataset, batch_size, num_points, augmentation)

        self._grid_spacing = grid_spacing

        self._shuffle_train = True
        self._shuffle_test = True

        # TODO make min_num_points_per_block variable
        self.min_num_points_per_block = 5

        self._ball_trees = self._calc_ball_trees(metric=metric)
        self._pc_center_pos = self._generate_center_positions()

    @property
    def ball_trees(self):
        return self._ball_trees

    @property
    def pc_center_pos(self):
        return self._pc_center_pos

    @property
    def test_sample_idx(self):
        # after shuffling we need to recalculate the indices
        if self._shuffle_test:
            self._test_sample_idx_cache = self.pc_center_pos[np.isin(self.pc_center_pos[:, 0], self.dataset.test_pc_idx)]
            self._shuffle_test = False

        return self._test_sample_idx_cache

    @property
    def train_sample_idx(self):
        # after shuffling we need to recalculate the indices
        if self._shuffle_train:
            self._train_sample_idx_cache = self.pc_center_pos[np.isin(self.pc_center_pos[:, 0], self.dataset.train_pc_idx)]
            self._shuffle_train = False

        return self._train_sample_idx_cache

    @staticmethod
    def _sample_data(data, num_sample):
        """ data is in N x ...
            we want to keep num_samplexC of them.
            if N > num_sample, we will randomly keep num_sample of them.
            if N < num_sample, we will randomly duplicate samples.
        """
        N = data.shape[0]
        if N == num_sample:
            return data, range(N)
        elif N > num_sample:
            sample = np.random.choice(N, num_sample)
            return data[sample, ...], sample
        else:
            sample = np.random.choice(N, num_sample - N)
            dup_data = data[sample, ...]
            return np.concatenate([data, dup_data], 0), list(range(N)) + (list(sample))

    def _calc_ball_trees(self, metric='euclidean'):
        ball_trees = []
        for pointcloud_data in tqdm(self.dataset.data, desc='Ball trees have to be calculated from scratch'):
            ball_trees.append(BallTree(pointcloud_data[:, :2], metric=metric))
        return ball_trees

    @abstractmethod
    def _generate_blob(self, index, train=True, aug_trans=True):
        raise NotImplementedError('Should be defined in subclass')

    def shuffle(self):
        self._rw_lock.acquire_write()

        try:
            self._shuffle_train = True
            self._shuffle_test = True

            # Randomly shuffle training data from epoch
            np.random.shuffle(self.pc_center_pos)
        finally:
            self._rw_lock.release_write()

    def _generate_center_positions(self):
        """
        Generate blob positions while making sure the grid stays inside the pointcloud bounding box
        :return: (point_cloud_id, pos_x, pos_y)
        """
        room_pos_list = []
        for room_id, room_data in enumerate(tqdm(self.dataset.data, desc='Calculate center positions')):
            room_maxs = np.amax(room_data[:, 0:2], axis=0)
            room_mins = np.amin(room_data[:, 0:2], axis=0)
            room_size = room_maxs - room_mins
            num_blobs = np.ceil(room_size / self._grid_spacing)
            num_blobs = num_blobs - np.array([1, 1]) + 1

            if num_blobs[0] <= 0:
                num_blobs[0] = 1
            if num_blobs[1] <= 0:
                num_blobs[1] = 1

            ctrs = [[room_mins[0] + x * self._grid_spacing + self._grid_spacing / 2.0,
                     room_mins[1] + y * self._grid_spacing + self._grid_spacing / 2.0]
                    for x in range(int(num_blobs[0]))
                    for y in range(int(num_blobs[1]))]

            blob_point_ids_all = self.ball_trees[room_id].query_radius(np.reshape(ctrs, [-1, 2]),
                                                                       r=self._grid_spacing / 2.0)

            blob_point_ids_all = np.reshape(blob_point_ids_all, [-1, 1])

            ctrs = np.reshape(ctrs, [-1, 1, 2])
            for i in range(np.shape(ctrs)[0]):
                npoints = 0
                for j in range(np.shape(ctrs)[1]):
                    npoints = npoints + np.shape(blob_point_ids_all[i, j])[0]
                if npoints >= self.min_num_points_per_block:  # TODO CHECK IF MINPOINTS 5 IS GOOD
                    room_pos_list.append(np.reshape(np.array([room_id, ctrs[i, 0, 0], ctrs[i, 0, 1]]), (1, 3)))

        return np.concatenate(room_pos_list)


if __name__ == '__main__':
    pass
