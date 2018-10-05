from .center_batch_generator import *


class NeighboringGridBatchGenerator(CenterBatchGenerator):
    """
    neighboring grid batch generator where different blobs are placed next to each other
    """

    def __init__(self, dataset, params):
        print(params)
        super().__init__(dataset=dataset,
                         num_points=params['num_points'],
                         batch_size=params['batch_size'],
                         grid_spacing=params['grid_spacing'],
                         augmentation=params['augmentation'],
                         metric=params['metric'])

        self._grid_x = params['grid_x']
        self._grid_y = params['grid_y']

        self._radius = params['radius']

        # flattened version of grid
        self._num_of_blobs = self._grid_x * self._grid_y

    @property
    def num_of_blobs(self):
        return self._num_of_blobs

    @lazy_property
    def pointclouds_pl(self):
        return tf.reshape(self.next_element[0], (self._batch_size, self._num_of_blobs,
                                                 self._num_points, self.dataset.num_features + 3))

    @lazy_property
    def labels_pl(self):
        return tf.reshape(self.next_element[1], (self._batch_size, self._num_of_blobs, self._num_points))

    @lazy_property
    def mask_pl(self):
        return tf.reshape(self.next_element[2], (self._batch_size, self._num_of_blobs))

    @lazy_property
    def cloud_ids_pl(self):
        return tf.reshape(self.next_element[3], (self._batch_size, self._num_of_blobs))

    @lazy_property
    def point_ids_pl(self):
        return tf.reshape(self.next_element[4], (self._batch_size, self._num_of_blobs, self._num_points))

    def _generate_blob(self, index, train=True, aug_trans=True):
        b_data = np.zeros((self._grid_x, self._grid_y, self._num_points, self.dataset.num_features + 3), dtype=np.float32)
        b_label = np.zeros((self._grid_x, self._grid_y, self._num_points), dtype=np.int8)
        b_mask = np.ones((self._grid_x, self._grid_y), dtype=np.int8)
        b_cloud_ids = np.zeros((self._grid_x, self._grid_y), dtype=np.int32)
        b_point_ids = np.zeros((self._grid_x, self._grid_y, self._num_points), dtype=np.int64)

        self._rw_lock.acquire_read()

        try:
            if train:
                [pointcloud_id, grid_center_x, grid_center_y] = np.copy(self.train_sample_idx[index, :])
            else:
                [pointcloud_id, grid_center_x, grid_center_y] = np.copy(self.test_sample_idx[index, :])
        finally:
            self._rw_lock.release_read()

        pointcloud_id = int(pointcloud_id)
        pointcloud_data = self.dataset.data[pointcloud_id]
        max_x, max_y, max_z = np.amax(pointcloud_data[:, 0:3], axis=0)
        min_x, min_y, min_z = np.amin(pointcloud_data[:, 0:3], axis=0)

        diff_x = max_x - min_x
        diff_y = max_y - min_y
        diff_z = max_z - min_z

        noise_x = 0
        noise_y = 0

        num_points = 0

        while num_points < self.min_num_points_per_block:
            if aug_trans:
                noise_x = np.random.uniform(-self._grid_spacing / 2.0, self._grid_spacing / 2.0)
                noise_y = np.random.uniform(-self._grid_spacing / 2.0, self._grid_spacing / 2.0)

            # Create centers
            ctrs = []
            for grid_x in range(self._grid_x):
                ctr_x = grid_center_x + grid_x * self._grid_spacing + noise_x
                for grid_y in range(self._grid_y):
                    ctr_y = grid_center_y + grid_y * self._grid_spacing + noise_y
                    ctr = np.array([ctr_x, ctr_y])
                    ctrs.append(ctr)

            blob_point_ids_all = self.ball_trees[pointcloud_id].query_radius(ctrs, r=self._radius)
            num_points = len(blob_point_ids_all[0])

        curr_id = -1
        for grid_x in range(self._grid_x):
            ctr_x = grid_center_x + grid_x * self._grid_spacing + noise_x
            for grid_y in range(self._grid_y):
                ctr_y = grid_center_y + grid_y * self._grid_spacing + noise_y
                ctr = np.array([ctr_x, ctr_y])

                curr_id += 1

                # New
                blob_point_ids = blob_point_ids_all[curr_id]
                blob = np.copy(pointcloud_data[blob_point_ids, :])

                if blob.shape[0] < self.min_num_points_per_block:  # check here if it is empty, set mask to zero
                    b_mask[grid_x, grid_y] = 0
                else:  # blob is not empty
                    blob, point_ids = self._sample_data(blob, self._num_points)

                    # apply normalizations to blob
                    blob[:, :self.dataset.num_features] /= self.dataset.normalization

                    # Normalized coordinates
                    additional_feats = np.zeros((self._num_points, 3))
                    blob = np.concatenate((blob, additional_feats), axis=-1)

                    blob[:, -1] = blob[:, self.dataset.num_features]  # put label to the end
                    blob[:, self.dataset.num_features] = blob[:, 0] / diff_x
                    blob[:, self.dataset.num_features + 1] = blob[:, 1] / diff_y
                    blob[:, self.dataset.num_features + 2] = blob[:, 2] / diff_z
                    blob[:, 0:2] -= ctr

                    b_label[grid_x, grid_y, :] = blob[:, -1]
                    b_data[grid_x, grid_y, :, :] = blob[:, :-1]
                    b_point_ids[grid_x, grid_y, :] = blob_point_ids[point_ids]

                    b_cloud_ids[grid_x, grid_y] = pointcloud_id

        return b_data, b_label, b_mask, b_cloud_ids, b_point_ids


if __name__ == '__main__':
    pass
