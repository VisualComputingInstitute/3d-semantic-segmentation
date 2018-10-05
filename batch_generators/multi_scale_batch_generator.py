from .center_batch_generator import *
import numpy as np


class MultiScaleBatchGenerator(CenterBatchGenerator):
    """
    Batch Generator for multi scale batches of different radii where the centers are equal
    """

    def __init__(self, dataset, params):
        super().__init__(dataset=dataset,
                         num_points=params['num_points'],
                         batch_size=params['batch_size'],
                         grid_spacing=params['grid_spacing'],
                         augmentation=params['augmentation'],
                         metric=params['metric'])

        self._radii = params['radii']

    @lazy_property
    def pointclouds_pl(self):
        return tf.reshape(self.next_element[0], (self._batch_size, len(self._radii),
                                                 self._num_points, self.dataset.num_features + 3))

    @lazy_property
    def labels_pl(self):
        return tf.reshape(self.next_element[1], (self._batch_size, 1, self._num_points))

    @lazy_property
    def mask_pl(self):
        return tf.reshape(self.next_element[2], (self._batch_size, 1))

    @lazy_property
    def cloud_ids_pl(self):
        return tf.reshape(self.next_element[3], (self._batch_size, 1))

    @lazy_property
    def point_ids_pl(self):
        return tf.reshape(self.next_element[4], (self._batch_size, 1, self._num_points))

    def _generate_blob(self, index, train=True, aug_trans=True):
        b_data = np.zeros((len(self._radii), self._num_points, self.dataset.num_features + 3), dtype=np.float32)
        b_label = np.zeros((len(self._radii), self._num_points), dtype=np.int8)
        b_mask = np.ones(len(self._radii), dtype=np.int8)
        b_cloud_ids = np.zeros(len(self._radii), dtype=np.int32)
        b_point_ids = np.zeros((len(self._radii), self._num_points), dtype=np.int64)

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

        if aug_trans:
            noise_x = np.random.uniform(-self._grid_spacing / 2.0, self._grid_spacing / 2.0)
            noise_y = np.random.uniform(-self._grid_spacing / 2.0, self._grid_spacing / 2.0)

        ctr_x = grid_center_x + noise_x
        ctr_y = grid_center_y + noise_y
        ctr = np.array([ctr_x, ctr_y])

        ctrs = [ctr for _ in range(len(self._radii))]

        blob_point_ids_all = self.ball_trees[pointcloud_id].query_radius(ctrs, r=self._radii)

        for radius_id, radius in enumerate(self._radii):
            blob_point_ids = np.array(blob_point_ids_all[radius_id])
            blob = pointcloud_data[blob_point_ids, :]

            if blob.shape[0] < self.min_num_points_per_block:  # check here if it is empty, set mask to zero
                b_mask[radius_id] = 0
            else:  # blob is not empty
                blob, point_ids = self._sample_data(blob, self._num_points)

                # apply normalizations to blob
                blob[:, :self.dataset.num_features] /= self.dataset.normalization

                # Normalized coordinates
                additional_feats = np.zeros((self._num_points, 3))
                blob = np.concatenate((blob, additional_feats), axis=-1)

                blob[:, -1] = blob[:, self.dataset.num_features]  # put label to the end
                blob[:, self.dataset.num_features] = blob[:, 0] / diff_x
                blob[:, self.dataset.num_features+1] = blob[:, 1] / diff_y
                blob[:, self.dataset.num_features+2] = blob[:, 2] / diff_z
                blob[:, 0:2] -= ctr

                b_label[radius_id, :] = blob[:, -1]
                b_data[radius_id, :, :] = blob[:, :-1]
                b_point_ids[radius_id, :] = blob_point_ids[point_ids]
                b_cloud_ids[radius_id] = pointcloud_id

        # reference radius
        b_label = np.reshape(b_label[1, :], (1, b_label.shape[1]))
        b_mask = np.reshape(b_mask[1], (1))
        b_point_ids = np.reshape(b_point_ids[1, :],
                                     (1, b_point_ids.shape[1]))
        b_cloud_ids = np.reshape(b_cloud_ids[1], (1))

        return b_data, b_label, b_mask, b_cloud_ids, b_point_ids


if __name__ == '__main__':
    pass
