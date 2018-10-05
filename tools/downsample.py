"""
Downsample full sized point clouds in order to speed up batch generation as well as better block representations.
Resulting point clouds will be saved at the appropriate positions in the file system
(For further information consult the wiki)
"""

import numpy as np
import argparse
import tools
from termcolor import colored
import os
from tqdm import tqdm


def blockwise_uniform_downsample(data_labels, cell_size):
    data_dim = data_labels.shape[1] - 1

    number_classes = int(data_labels[:, -1].max()) + 1  # counting starts with label 0

    d = {}
    for i in tqdm(range(data_labels.shape[0]), desc='downsampling of points'):
        # create block boundaries
        x = int(data_labels[i, 0] / cell_size)
        y = int(data_labels[i, 1] / cell_size)
        z = int(data_labels[i, 2] / cell_size)

        # add space for one hot encoding (used for easier class occurence counting)
        # and counter value at the end of the row
        new_tuple = np.zeros([data_labels.shape[1] + number_classes], dtype=float)
        new_tuple[0:data_dim] = data_labels[i, 0:-1]
        new_tuple[data_dim+int(data_labels[i, -1])] = 1  # set label to one for the corresponding class
        new_tuple[-1] = 1  # count elements in the block

        # note: elementwise addition in numpy arrays!!!
        try:
            d[(x, y, z)] += new_tuple
        except:
            d[(x, y, z)] = new_tuple

    data = []
    labels = []

    for _, v in d.items():
        N = v[-1]  # number of points in voxel

        # aggregate points in block to one normalized point
        data.append([v[i] / N for i in range(data_dim)])

        # find most prominent label (excluding counter and not in one hot encoding anymore)
        labels.append(np.argmax(v[data_dim:-1]))

    data = np.stack(data, axis=0)
    labels = np.stack(labels, axis=0)

    data_labels_new = np.hstack([data, np.expand_dims(labels, 1)]).astype(np.float32)
    return data_labels_new


def main(params):
    for dirpath, dirnames, filenames in os.walk(params.data_dir):
        if os.path.basename(dirpath) == 'full_size':
            for filename in [f for f in filenames if f.endswith(".npy")]:
                print(f"downsampling {filename} in progress ...")
                data_labels = np.load(os.path.join(dirpath, filename))
                sampled_data_labels = blockwise_uniform_downsample(data_labels, params.cell_size)

                out_folder = os.path.join(os.path.dirname(dirpath), f"sample_{params.cell_size}")

                if not os.path.exists(out_folder):
                    os.makedirs(out_folder)

                np.save(os.path.join(out_folder, filename), sampled_data_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert original data set to uniformly downsampled numpy version '
                                                 'in order to speed up batch generation '
                                                 'as well as better block representations')

    parser.add_argument('--data_dir', required=True, help='root directory of original data')
    parser.add_argument('--cell_size', type=float, default=0.03, help='width/length of downsampling cell')
    params = parser.parse_args()

    tools.pretty_print_arguments(params)

    main(params)

    print(colored('Finished successfully', 'green'))
