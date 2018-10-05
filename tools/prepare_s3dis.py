"""
adapted from https://github.com/charlesq34/pointnet
"""

import os
import numpy as np
import glob
import argparse
import tools
from pathlib import Path
from tqdm import tqdm


def collect_point_label(anno_path, out_filename):
    """ Convert original dataset files to data_label file (each line is XYZRGBL).
        We aggregated all the points from each instance in the room.

    Args:
        anno_path: path to annotations. e.g. Area_1/office_2/Annotations/
        out_filename: path to save collected points and labels (each line is XYZRGBL)
    Returns:
        None
    Note:
        the points are shifted before save, the most negative point is now at origin.
    """

    g_classes = [x.rstrip() for x in open('meta/class_names.txt')]
    g_class2label = {cls: i for i, cls in enumerate(g_classes)}

    points_list = []

    for f in glob.glob(os.path.join(anno_path, '*.txt')):
        cls = os.path.basename(f).split('_')[0]
        if cls not in g_classes:  # note: in some room there is 'staris' class..
            cls = 'clutter'
        points = np.loadtxt(f)
        labels = np.ones((points.shape[0], 1)) * g_class2label[cls]
        points_list.append(np.concatenate([points, labels], 1))  # Nx7

    data_label = np.concatenate(points_list, 0)
    xyz_min = np.amin(data_label, axis=0)[0:3]
    data_label[:, 0:3] -= xyz_min
    data_label = data_label.astype(dtype=np.float32)

    output_folder = Path(os.path.dirname(out_filename))
    output_folder.mkdir(parents=True, exist_ok=True)

    np.save(out_filename, data_label)


def main(params):
    anno_paths = [x[0] for x in os.walk(params.input_dir) if x[0].endswith('Annotations')]

    # Note: there is an extra character in the v1.2 data in Area_5/hallway_6. It's fixed manually.
    for anno_path in tqdm(anno_paths):
        elements = anno_path.split('/')
        out_filename = elements[-3] + '_' + elements[-2] + '.npy'  # Area_1_hallway_1.npy
        try:
            collect_point_label(anno_path, os.path.join(params.output_dir, elements[-3], 'full_size', out_filename))
        except Exception as e:
            print(str(e))
            print(out_filename)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert original S3DIS dataset to npy based file format used'
                                                 'by our framework')

    parser.add_argument('--input_dir', required=True, help='root directory of original data')
    parser.add_argument('--output_dir', required=True, help='root directory of output npys')
    params = parser.parse_args()

    tools.pretty_print_arguments(params)

    main(params)
