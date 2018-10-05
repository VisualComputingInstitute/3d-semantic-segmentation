"""
Contains methods for evaluating and exporting the result of the network
"""

import os
import numpy as np
from typing import Dict
from tqdm import tqdm
from sklearn.neighbors import BallTree


def knn_interpolation(cumulated_pc: np.ndarray, full_sized_data: np.ndarray, k=5):
    """
    Using k-nn interpolation to find labels of points of the full sized pointcloud
    :param cumulated_pc: cumulated pointcloud results after running the network
    :param full_sized_data: full sized point cloud
    :param k: k for k nearest neighbor interpolation
    :return: pointcloud with predicted labels in last column and ground truth labels in last but one column
    """

    labeled = cumulated_pc[cumulated_pc[:, -1] != -1]
    to_be_predicted = full_sized_data.copy()

    ball_tree = BallTree(labeled[:, :3], metric='euclidean')

    knn_classes = labeled[ball_tree.query(to_be_predicted[:, :3], k=k)[1]][:, :, -1].astype(int)

    interpolated = np.zeros(knn_classes.shape[0])

    for i in range(knn_classes.shape[0]):
        interpolated[i] = np.bincount(knn_classes[i]).argmax()

    output = np.zeros((to_be_predicted.shape[0], to_be_predicted.shape[1]+1))
    output[:, :-1] = to_be_predicted

    output[:, -1] = interpolated

    return output


def calculate_scores(cumulated_result: Dict[str, np.ndarray], num_classes: int):
    """
    calculate evaluation metrics of the prediction
    :param cumulated_result: cumulated_result: last column = predicted label; last but one column = ground truth
    :param num_classes: number of distinct classes of the dataset
    :return: class_acc, class_iou, overall_acc
    """
    total_seen_from_class = [0 for _ in range(num_classes)]
    total_pred_from_class = [0 for _ in range(num_classes)]
    total_correct_from_class = [0 for _ in range(num_classes)]

    for key in cumulated_result.keys():
        for class_id in range(num_classes):
            total_seen_from_class[class_id] += (cumulated_result[key][:, -2] == class_id).sum()
            total_pred_from_class[class_id] += (cumulated_result[key][:, -1] == class_id).sum()

            total_correct_from_class[class_id] += \
                np.logical_and((cumulated_result[key][:, -2] == class_id),
                               (cumulated_result[key][:, -1] == class_id)).sum()

    class_acc = [total_correct_from_class[i] / total_seen_from_class[i] for i in range(num_classes)]

    class_iou = [total_correct_from_class[i] /
                 (total_seen_from_class[i] + total_pred_from_class[i] - total_correct_from_class[i])
                 for i in range(num_classes)]

    overall_acc = sum(total_correct_from_class) / sum(total_seen_from_class)

    return class_acc, class_iou, overall_acc


def save_pc_as_obj(cumulated_result: Dict[str, np.ndarray], label_colors: np.ndarray, save_dir: str):
    """
    save pointclouds as obj files for later inspection with meshlab
    :param cumulated_result: cumulated_result: last column = predicted label; last but one column = ground truth
    :param label_colors: npy array containing the color information for each label class
    :param save_dir: directory to save obj point clouds
    :return: None
    """
    pointclouds_path = save_dir + '/pointclouds'

    for key in tqdm(cumulated_result.keys(), desc='Save obj point clouds to disk'):
        if not os.path.exists(pointclouds_path):
            os.makedirs(pointclouds_path)

        # Save predicted point clouds as obj files for later inspection using meshlab
        fout = open(f"{pointclouds_path}/{key}_pred.obj", 'w')
        pointcloud = cumulated_result[key]
        for j in range(pointcloud.shape[0]):
            color = label_colors[pointcloud[j, -1].astype(int)]
            fout.write(f"v {str(pointcloud[j, 0]).replace('.', ',')}"
                       f" {str(pointcloud[j, 1]).replace('.', ',')}"
                       f" {str(pointcloud[j, 2]).replace('.', ',')}"
                       f" {color[0]} {color[1]} {color[2]}\n")
        fout.close()


def save_npy_results(cumulated_result: Dict[str, np.ndarray], save_dir: str):
    """
    save cumulated results to disk
    :param cumulated_result: last column = predicted label; last but one column = ground truth
    :param save_dir: directory to save npy arrays
    :return: None
    """
    results_npy_path = save_dir + '/results_npy'

    for key in tqdm(cumulated_result.keys(), desc='Save npy results to disk'):
        if not os.path.exists(results_npy_path):
            os.makedirs(results_npy_path)

        np.save(f"{results_npy_path}/{key}", cumulated_result[key])
