import argparse
from datasets import *
from models import *
import yaml
from tools.tools import *
from pathlib import Path
import tools.evaluation as evaluation
import shutil
import logging

avg_iou_per_epoch = [0]
avg_class_acc_per_epoch = [0]
avg_loss_per_epoch = [0]


def main(config: dict, log_dir: str, isTrain: bool):
    with tf.Graph().as_default():
        Dataset = import_class('datasets', config['dataset']['name'])
        dataset = Dataset(config['dataset']['data_path'],
                          is_train=isTrain,
                          test_sets=config['dataset']['test_sets'],
                          downsample_prefix=config['dataset']['downsample_prefix'],
                          is_colors=config['dataset']['colors'],
                          is_laser=config['dataset']['laser'],
                          n_classes=config['dataset']['num_classes'])

        BatchGenerator = import_class('batch_generators', config['batch_generator']['name'])
        batch_generator = BatchGenerator(dataset, config['batch_generator']['params'])

        Model = import_class('models', config['model']['name'])
        model = Model(batch_generator, config['model'].get('params'))

        if isTrain:
            Optimizer = import_class('optimizers', config['optimizer']['name'])
            optimizer = Optimizer(model, config['optimizer']['params'])

            sess, ops, writer, saver, epoch_start = prepare_network(model, log_dir, optimizer, isTrain=isTrain,
                                                                    model_path=config.get('resume_path'))

            for epoch in range(epoch_start, config['train']['epochs']):
                train_one_epoch(sess, ops, writer, model, epoch, config['train']['epochs'])
                eval_one_epoch(sess, ops, model, dataset, epoch, config['train']['epochs'])

                # Save the variables to disk.
                if epoch % 10 == 0:
                    path = Path(f"{log_dir}/model_ckpts")
                    path.mkdir(parents=True, exist_ok=True)
                    saver.save(sess, os.path.join(f"{log_dir}/model_ckpts",
                                                  f"{epoch+1:03d}_model.ckpt"))
        else:
            sess, ops, writer, saver, _ = prepare_network(model, log_dir,
                                                          isTrain=isTrain, model_path=config['model_path'])
            predict_on_test_set(sess, ops, model, dataset, log_dir)


def prepare_network(model: MultiBlockModel, log_dir: str, optimizer=None, isTrain=True, model_path=None):
    # Create a session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    config.log_device_placement = False
    sess = tf.Session(config=config)

    with tf.device('/gpu:0'):
        model.register_summary()
        if optimizer is not None:
            optimizer.register_summary()

        # Add summary writers
        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(log_dir, 'tensorflow'), sess.graph)

        if optimizer is not None:
            ops = {'pointclouds_pl': model.batch_generator.pointclouds_pl,
                   'labels_pl': model.batch_generator.labels_pl,
                   'mask_pl': model.batch_generator.mask_pl,
                   'eval_per_epoch_pl': model.eval_per_epoch_pl,
                   'is_training_pl': model.is_training_pl,
                   'pred': model.prediction,
                   'pred_sm': model.prediction_sm,
                   'loss': model.loss,
                   'train_op': optimizer.optimize,
                   'merged': merged,
                   'step': optimizer.global_step,
                   'correct': model.correct,
                   'labels': model.labels,
                   'handle_pl': model.batch_generator.handle_pl,
                   'iterator_train': model.batch_generator.iterator_train,
                   'iterator_test': model.batch_generator.iterator_test,
                   'cloud_ids_pl': model.batch_generator.cloud_ids_pl,
                   'point_ids_pl': model.batch_generator.point_ids_pl,
                   'next_element': model.batch_generator.next_element
                   }
        else:
            ops = {'pointclouds_pl': model.batch_generator.pointclouds_pl,
                   'labels_pl': model.batch_generator.labels_pl,
                   'mask_pl': model.batch_generator.mask_pl,
                   'eval_per_epoch_pl': model.eval_per_epoch_pl,
                   'is_training_pl': model.is_training_pl,
                   'pred': model.prediction,
                   'pred_sm': model.prediction_sm,
                   'loss': model.loss,
                   'correct': model.correct,
                   'labels': model.labels,
                   'handle_pl': model.batch_generator.handle_pl,
                   'iterator_train': model.batch_generator.iterator_train,
                   'iterator_test': model.batch_generator.iterator_test,
                   'cloud_ids_pl': model.batch_generator.cloud_ids_pl,
                   'point_ids_pl': model.batch_generator.point_ids_pl,
                   'next_element': model.batch_generator.next_element
                   }

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {model.is_training_pl: isTrain})

        # Add ops to save and restore all the variables.
        saver = tf.train.Saver()

        epoch_number = 0

        if model_path is not None:
            # resume training
            latest_checkpoint_path = tf.train.latest_checkpoint(model_path)
            # extract latest training epoch number
            epoch_number = int(latest_checkpoint_path.split('/')[-1].split('_')[0])

            saver.restore(sess, latest_checkpoint_path)

        return sess, ops, train_writer, saver, epoch_number


def train_one_epoch(sess, ops, train_writer, model, epoch, max_epoch):
    model.batch_generator.shuffle()

    for _ in tqdm(range(model.batch_generator.num_train_batches),
                  desc=f"Running training epoch {epoch+1:03d} / {max_epoch:03d}"):

        a = np.reshape(np.array(avg_class_acc_per_epoch[-1]), [1, 1])
        b = np.reshape(np.array(avg_iou_per_epoch[-1]), [1, 1])
        c = np.reshape(np.array(avg_loss_per_epoch[-1]), [1, 1])
        eval_per_epoch = np.concatenate((a, b, c))

        handle_train = sess.run(ops['iterator_train'].string_handle())
        feed_dict = {ops['is_training_pl']: True,
                     ops['eval_per_epoch_pl']: eval_per_epoch,
                     ops['handle_pl']: handle_train}

        start_time = time.time()

        summary, step, _, loss_val, pc_val, pred_val, labels_val, correct_val = sess.run(
            [ops['merged'], ops['step'], ops['train_op'], ops['loss'],
             ops['pointclouds_pl'],
             ops['pred'], ops['labels'], ops['correct']],
             feed_dict=feed_dict)

        elapsed_time = time.time() - start_time
        summary2 = tf.Summary()
        summary2.value.add(tag='secs_per_iter', simple_value=elapsed_time)
        train_writer.add_summary(summary2, step)
        train_writer.add_summary(summary, step)


def eval_one_epoch(sess, ops, model, dataset, epoch, max_epoch):
    total_correct = 0
    total_seen = 0
    loss_sum = 0

    # Compute avg IoU over classes
    total_seen_class = [0 for _ in range(dataset.num_classes)]  # true_pos + false_neg i.e. all points from this class
    total_correct_class = [0 for _ in range(dataset.num_classes)]  # true_pos
    total_pred_class = [0 for _ in range(dataset.num_classes)]  # true_pos + false_pos i.e. num pred classes

    overall_acc = []

    for _ in tqdm(range(model.batch_generator.num_test_batches),
                  desc='Running evaluation epoch %04d / %04d' % (epoch+1, max_epoch)):

        a = np.reshape(np.array(avg_class_acc_per_epoch[-1]), [1, 1])
        b = np.reshape(np.array(avg_iou_per_epoch[-1]), [1, 1])
        c = np.reshape(np.array(avg_loss_per_epoch[-1]), [1, 1])
        eval_per_epoch = np.concatenate((a, b, c))

        handle_test = sess.run(ops['iterator_test'].string_handle())
        feed_dict = {ops['is_training_pl']: False,
                     ops['eval_per_epoch_pl']: eval_per_epoch,
                     ops['handle_pl']: handle_test}

        _, step, loss_val, pred_val, correct_val, labels_val, batch_mask, batch_cloud_ids, batch_point_ids = sess.run(
            [ops['merged'], ops['step'], ops['loss'],
             ops['pred_sm'], ops['correct'], ops['labels'],
             ops['mask_pl'], ops['cloud_ids_pl'], ops['point_ids_pl']], feed_dict=feed_dict)

        total_correct += np.sum(correct_val)  # shape: scalar
        total_seen += pred_val.shape[0] * pred_val.shape[1]

        overall_acc.append(total_correct / total_seen)

        loss_sum += loss_val

        pred_val = np.argmax(pred_val, 2)  # shape: (BS*B' x N)

        for i in range(labels_val.shape[0]):  # iterate over blocks
            for j in range(labels_val.shape[1]):  # iterate over points in block
                lbl_gt = int(labels_val[i, j])
                lbl_pred = int(pred_val[i, j])
                total_seen_class[lbl_gt] += 1
                total_correct_class[lbl_gt] += (lbl_pred == lbl_gt)
                total_pred_class[lbl_pred] += 1

    iou_per_class = np.zeros(dataset.num_classes)
    iou_per_class_mask = np.zeros(dataset.num_classes, dtype=np.int8)
    for i in range(dataset.num_classes):
        denominator = float(total_seen_class[i] + total_pred_class[i] - total_correct_class[i])

        if denominator != 0:
            iou_per_class[i] = total_correct_class[i] / denominator
        else:
            iou_per_class_mask[i] = 1

    iou_per_class_masked = np.ma.array(iou_per_class, mask=iou_per_class_mask)

    total_seen_class_mask = [1 if seen == 0 else 0 for seen in total_seen_class]

    class_acc = np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float)

    class_acc_masked = np.ma.array(class_acc, mask=total_seen_class_mask)

    avg_iou = iou_per_class_masked.mean()
    avg_loss = loss_sum / float(total_seen / model.batch_generator.num_points)
    avg_class_acc = class_acc_masked.mean()
    avg_class_acc_per_epoch.append(avg_class_acc)
    avg_iou_per_epoch.append(avg_iou)
    avg_loss_per_epoch.append(avg_loss)

    logging.info(f"[Epoch {epoch+1:03d}] avg class acc:   {avg_class_acc}")
    logging.info(f"[Epoch {epoch+1:03d}] avg iou:         {avg_iou}")
    logging.info(f"[Epoch {epoch+1:03d}] avg overall acc: {np.mean(overall_acc)}")


def predict_on_test_set(sess, ops, model, dataset: GeneralDataset, log_dir: str):
    is_training = False

    cumulated_result = {}

    for _ in tqdm(range(model.batch_generator.num_test_batches)):
        handle_test = sess.run(ops['iterator_test'].string_handle())
        feed_dict = {ops['is_training_pl']: is_training,
                     ops['eval_per_epoch_pl']: np.zeros((3,1)),
                     ops['handle_pl']: handle_test}

        loss_val, pred_val, correct_val, labels_val, batch_mask, batch_cloud_ids, batch_point_ids = sess.run(
            [ops['loss'], ops['pred_sm'], ops['correct'], ops['labels'],
             ops['mask_pl'], ops['cloud_ids_pl'], ops['point_ids_pl']], feed_dict=feed_dict)

        num_classes = pred_val.shape[2]
        num_batches = pred_val.shape[0]

        batch_mask = np.array(batch_mask, dtype=bool)  # shape: (BS, B) - convert mask to bool
        batch_point_ids = batch_point_ids[batch_mask]  # shape: (B, N)
        batch_cloud_ids = batch_cloud_ids[batch_mask]  # shape: (B)

        for batch_id in range(num_batches):
            pc_id = batch_cloud_ids[batch_id]
            pc_name = dataset.file_names[pc_id]

            for point_in_batch, point_id in enumerate(batch_point_ids[batch_id, :]):
                num_fs_properties = dataset.data[pc_id].shape[1]
                if pc_name not in cumulated_result:
                    # if there is not information about the point cloud so far, initialize it
                    # label -1 means that there is not label given so far
                    # cumulate predictions for the same point
                    cumulated_result[pc_name] = np.zeros((dataset.data[pc_id].shape[0],
                                                          num_fs_properties + num_classes + 1))
                    cumulated_result[pc_name][:, :num_fs_properties] = dataset.data[pc_id]
                    cumulated_result[pc_name][:, -1] = -1

                cumulated_result[pc_name][point_id, num_fs_properties:-1] += pred_val[batch_id, point_in_batch]
                cumulated_result[pc_name][point_id, -1] = np.argmax(cumulated_result[pc_name][point_id,
                                                                    num_fs_properties:-1])

    for key in tqdm(cumulated_result.keys(), desc='knn interpolation for full sized point cloud'):
        cumulated_result[key] = evaluation.knn_interpolation(cumulated_result[key], dataset.full_sized_data[key])

    class_acc, class_iou, overall_acc = evaluation.calculate_scores(cumulated_result, dataset.num_classes)

    logging.info(f"   overall accuracy: {overall_acc}")
    logging.info(f"mean class accuracy: {np.nanmean(class_acc)}")
    logging.info(f"           mean iou: {np.nanmean(class_iou)}")

    for i in range(dataset.num_classes):
        logging.info(f"accuracy for class {i}: {class_acc[i]}")
        logging.info(f"     iou for class {i}: {class_iou[i]}")

    evaluation.save_npy_results(cumulated_result, log_dir)
    evaluation.save_pc_as_obj(cumulated_result, dataset.label_colors(), log_dir)


if __name__ == '__main__':
    log_dir = setup_logger()

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="experiment definition file", metavar="FILE", required=True)
    args = parser.parse_args()

    params = parser.parse_args()

    with open(params.config, 'r') as stream:
        try:
            config = yaml.load(stream)
            # backup config file
            shutil.copy(params.config, log_dir)

            isTrain = False

            if config['modus'] == 'TRAIN_VAL':
                isTrain = True
            elif config['modus'] == 'TEST':
                isTrain = False

            main(config, log_dir, isTrain)
        except yaml.YAMLError as exc:
            logging.error('Configuration file could not be read')
            exit(1)
