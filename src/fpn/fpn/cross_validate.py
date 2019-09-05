import tensorflow as tf
import os
import numpy as np

TF_EVENT_FILE_DIR = 'runs'


if __name__ == '__main__':

    event_file_paths = []
    for path, _, files in os.walk(TF_EVENT_FILE_DIR):
        for name in files:
            event_file_paths.append(os.path.join(path, name))

    event_file_paths = [p for p in event_file_paths if 'mot_2017' in p]
    # event_file_paths = [p for p in event_file_paths if 'mot19_cvpr' in p]

    splits_dict = {}
    for p in event_file_paths:
        split_path = p.split('/')
        key = os.path.join(split_path[1], split_path[3])

        if key in splits_dict:
            splits_dict[key].append(p)
        else:
            splits_dict[key] = [p]

    for key, sequence_paths in splits_dict.items():
        seq_ap_means = {}
        seq_epochs = {}

        for p in sequence_paths:
            base_path = os.path.dirname(p)

            mean_ap_val = []
            epochs = []
            for summary in tf.train.summary_iterator(p):
                value = summary.summary.value
                if len(value) > 0:
                    if 'mean_ap/val' in value[0].tag:
                        mean_ap_val.append(value[0].simple_value)
                        epochs.append(summary.step)

            if len(mean_ap_val) > 0:
                if base_path in seq_ap_means:
                    seq_ap_means[base_path] += mean_ap_val
                    seq_epochs[base_path] += epochs
                else:
                    seq_ap_means[base_path] = mean_ap_val
                    seq_epochs[base_path] = epochs

        seq_ap_means = np.array([v for k, v in seq_ap_means.items()])
        if seq_ap_means.ndim == 2:
            seq_mean = seq_ap_means.mean(axis=0)
            index = np.argmax(seq_mean)
            print(
                f'{key}: MEAN AP: {seq_mean[index]:.4f} EPOCH: {seq_epochs[base_path][index]}')
