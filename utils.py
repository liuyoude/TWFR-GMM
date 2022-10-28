"""
functional functions
"""
import math
import os
import re
import itertools
import glob

import torch
import yaml
import csv
import logging
import torchaudio
import numpy as np
import librosa
import sklearn
import random

system_sep = '/'


def load_yaml(file_path='./config.yaml'):
    with open(file_path) as f:
        params = yaml.safe_load(f)
    return params


def save_yaml_file(file_path, data: dict):
    with open(file_path, "w") as f:
        yaml.safe_dump(data, f, encoding='utf-8', allow_unicode=True)


def save_csv(file_path, data: list):
    with open(file_path, 'w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(data)


def save_model_state_dict(file_path, epoch=None, net=None, optimizer=None):
    import torch
    state_dict = {
        'epoch': epoch,
        'optimizer': optimizer.state_dict() if optimizer else None,
        'model': net.state_dict() if net else None,
    }
    torch.save(state_dict, file_path)


def get_logger(filename):
    logging.basicConfig(filename=filename, level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    logger.addHandler(logging.StreamHandler())
    return logger


def get_filename_list(dir_path, ext='wav', pattern='*'):
    """
    find all extention files under directory
    :param dir_path: directory path
    :param ext: extention name, like wav, png...
    :return: files path list
    """
    filename_list = []
    ext = ext if ext else '*'
    for root, dirs, files in os.walk(dir_path):
        file_path_pattern = os.path.join(root, f'{pattern}.{ext}')
        files = sorted(glob.glob(file_path_pattern))
        filename_list += files
    return sorted(filename_list)


def get_machine_section_list(target_dir, ext='wav'):
    """
    统计一个路径下的section列表
    """
    dir_path = os.path.abspath(f'{target_dir}/*.{ext}')
    files_path = sorted(glob.glob(dir_path))
    machine_section_list = sorted(list(set(
        itertools.chain.from_iterable([re.findall('section_[0-9][0-9]', ext_section) for ext_section in files_path])
    )))
    return machine_section_list


def get_label(filename, att2idx, file_att_2_idx):
    """根据属性字典，和标签字典将文件名转换为标签下标和属性onehot"""
    atts = filename2attributes(filename)
    file_att = '_'.join(atts)
    file_label = file_att_2_idx[file_att]
    one_hot = torch.zeros((len(att2idx.keys())))
    for att in atts: one_hot[att2idx[att]] = 1
    return file_label, one_hot


# getting target dir file list and label list
def get_valid_file_list(target_dir,
                        section_name,
                        prefix_normal='normal',
                        prefix_anomaly='anomaly',
                        ext='wav'):
    normal_files_path = f'{target_dir}/{section_name}_*_{prefix_normal}_*.{ext}'
    normal_files = sorted(glob.glob(normal_files_path))
    normal_labels = np.zeros(len(normal_files))

    anomaly_files_path = f'{target_dir}/{section_name}_*_{prefix_anomaly}_*.{ext}'
    anomaly_files = sorted(glob.glob(anomaly_files_path))
    anomaly_labels = np.ones(len(anomaly_files))

    files = np.concatenate((normal_files, anomaly_files), axis=0)
    labels = np.concatenate((normal_labels, anomaly_labels), axis=0)

    domain_list = []
    for file in files: domain_list.append('source' if ('source' in file) else 'target')
    return files, labels, domain_list


def get_eval_file_list(target_dir, id_name, ext='wav'):
    files_path = f'{target_dir}/{id_name}*.{ext}'
    files = sorted(glob.glob(files_path))
    return files


def filename2attributes(filename):
    """文件名转成属性列表"""
    attribute_list = []
    machine, name = filename.split('/')[0], filename.split('/')[2]
    attribute_list.append(machine)
    f_split_list = os.path.splitext(name)[0].split('_')
    name_list = f_split_list[6:]
    # TODO add section
    # section = '_'.join(f_split_list[:2])
    # domain = f_split_list[2]
    # if domain == 'target':
    #     attribute_list.append(section)
    # TODO add domain
    domain = f_split_list[2]
    attribute_list.append(domain)
    for i in range(len(name_list) // 2):
        attribute_list.append('_'.join(name_list[2 * i: 2 * i + 2]))
    # attribute = ''
    # for idx, meta in enumerate(name_list):
    #     if idx % 2 == 0:
    #         attribute += meta
    #     else:
    #         attribute += '_'
    #         attribute += meta
    #         attribute_list.append(attribute)
    #         attribute = ''
    return attribute_list


def get_attributes(dir_path, machine=None, section=None, domain=None):
    """得到路径下的所有属性，和属性对应的文件数量"""
    attributes = set()
    state = {}
    sep = system_sep
    filename_list = get_filename_list(dir_path, ext='wav')
    for filename in filename_list:
        machine_ = filename.split(sep)[-3]
        section_ = '_'.join(os.path.basename(filename).split('_')[:2])
        domain_ = os.path.basename(filename).split('_')[2]
        if machine and machine_ != machine: continue
        if section and section_ != section: continue
        if domain and domain_ != domain: continue
        filename = filename.split(sep)[-3:]
        filename = '/'.join(filename)
        attribute_list = filename2attributes(filename)
        for attribute in attribute_list:
            attributes.add(attribute)
            if attribute not in state.keys():
                state[attribute] = 1
            else:
                state[attribute] += 1
    return sorted(attributes), state


def get_file_attributes(dir_path, machine=None, section=None, domain=None):
    """得到一个路径下所有的标签（属性组成），和标签对应文件数量"""
    file_attributes = set()
    state = {}
    sep = system_sep
    filename_list = get_filename_list(dir_path, ext='wav')
    for filename in filename_list:
        machine_ = filename.split(sep)[-3]
        section_ = '_'.join(os.path.basename(filename).split('_')[:2])
        domain_ = os.path.basename(filename).split('_')[2]
        if machine and machine_ != machine: continue
        if section and section_ != section: continue
        if domain and domain_ != domain: continue
        filename = filename.split(sep)[-3:]
        filename = '/'.join(filename)
        attribute_list = filename2attributes(filename)
        file_attribute = '_'.join(attribute_list)
        file_attributes.add(file_attribute)
        if file_attribute not in state.keys():
            state[file_attribute] = 1
        else:
            state[file_attribute] += 1
    return sorted(file_attributes), state


def map_attribuate(dir_path, machine=None, section=None, domain=None):
    """属性下标映射"""
    attributes, _ = get_attributes(dir_path, machine, section, domain)
    idx2att, att2idx = {}, {}
    attributes = list(attributes)
    for idx, att in enumerate(attributes):
        idx2att[idx] = att
        att2idx[att] = idx
    return att2idx, idx2att


def map_file_attribute(dir_path, machine=None, section=None, domain=None):
    """标签下标映射"""
    file_attribute_2_idx = {}
    idx_2_file_attribuate = {}
    file_attributes, _ = get_file_attributes(dir_path, machine, section, domain)
    for idx, file_attribute in enumerate(list(file_attributes)):
        file_attribute_2_idx[file_attribute] = idx
        idx_2_file_attribuate[idx] = file_attribute
    return file_attribute_2_idx, idx_2_file_attribuate


def cal_file_att_weights(idx_2_file_att, state):
    weights = []
    samples_per_cls = []
    for idx in range(len(idx_2_file_att.keys())):
        file_att = idx_2_file_att[idx]
        num_files = state[file_att]

        weights.append(1 / num_files)
        samples_per_cls.append(num_files)
    sum_w = np.sum(weights)
    weights = [weight / sum_w for weight in weights]
    return weights, samples_per_cls


def cal_anomaly_score(probs, machine_section_file_atts, file_att_2_idx):
    eps = 1e-8
    # k = min(k, len(machine_section_file_atts))
    releated_probs = np.array(sorted([probs[file_att_2_idx[att]] for att in machine_section_file_atts], reverse=True))
    # anomaly_score = - np.log10(np.mean(releated_probs) + eps) + np.min(- np.log10(releated_probs + eps))
    anomaly_score = - np.log10(np.mean(releated_probs) + eps)
    # anomaly_score = np.min(- np.log10(releated_probs + eps))
    return anomaly_score


def cal_anomaly_score_att(probs, att_probs, machine_section_file_atts, file_att_2_idx, att2idx):
    eps = 1e-8
    # k = min(k, len(machine_section_file_atts))
    releated_probs = np.array([probs[file_att_2_idx[att]] for att in machine_section_file_atts])
    # anomaly_score = - np.log10(np.sum(releated_probs) + eps) + np.min(- np.log10(releated_probs + eps))
    # anomaly_score1 = - np.log10(np.mean(releated_probs) + eps)
    idx = np.argmax(releated_probs)
    file_att = machine_section_file_atts[idx]
    name_list = file_att.split('_')
    att_list = [name_list[0]]
    for i in range(len(name_list[1:]) // 2):
        att_list.append('_'.join(name_list[1 + 2 * i: 2 * i + 3]))
    releated_att_probs = np.array([att_probs[att2idx[att]] for att in att_list])
    anomaly_score = - np.log10(np.mean(releated_att_probs) + eps)
    return anomaly_score

# def cal_anomaly_score_att(probs, att_probs, machine_section_file_atts, file_att_2_idx, att2idx):
#     eps = 1e-8
#     # k = min(k, len(machine_section_file_atts))
#     releated_probs = np.array([probs[file_att_2_idx[att]] for att in machine_section_file_atts])
#     # anomaly_score = - np.log10(np.sum(releated_probs) + eps) + np.min(- np.log10(releated_probs + eps))
#     # anomaly_score = np.min(- np.log(releated_probs + eps))
#     # idx = np.argmax(releated_probs)
#     att_list = []
#     for file_att in machine_section_file_atts:
#         name_list = file_att.split('_')
#         att_list.append(name_list[0])
#         for i in range(len(name_list[1:]) // 2):
#             att_list.append('_'.join(name_list[1 + 2 * i: 2 * i + 3]))
#     releated_att_probs = np.array([att_probs[att2idx[att]] for att in att_list])
#     anomaly_score = (- np.log10(np.mean(releated_att_probs) + eps))
#     return anomaly_score


def cal_auc_pauc(y_true, y_pred, domain_list, max_fpr=0.1):
    y_true_s = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "source" or y_true[idx] == 1]
    y_pred_s = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "source" or y_true[idx] == 1]
    y_true_t = [y_true[idx] for idx in range(len(y_true)) if domain_list[idx] == "target" or y_true[idx] == 1]
    y_pred_t = [y_pred[idx] for idx in range(len(y_true)) if domain_list[idx] == "target" or y_true[idx] == 1]
    auc_s = sklearn.metrics.roc_auc_score(y_true_s, y_pred_s)
    auc_t = sklearn.metrics.roc_auc_score(y_true_t, y_pred_t)
    p_auc = sklearn.metrics.roc_auc_score(y_true, y_pred, max_fpr=max_fpr)
    return auc_s, auc_t, p_auc


def cal_statistic_data(dirs: list, sr=16000):
    import torch
    print('Get mean and std of each machine type for training...')
    wav2mel = Wave2Mel(sr=sr)
    mean, std, sum = 0, 0, 0
    for dir in dirs:
        machine_type = dir.split('/')[-2]
        filenames = get_filename_list(dir)
        for filename in filenames:
            x, _ = librosa.core.load(filename, sr=sr, mono=True)
            x_mel = wav2mel(torch.from_numpy(x))
            mean += torch.mean(x_mel)
            std += torch.std(x_mel)
            sum += 1
    mean /= sum
    std /= sum
    print(f'mean:{mean:.3f}\tstd:{std:.3f}')


def normalize(data, mean=None, std=None):
    if mean and std:
        return (data - mean) / (std * 2)
    else:
        return data


class Wave2Mel(object):
    def __init__(self, sr,
                 n_fft=1024,
                 n_mels=128,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.mel_transform = torchaudio.transforms.MelSpectrogram(sample_rate=sr,
                                                                  win_length=win_length,
                                                                  hop_length=hop_length,
                                                                  n_fft=n_fft,
                                                                  n_mels=n_mels,
                                                                  power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        # spec =  self.amplitude_to_db(self.mel_transform(x)).squeeze().transpose(-1,-2)
        return self.amplitude_to_db(self.mel_transform(x))


class Wave2Spec(object):
    def __init__(self, sr,
                 n_fft=1024,
                 win_length=1024,
                 hop_length=512,
                 power=2.0
                 ):
        self.transform = torchaudio.transforms.Spectrogram(n_fft=n_fft,
                                                           win_length=win_length,
                                                           hop_length=hop_length,
                                                           power=power)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    def __call__(self, x):
        # spec =  self.amplitude_to_db(self.mel_transform(x)).squeeze().transpose(-1,-2)
        return self.amplitude_to_db(self.transform(x))


def gwrp(data, decay, dim=1):
    data = np.sort(data, axis=dim)[:, ::-1]
    gwrp_w = decay ** np.arange(data.shape[dim])
    #gwrp_w[gwrp_w < 0.1] = 0.1
    sum_gwrp_w = np.sum(gwrp_w)
    data = data * gwrp_w
    out = np.sum(data, axis=dim)
    out = out / sum_gwrp_w
    # print(out.shape)
    return out


def data_statistics():
    dir_path = '../../data/dcase2022dataset'
    csv_path = './data_statistics.csv'
    csv_lines = []
    atts, att_state = get_attributes(dir_path)
    csv_lines.append(['Attributes', len(atts), atts])
    csv_lines.append([])

    source_file_attributes, source_state = get_file_attributes(dir_path, domain='source')
    target_file_attributes, target_state = get_file_attributes(dir_path, domain='target')
    csv_lines.append(['Source', len(source_file_attributes), source_file_attributes])
    csv_lines.append(['Target', len(target_file_attributes), target_file_attributes])
    csv_lines.append([])

    for (state, info) in zip([source_state, target_state, att_state], ['Source', 'Target', 'Attribute']):
        for key in sorted(state.keys()):
            csv_lines.append([info, key, state[key]])
        csv_lines.append([])

    machine_list = ['ToyCar', 'ToyTrain', 'bearing', 'fan', 'gearbox', 'slider', 'valve']
    section_list = ['section_00', 'section_01', 'section_02', 'section_03', 'section_04', 'section_05']
    for machine in machine_list:
        for section in section_list:
            file_atts, _ = get_file_attributes(dir_path, machine=machine, section=section)
            csv_lines.append([machine, section, len(file_atts), file_atts])
        csv_lines.append([])
    save_csv(csv_path, csv_lines)


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == '__main__':
    # get_attributes('./dcase2022dataset')
    # print(get_file_attribute('ToyTrain/test/section_00_target_test_normal_0008_car_C1_spd_6_mic_1_noise_2.wav'))
    # atts = get_attributes('./dcase2022dataset')
    # print(len(atts), sorted(atts))
    # att2idx, idx2att = map_attribuate(atts)
    # print(att2idx, idx2att)
    # map_file_attribute('./dcase2022dataset/dev_data')
    # data_statistics()
    # a = [i for i in [1, 2, 3] if i == 1 or i == 2]
    # print(a)
    # a = np.array([[1, 2], [4, 3]])
    # out = gwrp(a, decay=1)
    print(1)
    print(1)
    print(1)
    print(1)
    print(1)
    print(1)
    print(1)
    print(1)
