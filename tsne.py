import matplotlib.pyplot as plt
import numpy as np
import os
import joblib
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches
from sklearn.manifold import TSNE
import argparse

import utils

class MD_Metric:
    def __init__(self, covar):
        self.covar = covar
    def mahano_distance(self, x1, x2):
        distances = []
        for i in range(self.covar.shape[0]):
            covar_I = np.linalg.inv(self.covar[i])
            distance = np.sqrt(np.matmul(np.matmul(x1 - x2, covar_I), (x1 - x2).T))
            distances.append(distance)
        return np.min(distances)



def plot_embedding(data, label, anomaly_flag, label_desc, title, save_path, view='2D', gmm_n=2):
    num_class = len(label_desc)
    # label = np.reshape(label, (-1, 1))
    print(data.shape, label.shape)
    # label -= 4
    # num_class = 8
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    gmm_data = data[-gmm_n:, :]
    data = data[:-gmm_n, :]

    # label_color = []
    # for i in range(label.shape[0]):
    #     if label[i] >= 4:
    #         label_color.append(label[i] - 4)
    #     else:
    #         label_color.append(label[i])
    # label_shape = label
    # print(np.max(label_shape), np.min(label_shape), np.max(label_color), np.min(label_color))

    fig = plt.figure(figsize=(12,4))
    # 3D有待完善
    if view == '3D':
        ax = Axes3D(fig)
        ax.scatter(data[:, 0],
                   data[:, 1],
                   data[:, 2],
                   c=plt.cm.rainbow(label / num_class),
                   s=30,
                   alpha=0.8)
    else:
        for i in range(data.shape[0]):
            plt.scatter(data[i, 0],
                        data[i, 1],
                        # str(label[i]),
                        color=plt.cm.rainbow(label[i] / num_class) if label[i] >= 2 else plt.cm.gray(label[i] / 2),
                        s=20 if (label[i] == 1 or label[i] == 0) else 100,
                        label=label_desc[label[i]],
                        marker='o' if anomaly_flag[i] == 0 else 'x',
                        alpha=0.3 if (label[i] == 1) else 0.9
                        # fontdict={'weight': 'bold', 'size': 9},
                     )
        for i in range(gmm_data.shape[0]):
            plt.scatter(gmm_data[i, 0],
                        gmm_data[i, 1],
                        color='red',
                        s=200,
                        label='GMM_center',
                        marker='^',
                        alpha=0.9
                        # fontdict={'weight': 'bold', 'size': 9},
                     )


    i_list = list(range(num_class))
    patches = [mpatches.Patch(color=plt.cm.rainbow(i / num_class) if i >= 2 else plt.cm.gray(i / 2),
                              label=f'{label_desc[i]}') for i in i_list]
    patches.append(mpatches.Patch(color='red', label='GMM_center'))

    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    # plt.legend(handles=patches, ncol=3, loc='best')
    plt.savefig(save_path, dpi=600)
    plt.show()


def show_tSNE(args, machine=None, section=None):
    tsne_dir = os.path.join('./t-SNE', args.version, f'GMM-{args.gmm_n}')
    tsne_data_path = os.path.join(tsne_dir, 'tsne_data.db')
    tsne_data = joblib.load(tsne_data_path)

    label_desc = ['train_target', 'train_source', 'test_source', 'test_target']
    # no smote
    label = np.concatenate((np.ones(990)*1, np.ones(10)*0, np.ones(50)*2, np.ones(50)*3, np.ones(50)*2, np.ones(50)*3), axis=0).astype(int)
    anomaly_flag = np.concatenate((np.ones(1000)*0, np.ones(100)*0, np.ones(100)*1), axis=0).astype(int)
    # label = np.concatenate(
    #     (np.ones(50) * 2, np.ones(50) * 3, np.ones(50) * 2, np.ones(50) * 3),
    #     axis=0).astype(int)
    # anomaly_flag = np.concatenate((np.ones(100) * 0, np.ones(100) * 1), axis=0).astype(int)

    for index, (target_dir, train_dir) in enumerate(zip(sorted(args.valid_dirs), sorted(args.train_dirs[:7]))):
        machine_type = target_dir.split('/')[-2]
        if machine and machine_type != 'valve': continue
        machine_section_list = utils.get_machine_section_list(target_dir)
        for section_str in machine_section_list:
            if section and '02' not in section_str: continue
            save_path = os.path.join(tsne_dir, f'{machine_type}_{section_str}_tSNE.jpg')
            title = f'{machine_type}_{section_str}'
            gmm = tsne_data[machine_type][section_str]['gmm']
            train_features = tsne_data[machine_type][section_str]['train_features']
            # covar = np.cov(train_features, rowvar=False)[np.newaxis, :]
            test_features = tsne_data[machine_type][section_str]['test_features']
            features = np.concatenate((train_features, test_features, gmm.means_), axis=0)
            # features = test_features
            print('n-GMM', gmm.means_.shape, features.shape, label.shape)
            tsne = TSNE(n_components=2, random_state=0, perplexity=20, metric=MD_Metric(gmm.covariances_).mahano_distance)
            result = tsne.fit_transform(features)
            plot_embedding(result, label, anomaly_flag, label_desc, title, save_path, view='2D', gmm_n=gmm.means_.shape[0])


if __name__ == '__main__':
    # init config parameters
    params = utils.load_yaml(file_path='./config.yaml')
    parser = argparse.ArgumentParser(description=params['description'])
    for key, value in params.items():
        parser.add_argument(f'--{key}', default=value, type=type(value))
    args = parser.parse_args()
    machine = 'valve'
    section = '02'
    versions = ['mean-gmm', 'max-gmm', 'twfr-gmm']
    for version in versions:
        args.version = version
        args.gmm_n = 2
        show_tSNE(args, machine, section)






