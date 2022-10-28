import os.path
import torch
import librosa
import numpy as np
from collections import OrderedDict
import utils
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from collections import Counter


class SpecExtractor:
    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.dim = kwargs['dim']
        self.transform = kwargs['transform']
        self.logger = self.args.logger
        self.pool_type = self.args.pool_type  # mean,max,gwrp
        self.sm = SMOTE(sampling_strategy=0.2, random_state=self.args.seed, k_neighbors=3, n_jobs=32)
        # self.sm = BorderlineSMOTE(sampling_strategy=0.5, random_state=self.args.seed, k_neighbors=3, m_neighbors=6, n_jobs=32)
    def smote_extract(self, s_files, t_files):
        # SMOTE in audio of target domain
        s_xs, t_xs = [], []
        for file in s_files:
            (x, _) = librosa.core.load(file, sr=self.args.sr, mono=True)
            x = x[:self.args.sr * 10]
            s_xs.append(x)
            # x = torch.from_numpy(x)
            # x_mel = self.transform(x)
            # s_xs.append(x_mel.reshape(-1).numpy())
        for file in t_files:
            (x, _) = librosa.core.load(file, sr=self.args.sr, mono=True)
            x = x[:self.args.sr * 10]
            t_xs.append(x)
            # x = torch.from_numpy(x)
            # x_mel = self.transform(x)
            # t_xs.append(x_mel.reshape(-1).numpy())
        s_y = [1 for _ in s_files]
        t_y = [0 for _ in t_files]
        y = s_y + t_y
        xs = s_xs + t_xs
        res_xs, res_y = self.sm.fit_resample(xs, y)
        print(Counter(y), Counter(res_y))
        # ectract feature
        features = []
        for x in res_xs:
            x = torch.from_numpy(np.array(x)).float()
            x_mel = self.transform(x)
            feature = self.get_feature(x_mel, dim=self.dim).reshape(1, -1)
            features.append(feature)
        features = torch.cat(features, dim=0)
        # conv = np.cov(features, rowvar=False)
        # conv_I = np.linalg.inv(conv)
        return features.numpy()

    def extract(self, files):
        if len(files) > 100:
            machine = files[0].split('/')[-3]
            section = files[0].split('/')[-1][:10]
            self.logger.info(f'[{machine}|{section}|sum={len(files)}] Extract {self.pool_type} features in time...')
        features = []
        for file in files:
            (x, _) = librosa.core.load(file, sr=self.args.sr, mono=True)
            x = x[:self.args.sr * 10]
            x = torch.from_numpy(x)
            x_mel = self.transform(x)
            feature = self.get_feature(x_mel, dim=self.dim).reshape(1, -1)
            features.append(feature)
        features = torch.cat(features, dim=0)
        return features.numpy()

    def get_feature(self, x_mel, dim=0):
        """
        dim=0 : extract feature in frequency dimension
        dim=1 : extract feature in time dimension
        """
        if self.pool_type == 'mean':
            feature = x_mel.mean(dim=dim)
        elif self.pool_type == 'max':
            feature, _ = x_mel.max(dim=dim)
        elif self.pool_type == 'gwrp':
            feature = utils.gwrp(x_mel.numpy(), decay=self.args.decay, dim=dim)
            feature = torch.from_numpy(feature)
        else:
            raise ValueError('pool_type set error!"')
        return feature


if __name__ == '__main__':
    a = torch.rand((16, 17))
    b = a[:, 16:].mean(dim=1, keepdim=True)
    print(b.shape)
