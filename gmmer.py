import logging
import os
import sys
import sklearn
import numpy as np
import time
import re
import joblib
import torch.nn.functional as F

import torch
import librosa
import scipy
from sklearn.manifold import TSNE
from sklearn.mixture import GaussianMixture
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
# import spafe.fbanks.gammatone_fbanks as gf
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
# from torch.utils.tensorboard import SummaryWriter
from visdom import Visdom
from tqdm import tqdm
import utils


# torch.manual_seed(666)


class GMMer(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.version = self.args.version
        self.feature_extractor = kwargs['extractor']
        self.wav2mel = utils.Wave2Mel(sr=self.args.sr)
        self.logger = self.args.logger
        self.csv_lines = []

    def fit_GMM(self, data, n_components, means_init=None, precisions_init=None):
        gmm = GaussianMixture(n_components=n_components, covariance_type='full',
                              means_init=means_init, precisions_init=precisions_init,
                              tol=1e-6, reg_covar=1e-6, verbose=2)
        gmm.fit(data)
        return gmm

    def test(self, train_dirs, valid_dirs, save=True, gmm_n=2, use_search=False, use_smote=False, visual=False):
        """

        running on development dataset (the label(normal, anomaly) can be seen in test data)

        save: flag of saving results or not
        gmm_n: components of gmm
        use_search: using search params on config or not
        use_smote: using smote on target domain of train data or not
        visual: saving features for ploing t-SNE or not
        """
        csv_lines = []
        sum_auc_s, sum_auc_t, sum_pauc, num, total_time = 0, 0, 0, 0, 0
        h_sum_auc_s, h_sum_auc_t, h_sum_pauc = 0, 0, 0
        result_dir = os.path.join('./results', self.version, f'GMM-{gmm_n}') if not use_search else \
            os.path.join('./results', self.version, f'GMM-Mix')
        tsne_dir = os.path.join('./t-SNE', self.version, f'GMM-{gmm_n}') if not use_search else \
            os.path.join('./t-SNE', self.version, f'GMM-Mix')
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(tsne_dir, exist_ok=True)
        print('\n' + '=' * 20)
        t_sne_data = {} # for plot
        for index, (target_dir, train_dir) in enumerate(zip(sorted(valid_dirs), sorted(train_dirs))):
            time.sleep(1)
            start = time.perf_counter()
            machine_type = target_dir.split('/')[-2]
            t_sne_data[machine_type] = {} # for plot
            if self.args.pool_type == 'gwrp':
                decay = self.args.gwrp_decays[self.version][machine_type]
                self.feature_extractor.args.decay = decay
            gmm_n = gmm_n if not use_search else self.args.gmm_ns[self.version][machine_type]
            if use_search and self.version == 'smote-twfr-gmm':
                use_smote = self.args.smotes[machine_type]
            # result csv
            machine_section_list = utils.get_machine_section_list(target_dir)
            csv_lines.append([machine_type])
            csv_lines.append(['section', 'AUC(Source)', 'AUC(Target)', 'pAUC'])
            performance = []
            gmms, train_scores = [], []
            for section_str in machine_section_list:
                # train GMM
                self.logger.info(f'[{machine_type}|{section_str}] Fit GMM-{gmm_n}...')
                if self.args.pool_type == 'gwrp':
                    self.logger.info(f'Gwrp decay: {decay:.2f}')
                # train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_*')
                s_train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_source_*')
                t_train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_target_*')
                train_files = s_train_files + t_train_files
                self.logger.info(f'number of {section_str} files: {len(train_files)}')
                features = self.feature_extractor.extract(train_files) if not use_smote else \
                    self.feature_extractor.smote_extract(s_train_files, t_train_files)
                gmm = self.fit_GMM(features, n_components=gmm_n)
                gmms.append(gmm)
                # for plot
                if visual:
                    t_sne_data[machine_type][section_str] = {}
                    t_sne_data[machine_type][section_str]['train_features'] = features
                    t_sne_data[machine_type][section_str]['gmm'] = gmm
            # calculate threshold by using train data
            #     for file in train_files:
            #         feature = self.feature_extractor.extract([file])
            #         score = - np.max(gmm._estimate_log_prob(feature))
            #         train_scores.append(score)
            # max_score = np.max(train_scores)
            # min_score = np.min(train_scores)
            # train_scores = (train_scores - min_score) / (max_score - min_score)
            # shape_hat, loc_hat, scale_hat = scipy.stats.gamma.fit(train_scores)
            # decision_threshold = scipy.stats.gamma.ppf(q=self.args.decision_threshold, a=shape_hat, loc=loc_hat,
            #                                            scale=scale_hat)
            for idx, section_str in enumerate(machine_section_list):
                gmm = gmms[idx]
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{section_str}.csv')
                decision_path = os.path.join(result_dir, f'decision_result_{machine_type}_{section_str}.csv')
                test_files, y_true, domain_list = utils.get_valid_file_list(target_dir, section_str)
                y_pred = [0. for _ in test_files]
                anomaly_score_list = []
                decision_result_list = []
                test_features = []
                # decision_result_list.append(['decision threshold', decision_threshold, 'max', max_score, 'min', min_score])
                for file_idx, test_file in enumerate(test_files):
                    test_feature = self.feature_extractor.extract([test_file])
                    test_features.append(test_feature)
                    y_pred[file_idx] = - np.max(gmm._estimate_log_prob(test_feature))
                    # y_pred[file_idx] = (- np.max(gmm._estimate_log_prob(test_feature)) - min_score) / (max_score - min_score)
                    anomaly_score_list.append([os.path.basename(test_file), y_pred[file_idx]])
                    # if y_pred[file_idx] > decision_threshold:
                    #     decision_result_list.append([os.path.basename(test_file), y_pred[file_idx], 1])
                    # else:
                    #     decision_result_list.append([os.path.basename(test_file), y_pred[file_idx], 0])
                if save:
                    print(result_dir, csv_path)
                    utils.save_csv(csv_path, anomaly_score_list)
                    # utils.save_csv(decision_path, decision_result_list)
                # for plot
                if visual:
                    test_features = np.concatenate(test_features, axis=0)
                    t_sne_data[machine_type][section_str]['test_features'] = test_features
                # compute auc and pAuc
                auc_s, auc_t, p_auc = utils.cal_auc_pauc(y_true, y_pred, domain_list)
                performance.append([auc_s, auc_t, p_auc])
                csv_lines.append([section_str.split('_', 1)[1], auc_s, auc_t, p_auc])

            # calculate averages for AUCs and pAUCs
            amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
            hmean_performance = scipy.stats.hmean(
                np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon), axis=0)
            mean_auc_s, mean_auc_t, mean_p_auc = amean_performance[0], amean_performance[1], amean_performance[2]
            h_mean_auc_s, h_mean_auc_t, h_mean_p_auc = hmean_performance[0], hmean_performance[1], hmean_performance[2]
            sum_auc_s += mean_auc_s
            sum_auc_t += mean_auc_t
            sum_pauc += mean_p_auc
            h_sum_auc_s += h_mean_auc_s
            h_sum_auc_t += h_mean_auc_t
            h_sum_pauc += h_mean_p_auc
            num += 1
            time_nedded = time.perf_counter() - start
            total_time += time_nedded
            csv_lines.append(["arithmetic mean"] + list(amean_performance))
            csv_lines.append(["harmonic mean"] + list(hmean_performance))
            csv_lines.append([])
            self.logger.info(f'Test {machine_type}\tcost {time_nedded:.2f} sec\tavg_auc_s: {mean_auc_s:.3f}\t'
                             f'avg_auc_t: {mean_auc_t:.3f}\tavg_pauc: {mean_p_auc:.3f}')
        print(f'Total test time: {total_time:.2f} sec')
        result_path = os.path.join(result_dir, 'result.csv')
        avg_auc_s, avg_auc_t, avg_pauc = sum_auc_s / num, sum_auc_t / num, sum_pauc / num
        h_avg_auc_s, h_avg_auc_t, h_avg_pauc = h_sum_auc_s / num, h_sum_auc_t / num, h_sum_pauc / num
        csv_lines.append(['(A)Total Average', f'{avg_auc_s:.4f}', f'{avg_auc_t:.4f}', f'{avg_pauc:.4f}'])
        csv_lines.append(['(H)Total Average', f'{h_avg_auc_s:.4f}', f'{h_avg_auc_t:.4f}', f'{h_avg_pauc:.4f}'])
        if save: utils.save_csv(result_path, csv_lines)
        self.logger.info(f'avg_auc_s: {avg_auc_s:.3f}\tavg_auc_t: {avg_auc_t:.3f}\tavg_pauc: {avg_pauc:.3f}')

        # for plot
        if visual:
            tsne_data_path = os.path.join(tsne_dir, 'tsne_data.db')
            joblib.dump(t_sne_data, tsne_data_path)

    def eval(self, train_dirs, test_dirs, save=True, gmm_n=2, use_search=False, use_smote=False):
        """

        running on evaluation dataset (the label(normal, anomaly) can't be seen in test data)

        save: flag of saving results or not
        gmm_n: components of gmm
        use_search: using search params on config or not
        use_smote: using smote on target domain of train data or not
        """
        team_name = f'{self.args.version}-{gmm_n}' if not use_search else f'{self.args.version}-Mix'
        result_dir = os.path.join('./evaluator/teams', team_name)
        train_result_dir = os.path.join('./evaluator/trains', team_name)
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(train_result_dir, exist_ok=True)
        print('\n' + '=' * 20)
        for index, (target_dir, train_dir) in enumerate(zip(sorted(test_dirs), sorted(train_dirs))):
            machine_type = target_dir.split('/')[-2]
            if self.args.pool_type == 'gwrp':
                decay = self.args.gwrp_decays[machine_type]
                self.feature_extractor.args.decay = decay
            gmm_n = gmm_n if not use_search else self.args.gmm_ns[machine_type]
            if use_search and self.version == 'smote-twfr-gmm':
                use_smote = self.args.smotes[machine_type]
            # get machine list
            machine_section_list = utils.get_machine_section_list(target_dir)
            for section_str in machine_section_list:
                # train GMM
                self.logger.info(f'[{machine_type}|{section_str}] Fit GMM-{gmm_n}...')
                if self.args.pool_type == 'gwrp':
                    self.logger.info(f'Gwrp decay: {decay:.2f}')
                s_train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_source_*')
                t_train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_target_*')
                train_files = s_train_files + t_train_files
                self.logger.info(f'number of {section_str} files: {len(train_files)}')
                if not use_smote:
                    features = self.feature_extractor.extract(train_files)
                else:
                    features = self.feature_extractor.smote_extract(s_train_files, t_train_files)
                if self.args.norm:
                    features = F.normalize(torch.from_numpy(features), dim=1).numpy()
                gmm = self.fit_GMM(features, n_components=gmm_n)
                #
                test_files = utils.get_eval_file_list(target_dir, section_str)
                csv_path = os.path.join(result_dir, f'anomaly_score_{machine_type}_{section_str}.csv')
                train_csv_path = os.path.join(train_result_dir, f'anomaly_score_{machine_type}_{section_str}.csv')
                print('create test anomaly score files...')
                anomaly_score_list = []
                y_pred = [0. for _ in test_files]
                for file_idx, file_path in enumerate(test_files):
                    test_feature = self.feature_extractor.extract([file_path])
                    if self.args.norm:
                        test_feature = F.normalize(torch.from_numpy(test_feature), dim=1).numpy()
                    y_pred[file_idx] = - np.max(gmm._estimate_log_prob(test_feature))
                    anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                    if save: utils.save_csv(csv_path, anomaly_score_list)
                # print('record train files score for computing threshold...')
                # anomaly_score_list = []
                # y_pred = [0. for _ in train_files]
                # for file_idx, file_path in enumerate(train_files):
                #     test_feature = self.feature_extractor.extract([file_path])
                #     y_pred[file_idx] = - np.max(gmm._estimate_log_prob(test_feature))
                #     anomaly_score_list.append([os.path.basename(file_path), y_pred[file_idx]])
                #     utils.save_csv(train_csv_path, anomaly_score_list)

    def search(self, train_dirs, valid_dirs, start=0, end=101, step=1, gmm_ns=None, use_smote=False):
        """

        search decay of gwrp on development dataset (the label(normal, anomaly) can be seen in test data) for
        best results (hmean-auc-s+hmean-auc-t+hmean-pauc)

        step: search step/100 on [start/100, end/100)
        gmm_ns: components of gmm list for searching
        use_smote: using smote on target domain of train data or not
        """
        # set gwrp pool type
        if gmm_ns is None:
            gmm_ns = [1, 2]
        best_gwrp_decays = {}
        best_gmm_n = {}
        best_metrics = {}
        best_sum_metrics = {}
        machine_list = []
        for train_dir in train_dirs:
            machine = train_dir.split('/')[-2]
            machine_list.append(machine)
        for machine in machine_list:
            best_metrics[machine] = {}
            best_sum_metrics[machine] = 0
        result_dir = os.path.join('./results', self.args.version)
        os.makedirs(result_dir, exist_ok=True)
        for gmm_n in gmm_ns:
            for decay in np.arange(start, end, step):
                decay /= 100
                self.feature_extractor.args.decay = decay
                csv_lines = []
                sum_auc_s, sum_auc_t, sum_pauc, num, total_time = 0, 0, 0, 0, 0
                h_sum_auc_s, h_sum_auc_t, h_sum_pauc = 0, 0, 0
                self.logger.info('\n' + '=' * 20)
                for index, (target_dir, train_dir) in enumerate(zip(sorted(valid_dirs), sorted(train_dirs))):
                    time.sleep(1)
                    time_start = time.perf_counter()
                    machine_type = target_dir.split('/')[-2]
                    # result csv
                    machine_section_list = utils.get_machine_section_list(target_dir)
                    csv_lines.append([machine_type])
                    csv_lines.append(['section', 'AUC(Source)', 'AUC(Target)', 'pAUC'])
                    performance = []
                    for section_str in machine_section_list:
                        # train GMM
                        self.logger.info(f'[{machine_type}|{section_str}|gmm_n={gmm_n}|decay={decay:.2f}] Fit GMM-{gmm_n}...')
                        s_train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_source_*')
                        t_train_files = utils.get_filename_list(train_dir, pattern=f'{section_str}_target_*')
                        train_files = s_train_files + t_train_files
                        self.logger.info(f'number of {section_str} files: {len(train_files)}')
                        features = self.feature_extractor.extract(train_files) if not use_smote else \
                            self.feature_extractor.smote_extract(s_train_files, t_train_files)
                        gmm = self.fit_GMM(features, n_components=gmm_n)
                        # get test info
                        test_files, y_true, domain_list = utils.get_valid_file_list(target_dir, section_str)
                        y_pred = [0. for _ in test_files]
                        for file_idx, test_file in enumerate(test_files):
                            test_feature = self.feature_extractor.extract([test_file])
                            y_pred[file_idx] = - np.max(gmm._estimate_log_prob(test_feature))
                        # compute auc and pAuc
                        auc_s, auc_t, p_auc = utils.cal_auc_pauc(y_true, y_pred, domain_list)
                        performance.append([auc_s, auc_t, p_auc])
                        csv_lines.append([section_str.split('_', 1)[1], auc_s, auc_t, p_auc])

                    # calculate averages for AUCs and pAUCs
                    amean_performance = np.mean(np.array(performance, dtype=float), axis=0)
                    hmean_performance = scipy.stats.hmean(
                        np.maximum(np.array(performance, dtype=float), sys.float_info.epsilon), axis=0)
                    mean_auc_s, mean_auc_t, mean_p_auc = amean_performance[0], amean_performance[1], amean_performance[2]
                    h_mean_auc_s, h_mean_auc_t, h_mean_p_auc = hmean_performance[0], hmean_performance[1], hmean_performance[2]
                    sum_auc_s += mean_auc_s
                    sum_auc_t += mean_auc_t
                    sum_pauc += mean_p_auc
                    h_sum_auc_s += h_mean_auc_s
                    h_sum_auc_t += h_mean_auc_t
                    h_sum_pauc += h_mean_p_auc
                    num += 1
                    time_nedded = time.perf_counter() - time_start
                    total_time += time_nedded
                    self.logger.info(f'[gmm_n={gmm_n}|decay={decay:.2f}] Test {machine_type}\tcost {time_nedded:.2f} sec\tavg_auc_s: {mean_auc_s:.3f}\t'
                                     f'avg_auc_t: {mean_auc_t:.3f}\tavg_pauc: {mean_p_auc:.3f}')
                    # best results on h_mean_auc_s + h_mean_auc_t + h_mean_p_auc
                    sum_metrics = h_mean_auc_s + h_mean_auc_t + h_mean_p_auc
                    if sum_metrics > best_sum_metrics[machine_type]:
                        best_sum_metrics[machine_type] = sum_metrics
                        best_gwrp_decays[machine_type] = decay
                        best_gmm_n[machine_type] = gmm_n
                        best_metrics[machine_type]['avg_auc_s'] = h_mean_auc_s
                        best_metrics[machine_type]['avg_auc_t'] = h_mean_auc_t
                        best_metrics[machine_type]['avg_p_auc'] = h_mean_p_auc
                    # best results on mean_auc_s + mean_auc_t + mean_p_auc
                    # sum_metrics = mean_auc_s + mean_auc_t + mean_p_auc
                    # if sum_metrics > best_sum_metrics[machine_type]:
                    #     best_sum_metrics[machine_type] = sum_metrics
                    #     best_gwrp_decays[machine_type] = decay
                    #     best_metrics[machine_type]['avg_auc_s'] = mean_auc_s
                    #     best_metrics[machine_type]['avg_auc_t'] = mean_auc_t
                    #     best_metrics[machine_type]['avg_p_auc'] = mean_p_auc
                print(f'Total test time: {total_time:.2f} sec')
                avg_auc_s, avg_auc_t, avg_pauc = sum_auc_s / num, sum_auc_t / num, sum_pauc / num
                h_avg_auc_s, h_avg_auc_t, h_avg_pauc = h_sum_auc_s / num, h_sum_auc_t / num, h_sum_pauc / num
                self.logger.info(f'h_avg_auc_s: {h_avg_auc_s:.3f}\th_avg_auc_t: {h_avg_auc_t:.3f}\th_avg_pauc: {h_avg_pauc:.3f}')
        # record searching result
        result_path = os.path.join(result_dir, f'result-gmm-{gmm_ns}.csv')
        csv_lines = []
        sum_auc_s, sum_auc_t, sum_p_auc, num = 0, 0, 0, 0
        for machine in machine_list:
            csv_lines.append([machine, 'AUC(Source)', 'AUC(Target)', 'pAUC'])
            auc_s = best_metrics[machine]['avg_auc_s']
            auc_t = best_metrics[machine]['avg_auc_t']
            p_auc = best_metrics[machine]['avg_p_auc']
            decay = best_gwrp_decays[machine]
            gmm_n = best_gmm_n[machine]
            csv_lines.append([f'gmm_n={gmm_n}', f'decay={decay:.2f}', f'{auc_s:.4f}', f'{auc_t:.4f}', f'{p_auc:.4f}'])
            csv_lines.append([])
            sum_auc_s += auc_s
            sum_auc_t += auc_t
            sum_p_auc += p_auc
            num += 1
        avg_auc_s, avg_auc_t, avg_pauc = sum_auc_s / num, sum_auc_t / num, sum_p_auc / num
        csv_lines.append(['Total Average', f'{avg_auc_s:.4f}', f'{avg_auc_t:.4f}', f'{avg_pauc:.4f}'])
        utils.save_csv(result_path, csv_lines)