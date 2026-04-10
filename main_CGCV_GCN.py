#!/usr/bin/env python
from __future__ import print_function

import argparse
import inspect
import os
import pickle
import random
import shutil
import time
from collections import OrderedDict

import seaborn as sns  # 导入seaborn库，用于绘制热图
import matplotlib.pyplot as plt

import numpy as np
# torch
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from torch.optim.lr_scheduler import _LRScheduler
from tqdm import tqdm

import torch.nn.functional as F
from sklearn.metrics import accuracy_score, average_precision_score, precision_score, f1_score, recall_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


class GradualWarmupScheduler(_LRScheduler):
    def __init__(self, optimizer, total_epoch, after_scheduler=None):
        self.total_epoch = total_epoch
        self.after_scheduler = after_scheduler
        self.finished = False
        self.last_epoch = -1
        super().__init__(optimizer)

    def get_lr(self):
        return [base_lr * (self.last_epoch + 1) / self.total_epoch for base_lr in self.base_lrs]

    def step(self, epoch=None, metric=None):
        if self.last_epoch >= self.total_epoch - 1:
            if metric is None:
                return self.after_scheduler.step(epoch)
            else:
                return self.after_scheduler.step(metric, epoch)
        else:
            return super(GradualWarmupScheduler, self).step(epoch)

def init_seed(seed):
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    # torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_parser():
    # parameter priority: command line > config > default
    parser = argparse.ArgumentParser(
        description='Spatial Temporal Graph Convolution Network')
    parser.add_argument(
        '--work-dir',
        default='./work_dir/temp',
        help='the work folder for storing results')

    parser.add_argument('-model_saved_name', default='')
    parser.add_argument(
        '--config',
        default='./config/EGait_journal/train_diff_combine_double_score_fagg.yaml',
        # default='./config/kinetics-skeleton/train_joint.yaml',
        help='path to the configuration file')

    # processor
    parser.add_argument(
        '--phase', default='train', help='must be train or test')
    parser.add_argument(
        '--save-score',
        type=str2bool,
        default=False,
        help='if ture, the classification score will be stored')

    # visulize and debug
    parser.add_argument(
        '--seed', type=int, default=3407, help='random seed for pytorch')
    parser.add_argument(
        '--log-interval',
        type=int,
        default=100,
        help='the interval for printing messages (#iteration)')
    parser.add_argument(
        '--save-interval',
        type=int,
        default=2,
        help='the interval for storing models (#iteration)')
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=5,
        help='the interval for evaluating models (#iteration)')
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default=True,
        help='print logging or not')
    parser.add_argument(
        '--show-topk',
        type=int,
        default=[1, 2],
        nargs='+',
        help='which Top K accuracy will be shown')

    # feeder
    parser.add_argument(
        '--feeder', default='feeder.feeder', help='data loader will be used')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=20,
        help='the number of worker for data loader')
    parser.add_argument(
        '--train-feeder-args',
        default=dict(),
        help='the arguments of data loader for training')
    parser.add_argument(
        '--test-feeder-args',
        default=dict(),
        help='the arguments of data loader for test')

    parser.add_argument('--train_ratio', default=0.9)
    parser.add_argument('--val_ratio', default=0.0)
    parser.add_argument('--test_ratio', default=0.1)

    # model
    parser.add_argument('--model', default=None, help='the model will be used')
    parser.add_argument(
        '--model-args',
        type=dict,
        default=dict(),
        help='the arguments of model')
    parser.add_argument(
        '--weights',
        default=None,
        help='the weights for network initialization')
    parser.add_argument(
        '--ignore-weights',
        type=str,
        default=[],
        nargs='+',
        help='the name of weights which will be ignored in the initialization')

    # optim
    parser.add_argument(
        '--base-lr', type=float, default=0.01, help='initial learning rate')
    parser.add_argument(
        '--step',
        type=int,
        default=[20, 40, 60],
        nargs='+',
        help='the epoch where optimizer reduce the learning rate')
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        nargs='+',
        help='the indexes of GPUs for training or testing')
    parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
    parser.add_argument(
        '--nesterov', type=str2bool, default=False, help='use nesterov or not')
    parser.add_argument(
        '--batch-size', type=int, default=2, help='training batch size')
    parser.add_argument(
        '--test-batch-size', type=int, default=2, help='test batch size')
    parser.add_argument(
        '--start-epoch',
        type=int,
        default=0,
        help='start training from which epoch')
    parser.add_argument(
        '--num-epoch',
        type=int,
        default=80,
        help='stop training in which epoch')
    parser.add_argument(
        '--weight_decay',
        type=float,
        default=0.0005,
        help='weight decay for optimizer')
    parser.add_argument('--save_model', default=False)
    parser.add_argument('--only_train_part', default=False)
    parser.add_argument('--only_train_epoch', default=0)
    parser.add_argument('--warm_up_epoch', default=0)
    return parser


class Processor():
    def __init__(self, arg):
        self.arg = arg
        if arg.phase == 'train':
            self.save_arg()
            if not arg.train_feeder_args['debug']:
                if os.path.isdir(arg.model_saved_name):
                    print('log_dir: ', arg.model_saved_name, 'already exist')
                    answer = input('delete it? y/n:')
                    if answer == 'y':
                        shutil.rmtree(arg.model_saved_name)
                        print('Dir removed: ', arg.model_saved_name)
                        input('Refresh the website of tensorboard by pressing any keys')
                    else:
                        print('Dir not removed: ', arg.model_saved_name)
                self.train_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'train'), 'train')
                self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'val'), 'val')
            else:
                self.train_writer = self.val_writer = SummaryWriter(os.path.join(arg.model_saved_name, 'test'), 'test')
        self.global_step = 0
        self.load_model()
        self.load_optimizer()
        self.load_data()
        self.lr = self.arg.base_lr
        self.best_acc = 0
        self.best_H = 0
        self.best_S = 0
        self.best_A = 0
        self.best_N = 0
        self.Bh, self.Bn, self.Ba, self.Bs = 0, 0, 0, 0
        self.epoch_num = 0

    def load_data(self):
        Feeder = import_class(self.arg.feeder)
        self.data_loader = dict()

        if self.arg.phase == 'train':
            self.data_loader['train'] = torch.utils.data.DataLoader(
                dataset=Feeder(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker,
                drop_last=True,
                worker_init_fn=init_seed)
        self.data_loader['test'] = torch.utils.data.DataLoader(
            dataset=Feeder(**self.arg.test_feeder_args),
            batch_size=self.arg.test_batch_size,
            shuffle=False,
            num_workers=self.arg.num_worker,
            drop_last=False,
            worker_init_fn=init_seed)

    def load_model(self):
        output_device = self.arg.device[0] if type(self.arg.device) is list else self.arg.device
        self.output_device = output_device
        Model = import_class(self.arg.model)
        shutil.copy2(inspect.getfile(Model), self.arg.work_dir)
        # print(Model)
        self.model = Model(**self.arg.model_args).cuda(output_device)
        # print(self.model)
        self.loss = nn.CrossEntropyLoss().cuda(output_device)
        # self.loss = nn.BCELoss().cuda(output_device)
        self.loss2 = nn.MSELoss().cuda(output_device)
        if self.arg.weights:
            # self.global_step = int(arg.weights[:-3].split('-')[-1])
            self.print_log('Load weights from {}.'.format(self.arg.weights))
            if '.pkl' in self.arg.weights:
                with open(self.arg.weights, 'r') as f:
                    weights = pickle.load(f)
            else:
                weights = torch.load(self.arg.weights)

            weights = OrderedDict(
                [[k.split('module.')[-1],
                  v.cuda(output_device)] for k, v in weights.items()])

            keys = list(weights.keys())
            for w in self.arg.ignore_weights:
                for key in keys:
                    if w in key:
                        if weights.pop(key, None) is not None:
                            self.print_log('Sucessfully Remove Weights: {}.'.format(key))
                        else:
                            self.print_log('Can Not Remove Weights: {}.'.format(key))

            try:
                self.model.load_state_dict(weights)
            except:
                state = self.model.state_dict()
                diff = list(set(state.keys()).difference(set(weights.keys())))
                print('Can not find these weights:')
                for d in diff:
                    print('  ' + d)
                state.update(weights)
                self.model.load_state_dict(state)

        if type(self.arg.device) is list:
            if len(self.arg.device) > 1:
                self.model = nn.DataParallel(
                    self.model,
                    device_ids=self.arg.device,
                    output_device=output_device)

    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

        lr_scheduler_pre = optim.lr_scheduler.MultiStepLR(
            self.optimizer, milestones=self.arg.step, gamma=0.1)

        self.lr_scheduler = GradualWarmupScheduler(self.optimizer, total_epoch=self.arg.warm_up_epoch,
                                                   after_scheduler=lr_scheduler_pre)
        self.print_log('using warm up, epoch: {}'.format(self.arg.warm_up_epoch))

    def save_arg(self):
        # save arg
        arg_dict = vars(self.arg)
        if not os.path.exists(self.arg.work_dir):
            os.makedirs(self.arg.work_dir)
        with open('{}/config.yaml'.format(self.arg.work_dir), 'w') as f:
            yaml.dump(arg_dict, f)

    def adjust_learning_rate(self, epoch):
        if self.arg.optimizer == 'SGD' or self.arg.optimizer == 'Adam':
            if epoch < self.arg.warm_up_epoch:
                lr = self.arg.base_lr * (epoch + 1) / self.arg.warm_up_epoch
            else:
                lr = self.arg.base_lr * (
                        0.1 ** np.sum(epoch >= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            return lr
        else:
            raise ValueError()

    def print_time(self):
        localtime = time.asctime(time.localtime(time.time()))
        self.print_log("Local current time :  " + localtime)

    def print_log(self, str, print_time=True):
        if print_time:
            localtime = time.asctime(time.localtime(time.time()))
            str = "[ " + localtime + ' ] ' + str
        print(str)
        if self.arg.print_log:
            with open('{}/log.txt'.format(self.arg.work_dir), 'a') as f:
                print(str, file=f)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def split_time(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def plot_confusion_matrix(self, conf_matrix, class_names, log_path, dpi=600):
        plt.figure(figsize=(8, 6))
        sns.heatmap(conf_matrix, annot=True, fmt=".8f", cmap='Oranges',
                    xticklabels=class_names, yticklabels=class_names,
                    cbar=True, cbar_kws={'label': 'Intensity'})  # Added colorbar with label

        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')

        save_path = os.path.join(log_path, "confusion_matrix.png")

        plt.savefig(save_path, dpi=dpi)
        plt.close()

    def train(self, epoch, save_model=False):
        a = 0.5
        self.model.train()
        self.print_log('Training epoch: {}'.format(epoch + 1))
        loader = self.data_loader['train']
        self.adjust_learning_rate(epoch)
        # for name, param in self.model.named_parameters():
        #     self.train_writer.add_histogram(name, param.clone().cpu().data.numpy(), epoch)

        loss_value = []
        self.train_writer.add_scalar('epoch', epoch, self.global_step)
        self.record_time()
        timer = dict(dataloader=0.001, model=0.001, statistics=0.001)
        process = tqdm(loader)

        epoch_start_time = time.time()

        if self.arg.only_train_part:
            if epoch > self.arg.only_train_epoch:
                print('only train part, require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = True
                        # print(key + '-require grad')
            else:
                print('only train part, do not require grad')
                for key, value in self.model.named_parameters():
                    if 'PA' in key:
                        value.requires_grad = False
                        # print(key + '-not require grad')
        train_class_total_num = np.array([0, 0, 0, 0])
        train_class_true_num = np.array([0, 0, 0, 0])
        total_acc, cnt = 0, 0

        epoch_start_time = time.time()  # epoch 开始时间
        for batch_idx, (data_p, data_k, label, feature, index) in enumerate(process):
            if len(label.size()) > 1:
                train_mode = 'MLL'
            else:
                train_mode = 'SLL'

            self.global_step += 1

            data_k = data_k.float().cuda(self.output_device).requires_grad_(False)
            data_p = data_p.float().cuda(self.output_device).requires_grad_(False)
            data_a = feature.float().cuda(self.output_device).requires_grad_(False)
            label = label.long().cuda(self.output_device).requires_grad_(False)

            if train_mode == 'MLL': label = label.to(torch.float32)
            feature = feature.float().cuda(self.output_device).requires_grad_(False)
            timer['dataloader'] += self.split_time()

            # forward
            output_p, output_k, output_a, output2_pa, output2_ka = self.model(data_p, data_k, data_a)
            output1 = (output_p + output_k + output_a) / 3
            output2 = (output2_pa + output2_ka) / 2
            output = (output1 + output2) / 2

            if train_mode == 'MLL':
                output_p = F.sigmoid(output_p)
                output_k = F.sigmoid(output_k)
                output_a = F.sigmoid(output_a)
                output2_pa = F.sigmoid(output2_pa)
                output2_am = F.sigmoid(output2_am)

            loss1_p = self.loss(output_p, label)
            loss1_k = self.loss(output_k, label)
            loss1_a = self.loss(output_a, label)
            loss2_pa = self.loss(output2_pa, label)
            loss2_ka = self.loss(output2_ka, label)

            loss1 = (loss1_p + loss1_k + loss1_a) / 3
            loss2 = (loss2_pa + loss2_ka) / 2

            loss = a * loss1 + (1 - a) * loss2

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            loss_value.append(loss.data.item())

            timer['model'] += self.split_time()

            if train_mode == 'SLL':
                value, predict_label = torch.max(output.data, 1)
                total_acc += torch.sum((predict_label == label.data).float())
                cnt += label.size(0)
                trues = list(label.data.cpu().numpy())
                for idx, lb in enumerate(predict_label):
                    train_class_total_num[trues[idx]] += 1
                    train_class_true_num[trues[idx]] += int(lb == trues[idx])
            else:
                total_acc += torch.round(output).eq(label).sum()
                cnt += label.numel()
                class_total_num = torch.round(output).eq(1).sum(axis=0)
                class_true_num = (torch.round(output).eq(label) & label.eq(1)).sum(axis=0)
                for idx in range(len(class_total_num)): train_class_total_num[idx] += class_total_num[idx]
                for idx in range(len(class_true_num)): train_class_true_num[idx] += class_true_num[idx]

            self.train_writer.add_scalar('loss', loss.data.item(), self.global_step)
            self.train_writer.add_scalar('loss_1p', loss1_p, self.global_step)
            self.train_writer.add_scalar('loss_1m', loss1_k, self.global_step)

            self.train_writer.add_scalar('loss_2', loss2, self.global_step)

            # statistics
            self.lr = self.optimizer.param_groups[0]['lr']
            self.train_writer.add_scalar('lr', self.lr, self.global_step)

            timer['statistics'] += self.split_time()

        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        mean_loss = np.mean(loss_value)

        self.epoch_losses.append(mean_loss)
        self.epoch_times.append(epoch_time)
        self.train_writer.add_scalar('loss', mean_loss, self.global_step)

        # statistics of time consumption and loss XieJC
        proportion = {
            k: '{:02d}%'.format(int(round(v * 100 / sum(timer.values()))))
            for k, v in timer.items()
        }
        self.print_log(  # 平均loss
            '\tMean training loss: {:.4f}.'.format(np.mean(loss_value)))
        self.print_log(
            '\tTime consumption: [Data]{dataloader}, [Network]{model}'.format(
                **proportion))
        print('Happy:{},Sad:{},Angry:{},Neutral:{}'.format(train_class_true_num[0] * 1.0 / train_class_total_num[0],
                                                           train_class_true_num[1] * 1.0 / train_class_total_num[1],
                                                           train_class_true_num[2] * 1.0 / train_class_total_num[2],
                                                           train_class_true_num[3] * 1.0 / train_class_total_num[3]))
        print('Train Accuracy: {: .2f}%'.format(100 * total_acc * 1.0 / cnt))
        print(f"Epoch {epoch + 1} - Learning rate: {self.optimizer.param_groups[0]['lr']}")


    def eval(self, epoch, save_score=False, loader_name=['test'], wrong_file=None,
             # '/home/Timeless/egait_runs/wrong.txt'
             result_file=None, log_path='/home/Timeless/egait_runs'):
        if wrong_file is not None:
            f_w = open(wrong_file, 'w')
        if result_file is not None:
            f_r = open(result_file, 'a')  # 使用'a'模式来追加记录

        self.model.eval()

        test_class_total_num = np.array([0, 0, 0, 0])
        test_class_true_num = np.array([0, 0, 0, 0])
        test_class_pred_num = np.array([0, 0, 0, 0])
        total_acc, cnt = 0, 0
        total_label = np.array([])  # 用于保存所有标签
        total_pre = np.array([])  # 用于保存所有预测结果

        self.print_log('Eval epoch: {}'.format(epoch + 1))
        for ln in loader_name:
            loss_value = []
            score_frag = []
            right_num_total = 0
            total_num = 0
            loss_total = 0

            class_right_num = [0, 0, 0, 0]
            class_total_num = [0, 0, 0, 0]
            step = 0
            process = tqdm(self.data_loader[ln])
            for batch_idx, (data_p, data_k, label, feature, index) in enumerate(process):
                if len(label.size()) > 1:
                    test_mode = 'MLL'
                else:
                    test_mode = 'SLL'
                with torch.no_grad():
                    data_k = data_k.float().cuda(self.output_device)
                    data_p = data_p.float().cuda(self.output_device)
                    data_a = feature.float().cuda(self.output_device)
                    # feature = feature.float().cuda(self.output_device)
                    label = label.long().cuda(self.output_device)

                    if test_mode == 'MLL':
                        label = label.to(torch.float32)

                    output_p, output_k, output_a, output2_pa, output2_ka = self.model(data_p, data_k, data_a)
                    output1 = (output_p + output_k + output_a) / 3
                    output2 = (output2_pa + output2_ka) / 2
                    output = (output1 + output2) / 2

                    if test_mode == 'MLL':
                        output = F.sigmoid(output)

                    loss = self.loss(output, label)
                    loss_value.append(loss.data.item())

                    _, predict_label = torch.max(output.data, 1)
                    step += 1

                    if test_mode == 'SLL':
                        trues = list(label.data.cpu().numpy())
                        for idx, lb in enumerate(predict_label):
                            test_class_total_num[trues[idx]] += 1
                            test_class_true_num[trues[idx]] += int(lb == trues[idx])

                        total_acc += (predict_label == label).sum()
                        cnt += label.size(0)

                        output_flat = torch.round(predict_label).view(-1).cpu().numpy()
                        label_flat = label.view(-1).cpu().numpy()
                        total_label = np.concatenate((total_label, label_flat))
                        total_pre = np.concatenate((total_pre, output_flat))
                    else:
                        total_acc += torch.round(output).eq(label).sum()
                        cnt += label.numel()

                        class_total_num = torch.round(output).eq(1).sum(axis=0)
                        class_true_num = (torch.round(output).eq(label) & label.eq(1)).sum(axis=0)

                        for idx in range(len(class_total_num)):
                            test_class_total_num[idx] += class_total_num[idx]
                        for idx in range(len(class_true_num)):
                            test_class_true_num[idx] += class_true_num[idx]

                        output_flat = torch.round(predict_label).view(-1).cpu().numpy()
                        label_flat = label.view(-1).cpu().numpy()
                        total_label = np.concatenate((total_label, label_flat))
                        total_pre = np.concatenate((total_pre, output_flat))

                    if wrong_file is not None or result_file is not None:
                        predict = list(predict_label.cpu().numpy())
                        true = list(label.data.cpu().numpy())
                        for i, x in enumerate(predict):
                            if result_file is not None:
                                f_r.write(str(x) + ',' + str(true[i]) + '\n')
                            if x != true[i] and wrong_file is not None:
                                f_w.write(str(index[i]) + ',' + str(x) + ',' + str(true[i]) + '\n')

            loss = np.mean(loss_value)

            sklearn_precision = precision_score(total_label, total_pre, average='macro', zero_division=1)
            sklearn_recall = recall_score(total_label, total_pre, average='macro', zero_division=1)
            sklearn_f1 = f1_score(total_label, total_pre, average='macro', zero_division=1)

            accuracy = total_acc * 1.0 / cnt
            Happy = test_class_true_num[0] * 1.0 / test_class_total_num[0]
            Sad = test_class_true_num[1] * 1.0 / test_class_total_num[1]
            Angry = test_class_true_num[2] * 1.0 / test_class_total_num[2]
            Neutral = test_class_true_num[3] * 1.0 / test_class_total_num[3]

            evaluation_metrics_file = os.path.join(log_path, 'evaluation_metrics.txt')

            if accuracy > self.best_acc:
                # con_mat = confusion_matrix(y_true=total_label, y_pred=total_pre)
                # con_mat_norm = con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis]
                # cm = np.around(con_mat_norm, decimals=8)
                # class_names = ['Happy', 'Sad', 'Angry', 'Neutral']
                # self.plot_confusion_matrix(cm, class_names, log_path)
                # confusion_detail_file = os.path.join(log_path, 'confusion_matrix_details.txt')
                # with open(confusion_detail_file, 'w') as f:
                #     f.write("Confusion Matrix Detail (Rows: True Classes, Columns: Predicted Classes)\n")
                #     f.write("True Class | Predicted Class | Count | True Class Total | Predicted Class Total\n")
                #     for i in range(4):
                #         for j in range(4):
                #             count = con_mat[i, j]
                #             true_class_total = con_mat[i, :].sum()  # 实际类别i的总数
                #             predicted_class_total = con_mat[:, j].sum()  # 被预测为类别j的总数
                #             f.write(f"{i} | {j} | {count} | {true_class_total} | {predicted_class_total}\n")

                self.best_acc = accuracy
                self.epoch_num = epoch
                self.Bh = Happy
                self.Bn = Neutral
                self.Ba = Angry
                self.Bs = Sad

                state_dict = self.model.state_dict()
                weights = OrderedDict([[k.split('module.')[-1], v.cpu()] for k, v in state_dict.items()])
                torch.save(weights, self.arg.model_saved_name + '/best_acc_mkDit.pt')

                self.pre_b = sklearn_precision
                self.recall_b = sklearn_recall
                self.f1_b = sklearn_f1

                with open(evaluation_metrics_file, 'w') as f:
                    f.write(f"epoch: {self.epoch_num + 1}\n")
                    f.write(f"Top1: {accuracy * 100:.4f}%\n")
                    f.write(f"precision: {sklearn_precision * 100:.4f}%\n")
                    f.write(f"recall: {sklearn_recall * 100:.4f}%\n")
                    f.write(f"f1-score: {sklearn_f1 * 100:.4f}%\n")
                    f.write(
                        f"Happy: {Happy * 100:.4f}%, Sad: {Sad * 100:.4f}%, Angry: {Angry * 100:.4f}%, Neutral: {Neutral * 100:.4f}%\n")

            elif accuracy == self.best_acc:
                with open(evaluation_metrics_file, 'a') as f:
                    f.write(f"epoch: {epoch + 1}\n")
                    f.write(f"Top1: {accuracy * 100:.4f}%\n")
                    f.write(f"precision: {sklearn_precision * 100:.4f}%\n")
                    f.write(f"recall: {sklearn_recall * 100:.4f}%\n")
                    f.write(f"f1-score: {sklearn_f1 * 100:.4f}%\n")
                    f.write(
                        f"Happy: {Happy * 100:.4f}%, Sad: {Sad * 100:.4f}%, Angry: {Angry * 100:.4f}%, Neutral: {Neutral * 100:.4f}%\n")

            self.print_log(f'\tMean {ln} loss of {len(self.data_loader[ln])} batches: {np.mean(loss_value)}.')
            print(f'Top1: {accuracy * 100:.2f}%')
            self.print_log(f'Best acc: {self.best_acc * 100:.2f}%')
            print(f'Happy:{Happy:.4f}, Sad:{Sad:.4f}, Angry:{Angry:.4f}, Neutral:{Neutral:.4f}')

        if wrong_file is not None:
            f_w.close()
        if result_file is not None:
            f_r.close()

    def start(self):
        self.epoch_losses = []
        self.epoch_times = []

        if self.arg.phase == 'train':
            self.print_log('Parameters:\n{}\n'.format(str(vars(self.arg))))
            self.global_step = self.arg.start_epoch * len(self.data_loader['train']) / self.arg.batch_size
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                if self.lr < 1e-6:
                    break
                save_model = False
                start = time.time()
                self.train(epoch, save_model=save_model)
                end = time.time()
                print(end - start)

                start = time.time()
                self.eval(
                    epoch,
                    save_score=self.arg.save_score,
                    loader_name=['test'])
                end = time.time()
                print(end - start)

            self.print_log('best accuracy: {}'.format(self.best_acc))


        elif self.arg.phase == 'test':
            if not self.arg.test_feeder_args['debug']:
                wf = self.arg.model_saved_name + '/wrong.txt'
                rf = self.arg.model_saved_name + '/right.txt'
            else:
                wf = rf = None
            if self.arg.weights is None:
                raise ValueError('Please appoint --weights.')
            self.arg.print_log = False
            self.print_log('Model:   {}.'.format(self.arg.model))
            self.print_log('Weights: {}.'.format(self.arg.weights))
            self.eval(epoch=0, save_score=self.arg.save_score, loader_name=['test'], wrong_file=wf, result_file=rf)
            self.print_log('Done.\n')



def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])  # import return model
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


if __name__ == '__main__':
    parser = get_parser()

    # load arg form config file
    p = parser.parse_args()
    if p.config is not None:
        with open(p.config, 'r') as f:
            default_arg = yaml.load(f, Loader=yaml.FullLoader)
        key = vars(p).keys()
        for k in default_arg.keys():
            if k not in key:
                print('WRONG ARG: {}'.format(k))
                assert (k in key)
        parser.set_defaults(**default_arg)

    arg = parser.parse_args()
    init_seed(arg.seed)
    processor = Processor(arg)
    processor.start()
