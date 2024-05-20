# -*- coding: utf-8 -*-

import os
import time
import random
import logging
import argparse
import numpy as np
import pandas as pd

import nni
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score

from models.linkpred import LinkPredictor
from layers.propagation import calc_pagerank_iters
import precompute.push as push
from datasets import load_dataset, get_split_edge
from utils import get_pos_neg_edges, AUCLoss

# enable deterministic about CUBLAS
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ':4096:8'


class Instructor:
    def __init__(self, opt) -> None:
        # logging
        # self.logger = logging.getLogger(name=opt['dataset_str'])
        # log_path = './log/' + opt['dataset_str'] + '_link/'
        # if not os.path.exists(log_path):
        #     os.mkdir(log_path)
        # log_path += opt['mechanism']+'_'+str(opt['eps'])+'.log'
        # file_handler = logging.FileHandler(log_path)
        # file_handler.setLevel(level=logging.INFO)
        # self.logger.addHandler(file_handler)

        # self.logger.info('-' * 80)
        print('-' * 80)
        self.opt = opt

    def _load_data_init_model(self):
        # load dataset and perturb the features
        self.dataset = load_dataset(self.opt, 'link')
        self.split_edge = get_split_edge(self.opt['dataset_str'])
        # propagate
        if self.opt['propagation'] == 'power':
            self.features = calc_pagerank_iters(
                self.dataset.adj, self.dataset.attr,
                self.opt['alpha'], self.opt['niter'])
            self.num_nodes = self.dataset.adj.shape[0]
            self.dataset.adj = None
        else:
            self.num_nodes = self.dataset.adj.shape[0]
            self.dataset.adj = self.dataset.adj.tocoo()
            edges = np.stack((self.dataset.adj.row, self.dataset.adj.col))
            self.dataset.adj = None
            self.features = torch.FloatTensor(push.calc_pagerank_push(
                edges, self.dataset.attr, 
                self.num_nodes, self.opt['alpha'], self.opt['rmax'], self.opt['rrz']))

        self.loss_func = AUCLoss()
        model_args = {
            'in_channels': self.features.shape[1],
            'out_channels': 1,
            'hidden_channels': self.opt['hidden_channels'],
            'num_layers': self.opt['num_layers'],
            'drop_prob': self.opt['drop_prob']
        }

        self.features = self.features.to(self.opt['device'])
        self.model = LinkPredictor(**model_args).to(self.opt['device'])
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.opt['lr'], weight_decay=self.opt['reg_lambda'])

        self._print_args()
        self.global_eval = 0.

    def _print_args(self) -> None:
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape)).item()
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        # self.logger.info('n_trainable_params: {}, n_nontrainable_params: {}'.format(n_trainable_params, n_nontrainable_params))
        print('n_trainable_params: {}, n_nontrainable_params: {}'.format(n_trainable_params, n_nontrainable_params))
        # self.logger.info('> training arguments:')
        print('> training arguments:')
        for arg in self.opt:
            # self.logger.info('>>> {}: {}'.format(arg, self.opt[arg]))
            print('>>> {}: {}'.format(arg, self.opt[arg]))

    def _reset_params(self) -> None:
        for param in self.model.parameters():
            # init the weight
            if param.requires_grad and len(param.shape) > 1:
                nn.init.kaiming_uniform_(param, a=0, mode='fan_out', nonlinearity='relu')

    def _train(self) -> float:
        max_val_eval = 0.
        continue_not_increase = 0

        start_time = time.time()
        last_time = start_time

        for epoch in range(self.opt['epochs']):
            pos_train_edge, neg_train_edge = get_pos_neg_edges(
                                            self.split_edge, 'train',
                                            self.num_nodes)
            # pos_train_edge = pos_train_edge.to(self.opt['device'])
            # neg_train_edge = neg_train_edge.to(self.opt['device'])
            train_data_loader = DataLoader(
                dataset=range(pos_train_edge.size(0)), 
                batch_size=self.opt['batch_size'], shuffle=True)
            total_loss = total_examples = 0
            increase_flag = False

            for perm in train_data_loader:
                pos_edge = pos_train_edge[perm].t()
                neg_edge = neg_train_edge[perm].t()
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                self.optimizer.zero_grad()

                pos_out = self.model(
                    self.features[pos_edge[0]], self.features[pos_edge[1]])
                neg_out = self.model(
                    self.features[neg_edge[0]], self.features[neg_edge[1]])

                loss = self.loss_func(pos_out, neg_out)
                loss.backward()
                self.optimizer.step()

                total_examples += pos_out.size(0)
                total_loss += loss.item() * pos_out.size(0)

            val_eval = self._evaluate('valid')
            test_eval = self._evaluate('test')

            # nni.report_intermediate_result(val_eval)
    
            if (epoch + 1) % self.opt['log_steps'] == 0:
                # self.logger.info('>' * 80)
                print('>' * 80)
                duration = time.time() - last_time
                last_time = time.time()
                to_print = (f'epoch: {epoch + 1}, '
                            f'train loss: {total_loss / total_examples:.3f}, '
                            f'val eval: {100 * val_eval:.1f}, '
                            f'test eval: {100 * test_eval:.1f}, '
                            f'cost {duration:.3f} s')
                # self.logger.info(to_print)
                print(to_print)

            if val_eval > max_val_eval:
                increase_flag = True
                max_val_eval = val_eval
                if self.opt['save'] and val_eval > self.global_eval:
                    model_path = './state_dict/' + self.opt['dataset_str'] + '_link_' + \
                        self.opt['mechanism'] + '_' + str(self.opt['eps']) + '.pkl'
                    self.global_eval = val_eval
                    torch.save(self.model.state_dict(), model_path)

            if increase_flag == False:
                continue_not_increase +=1
                if continue_not_increase >= self.opt['patience']:
                    break
            else:
                continue_not_increase = 0
        # nni.report_final_result(val_eval)

        return max_val_eval

    @torch.no_grad()
    def _evaluate(self, mode) -> float:
        # switch model to evaluation mode
        self.model.eval()
        pos_edge, neg_edge = get_pos_neg_edges(self.split_edge, mode)
        # pos_edge, neg_edge = pos_edge.to(self.opt['device']), neg_edge.to(self.opt['device'])
        pos_preds, neg_preds = [], []
        for perm in DataLoader(range(pos_edge.size(0)), self.opt['batch_size']):
            edge = pos_edge[perm].t()
            pos_preds += [self.model(self.features[edge[0]], 
                self.features[edge[1]]).squeeze().cpu()]
        pos_preds = torch.cat(pos_preds, dim=0)
        for perm in DataLoader(range(neg_edge.size(0)), self.opt['batch_size']):
            edge = neg_edge[perm].t()
            neg_preds += [self.model(self.features[edge[0]], 
                self.features[edge[1]]).squeeze().cpu()]
        neg_preds = torch.cat(neg_preds, dim=0)
        y_pred = torch.cat((pos_preds, neg_preds), dim=0)
        y_true = torch.cat(
            (torch.ones(pos_preds.shape[0]), torch.zeros(neg_preds.shape[0])), dim=0)
        return roc_auc_score(y_true, y_pred)


    def run(self) -> None:
        if not os.path.exists('./log/'):
            os.mkdir('./log/')
        if not os.path.exists('./state_dict/'):
            os.mkdir('./state_dict/')
        important_args = ['lr', 'batch_size', 'num_layers', 'hidden_channels', 
                            'mechanism', 'eps', 'propagation',  'niter', 'alpha',
                            'rmax', 'rrz', 'drop_prob', 'reg_lambda', 'm']
        result = {}
        for arg in self.opt:
            if arg in important_args:
                value = self.opt[arg]
                if isinstance(value, list):
                    result[arg] = ','.join(str(v) for v in value)
                else:
                    result[arg] = value

        max_val_eval_list = []
        test_eval_list = []
        result['repeats'] = self.opt['repeats']

        for i in range(self.opt['repeats']):
            # self.logger.info('repeat: {}'.format(i+1))
            print('repeat: {}'.format(i+1))
            self._load_data_init_model()
            self._reset_params()
            max_val_eval = self._train()
            max_val_eval_list.append(max_val_eval)
            # test
            model_path = './state_dict/' + self.opt['dataset_str'] + '_link_' + \
                        self.opt['mechanism'] + '_' + str(self.opt['eps']) + '.pkl'
            self.model.load_state_dict(torch.load(model_path))
            test_eval = self._evaluate('test')
            # self.logger.info('max valid eval: {:.1f}, test eval: {:.1f}'.format(100 * max_val_eval, 100 * test_eval))
            print('max valid eval: {:.1f}, test eval: {:.1f}'.format(100 * max_val_eval, 100 * test_eval))
            test_eval_list.append(test_eval)
            # self.logger.info('#' * 80)
            print('#' * 80)
        max_val_eval_avg = sum(max_val_eval_list) / self.opt['repeats']
        test_eval_avg = sum(test_eval_list) / self.opt['repeats']
        result['max valid eval'], result['test eval'] = str(max_val_eval_list), str(test_eval_list)
        result['max valid eval avg'], result['test eval avg'] = max_val_eval_avg, test_eval_avg
        # self.logger.info("max valid eval avg: {:.1f}".format(100 * max_val_eval_avg))
        print("max valid eval avg: {:.1f}".format(100 * max_val_eval_avg))
        # self.logger.info("test eval avg: {:.1f}".format(100 * test_eval_avg))
        print("test eval avg: {:.1f}".format(100 * test_eval_avg))
        # df_result = pd.DataFrame(data=result, index=[0])
        # file_path = './results/' + self.opt['dataset_str'] + '_link/' + \
        #     self.opt['mechanism'] + '_' + str(self.opt['eps']) + '.csv'
        # if os.path.exists(file_path):
        #     df_result.to_csv(file_path, mode='a', index=False, header=False, encoding='utf-8')
        # else:
        #     df_result.to_csv(file_path, index=False, encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Pytorch Link Prediction")
    parser.add_argument('--seed', type=int, default=20159, help='random seed')
    parser.add_argument('--device', type=int, default=0, help='device')
    parser.add_argument('--dataset_str', type=str, default='cora', help='dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=1000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_layers', type=int, default=3, help='number of layers')
    parser.add_argument('--hidden_channels', type=int, default=256, help='hidden dimensions')
    parser.add_argument('--mechanism', type=str, default='hds', help='perturbation mechanism')
    parser.add_argument('--eps', type=float, default=float('inf'), help='privacy budget')
    parser.add_argument('--patience', type=int, default=50, help='early stopping')
    parser.add_argument('--log_steps', type=int, default=1, help='log step')
    parser.add_argument('--repeats', type=int, default=1, help='repeats')
    parser.add_argument('--save', type=bool, default=True, help='save the best model')
    parser.add_argument('--propagation', type=str, default='push', help='propagation')
    parser.add_argument('--niter', type=int, default=10, help='power iteration steps')
    parser.add_argument('--alpha', type=float, default=0.2, help='decay factor')
    parser.add_argument('--rmax', type=float, default=5e-4, help='push threshold')
    parser.add_argument('--rrz', type=float, default=0.5, help='convolution coefficient')
    parser.add_argument('--drop_prob', type=float, default=0.5, help='dropout probability')
    parser.add_argument('--reg_lambda', type=float, default=0., help='regularization lambda')
    parser.add_argument('--m', type=int, default=5, help='sample dimensions')

    opt = parser.parse_args()
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device('cuda', opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        np.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        # torch.set_deterministic(True)
        # torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic = True

    cpu_num = 5
    os.environ['OMP_NUM_THREADS'] = str(cpu_num)
    os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
    os.environ['MKL_NUM_THREADS'] = str(cpu_num)
    os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
    os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
    torch.set_num_threads(cpu_num)

    # optimized_params = nni.get_next_parameter()
    opt = vars(opt)
    # opt.update(optimized_params)
    ins = Instructor(opt)
    ins.run()
