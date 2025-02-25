import copy
from sklearn import datasets
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from torch import nn
from torch.autograd import Variable
from torch.utils import data
from models import Molormer, MultiLevelDDI
from collator import *
torch.manual_seed(2)
np.random.seed(3)
from configs import Model_config
from dataset import Dataset
import os
import argparse
from train_logging import LOG,LOSS_FUNCTIONS

os.environ['CUDA_VISIBLE_DEVICES'] = '2'


use_cuda = torch.cuda.is_available()
# device = torch.device("cuda:2" if use_cuda else "cpu")
# device = 'cpu'


common_args_parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter, add_help=False)
common_args_parser.add_argument('--loss', type=str, default='CrossEntropy', choices=[k for k, v in LOSS_FUNCTIONS.items()])
common_args_parser.add_argument('--score', type=str, default='All', help='roc-auc or MSE or All')
common_args_parser.add_argument('--savemodel', action='store_true', default=True, help='Saves model with highest validation score')
common_args_parser.add_argument('--logging', type=str, default='less')



args = None

def main():
    config = Model_config()
    print(config)
    global args
    args = common_args_parser.parse_args()
    args_dict = vars(args)

    loss_history = []

    model = MultiLevelDDI(**config)
    model = model.cuda()

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model, dim=0)

    params = {'batch_size': config['batch_size'],
              'shuffle': True,
              'num_workers': config['num_workers'],
              'drop_last': True,
              'collate_fn': collator}

    train_data = pd.read_csv('dataset/train.csv')
    val_data = pd.read_csv('dataset/val.csv')
    test_data = pd.read_csv('dataset/test.csv')

    training_set = Dataset(train_data.index.values, train_data.label.values, train_data)

    training_generator = data.DataLoader(training_set, **params)

    validation_set = Dataset(val_data.index.values, val_data.label.values, val_data)


    validation_generator = data.DataLoader(validation_set, **params)

    testing_set = Dataset(test_data.index.values, test_data.label.values, test_data)
    testing_generator = data.DataLoader(testing_set, **params)

    max_auc = 0
    model_max = copy.deepcopy(model)

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = LOSS_FUNCTIONS[args.loss].cuda()



    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    for epo in range(config['epochs']):
        model.train()
        for i, (d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input,
                p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input,
                label, (adj_1, nd_1, ed_1),(adj_2, nd_2, ed_2),d1,d2,mask_1,mask_2) in enumerate(training_generator):

            opt.zero_grad()
            score = model(d_node.cuda(), d_attn_bias.cuda(), d_spatial_pos.cuda(), d_in_degree.cuda(), d_out_degree.cuda(), d_edge_input.cuda(),
                          p_node.cuda(), p_attn_bias.cuda(), p_spatial_pos.cuda(), p_in_degree.cuda(), p_out_degree.cuda(), p_edge_input.cuda(),
                          adj_1.cuda(), nd_1.cuda(), ed_1.cuda(),
                          adj_2.cuda(), nd_2.cuda(), ed_2.cuda(),
                          d1.cuda(),d2.cuda(),mask_1.cuda(),mask_2.cuda()) # torch tensor
            # print(score.shape,score)

            label = torch.from_numpy(np.array(label)).long().cuda() # torch tensor
            loss= criterion(score, label)
            loss_history.append(loss)

            loss.backward()
            torch.nn.utils.clip_grad_value_(model.parameters(), 5.0)
            # 用于裁剪梯度，梯度裁剪是一种正则化技术，用于防止在训练深度学习模型时发生梯度爆炸
            opt.step()
            # scheduler.step()

            if (i % 300 == 0):
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                    loss.cpu().detach().numpy()))

        with torch.set_grad_enabled(False):
            model.eval()
            LOG[args.logging](
                model, training_generator, validation_generator, testing_generator, criterion, epo, args)

        torch.cuda.empty_cache()


    return model_max, loss_history



if __name__=='__main__':
    main()
