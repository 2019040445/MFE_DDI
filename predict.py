import numpy as np
import pandas as pd
import torch
from torch.utils import data
from tqdm import tqdm
torch.manual_seed(2)
np.random.seed(3)
from argparse import ArgumentParser
from dataset import Dataset
from torch.autograd import Variable
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, roc_curve, precision_recall_curve, auc, roc_auc_score
from collator import collator


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")

parser = ArgumentParser(description='Molormer Prediction.')
parser.add_argument('-b', '--batch-size', default=16, type=int,metavar='N')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N')


def test(data_generator, model):
    y_pred_bi = []
    y_pred = []
    y_label = []
    model.eval()
    loss_accumulate = 0.0
    count = 0.0

    for i, (d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input,
                p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input,
                label, (adj_1, nd_1, ed_1),(adj_2, nd_2, ed_2),d1,d2,mask_1,mask_2) in enumerate(tqdm(data_generator)):

        score = model(d_node.cuda(), d_attn_bias.cuda(), d_spatial_pos.cuda(), d_in_degree.cuda(), d_out_degree.cuda(), d_edge_input.cuda(),
                    p_node.cuda(), p_attn_bias.cuda(), p_spatial_pos.cuda(), p_in_degree.cuda(), p_out_degree.cuda(), p_edge_input.cuda(),
                    adj_1.cuda(), nd_1.cuda(), ed_1.cuda(),
                    adj_2.cuda(), nd_2.cuda(), ed_2.cuda(),
                    d1.cuda(),d2.cuda(),mask_1.cuda(),mask_2.cuda())
        
        label = torch.from_numpy(np.array(label)).long().cuda()
        score = torch.sigmoid(score)

        y_pred = score.cpu()

        label = label.cpu().numpy()
        y_pred = y_pred.numpy()

        
        y_label = y_label + label.tolist()
        y_pred_bi = y_pred_bi + y_pred.tolist()
    y_label = [item[0] for item in y_label]
    y_pred_bi = [item[0] for item in y_pred_bi]
    y_pred_bi = [1 if prob >= 0.4 else 0 for prob in y_pred_bi]
    accuracy = accuracy_score(y_label, y_pred_bi)
    precision = precision_score(y_label, y_pred_bi)
    recall = recall_score(y_label, y_pred_bi)
    f1 = f1_score(y_label, y_pred_bi)
    auroc = roc_auc_score(y_label, y_pred_bi)
    print(accuracy, precision, recall, f1, auroc) 

def main():
    args = parser.parse_args()

    model = torch.load('savedmodels/MultiLevelDDI')

    model = model.to(device)


    params = {'batch_size': args.batch_size,
              'shuffle': False,
              'num_workers': args.workers,
              'drop_last': False,
              'collate_fn': collator}
    # df_test = pd.read_csv('dataset/search_DB00852.csv')
    df_test = pd.read_csv('dataset/test.csv')


    testing_set = Dataset(df_test.index.values, df_test.label.values, df_test)
    testing_generator = data.DataLoader(testing_set, **params)

    print('--- Go for Predicting ---')
    with torch.set_grad_enabled(False):
        test(testing_generator, model)

    torch.cuda.empty_cache()

main()
print("Done!")
