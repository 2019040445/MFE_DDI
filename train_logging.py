import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import nn
import os


class CrossEntropy(nn.Module):#loss= criterion(score, label)
    def forward(self, input, target):
        scores = torch.sigmoid(input)
        # print(scores)
        target_active = (target == 1).float()  # from -1/1 to 0/1
        # print(target_active)
        loss_terms = -(target_active * torch.log(scores) + (1 - target_active) * torch.log(1 - scores))
        # print(loss_terms.sum())
        # print(len(loss_terms))
        b=loss_terms.sum()/len(loss_terms)
        return b
LOSS_FUNCTIONS={
    'CrossEntropy':CrossEntropy()
}

SAVEDMODELS_DIR = 'savedmodels_best/'
class Globals: # container for all objects getting passed between log calls
    evaluate_called = False

g = Globals()


def all_evaluate(output,target):
    #print(target,output)
    lengh=len(output)
    scores = torch.sigmoid(output)
    # print(type(target),type(scores))
    # print(target.device,scores.device)
    target = target.cpu()
    scores = scores.cpu()
    auroc = roc_auc_score(target, scores)
    scores = np.array(scores).astype(float)
    sum_scores = np.sum(scores)
    ave_scores = sum_scores / lengh
    target = np.array(target).astype(int)

    Confusion_M = np.zeros((2, 2), dtype=float)  # (TN FP),(FN,TP)
    for i in range(lengh):
        if (scores[i] < ave_scores):
            scores[i] = 0
        else:
            scores[i] = 1
    scores = np.array(scores).astype(int)

    for i in range(lengh):
        if(target[i]==scores[i]):
            if(target[i]==1):
                Confusion_M[0][0] += 1#TP
            else:
                Confusion_M[1][1] += 1#TN
        else:
            if(target[i]==1):
                Confusion_M[0][1] += 1#FP
            else:
                Confusion_M[1][0]  +=1#FN

    Confusion_M = np.array(Confusion_M, dtype=float)
    print('Confusion_M:', Confusion_M)
    accuracy = (Confusion_M[1][1] + Confusion_M[0][0]) / (
            Confusion_M[0][0] + Confusion_M[1][1] + Confusion_M[0][1] + Confusion_M[1][0])

    recall = Confusion_M[0][0] / (Confusion_M[0][0] + Confusion_M[0][1])
    precision = Confusion_M[0][0] / (Confusion_M[0][0] + Confusion_M[1][0])
    F1=2*precision*recall
    h=precision+recall
    F1=F1/h
    sum = 0.0
    for i in range(lengh):
        sum = sum + (target[i] - scores[i]) * (target[i] - scores[i])

    return F1,accuracy,recall,precision,auroc


SCORE_FUNCTIONS = {
    'All':all_evaluate}


def feed_net(net,dataloader, criterion):

    batch_outputs = []
    batch_losses = []
    batch_targets = []
    for i_batch, batch in enumerate(dataloader):

        d_node, d_attn_bias, d_spatial_pos, d_in_degree, d_out_degree, d_edge_input, p_node, p_attn_bias, p_spatial_pos, p_in_degree, p_out_degree, p_edge_input, label, (adj_1, nd_1, ed_1),(adj_2, nd_2, ed_2),d1,d2,mask_1,mask_2 = batch
        output = net(d_node.cuda(), d_attn_bias.cuda(), d_spatial_pos.cuda(), d_in_degree.cuda(), d_out_degree.cuda(), d_edge_input.cuda(),
                          p_node.cuda(), p_attn_bias.cuda(), p_spatial_pos.cuda(), p_in_degree.cuda(), p_out_degree.cuda(), p_edge_input.cuda(),
                          adj_1.cuda(), nd_1.cuda(), ed_1.cuda(),
                          adj_2.cuda(), nd_2.cuda(), ed_2.cuda(),
                          d1.cuda(),d2.cuda(),mask_1.cuda(),mask_2.cuda())
        label = torch.from_numpy(np.array(label)).long().cuda()
        loss = criterion(output, label)
        batch_outputs.append(output)
        batch_losses.append(loss.item())
        batch_targets.append(label)

    outputs = torch.cat(batch_outputs)

    loss = np.mean(batch_losses)#average loss
    targets = torch.cat(batch_targets)
    return outputs, loss, targets


def evaluate_net(net, train_dataloader, validation_dataloader, test_dataloader, criterion, args):
    global g
    if not g.evaluate_called:
        g.evaluate_called = True
        g.best_mean_train_score, g.best_mean_validation_score, g.best_mean_test_score = 0, 0, 0
        # g.train_subset_loader = train_dataloader

    # train_output, train_loss, train_target = feed_net(net,g.train_subset_loader, criterion)
    validation_output, validation_loss, validation_target = feed_net(net,validation_dataloader, criterion)
    test_output, test_loss, test_target = feed_net(net,test_dataloader, criterion)

    # train_scores = SCORE_FUNCTIONS[args.score](train_output, train_target)
    validation_scores = SCORE_FUNCTIONS[args.score](validation_output, validation_target)
    test_scores = SCORE_FUNCTIONS[args.score](test_output, test_target)
    new_best_model_found = validation_scores[4] > g.best_mean_validation_score

    if new_best_model_found:
        # g.best_mean_train_score = train_scores[4]
        g.best_mean_validation_score = validation_scores[4]
        g.best_mean_test_score = test_scores[4]

        if args.savemodel:
            path = SAVEDMODELS_DIR + type(net).__name__
            print(path)
            torch.save(net, path)

    if(args.score=='All'):
     return{
        'F1 score':{ 'validation': validation_scores[0], 'test': test_scores[0]},
        'Accuracy':{ 'validation': validation_scores[1], 'test': test_scores[1]},
        'Recall':{'validation': validation_scores[2], 'test': test_scores[2]},
        'Precision':{ 'validation': validation_scores[3], 'test': test_scores[3]},
        'auroc':{'validation': validation_scores[4], 'test': test_scores[4]},
        'best mean':{'validation': g.best_mean_validation_score, 'test': g.best_mean_test_score}
        
        }


def get_run_info(net, args):
    return {
        'net': type(net).__name__,
        'args': ', '.join([str(k) + ': ' + str(v) for k, v in vars(args).items()]),
        'modules': {name: str(module) for name, module in  net._modules.items()}
    }


def less_log(net, train_dataloader, validation_dataloader, test_dataloader, criterion, epoch, args):

    scalars = evaluate_net(net, train_dataloader, validation_dataloader, test_dataloader, criterion, args)
    global g
    if not g.evaluate_called:
        run_info = get_run_info(net, args)
        print('net: ' + run_info['net'])
        print('args: {' + run_info['args'] + '}')
        print('****** MODULES: ******')
        for name, description in run_info['modules'].items():
            print(name + ': ' + description)
        print('**********************')

    if(args.score=='All'):
        # print('epoch {}, F1 score :training mean: {}, validation mean: {}, testing mean: {}'.format(
        print('epoch {}, F1 score: validation mean: {}, testing mean: {}'.format(
         epoch+1,
        #  scalars['F1 score']['train'],
         scalars['F1 score']['validation'],
            scalars['F1 score']['test'])
            )
        print('          ACC: validation mean: {}, testing mean: {}'.format(
        # scalars['Accuracy']['train'],
        scalars['Accuracy']['validation'],
        scalars['Accuracy']['test']))

        print('          Precision: validation mean: {}, testing mean: {}'.format(
            # scalars['Precision']['train'],
            scalars['Precision']['validation'],
            scalars['Precision']['test']))
        print('          Recall: validation mean: {}, testing mean: {}'.format(
            # scalars['Recall']['train'],
            scalars['Recall']['validation'],
            scalars['Recall']['test']))
        print('          AUROC: validation mean: {}, testing mean: {}'.format(
            # scalars['auroc']['train'],
            scalars['auroc']['validation'],
            scalars['auroc']['test']))

        print('          best auroc: validation mean: {}, testing mean: {}'.format(
            #  scalars['best mean']['train'],
             scalars['best mean']['validation'],
                scalars['best mean']['test']))

        # print('          loss:training mean: {}'.format(
        #     scalars['loss']['train'],))

    else:
         mean_score_key = 'mean {}'.format(args.score)
         print('epoch {}, training mean {}: {}, validation mean {}: {}, testing mean {}:{}'.format(
                epoch + 1,
                args.score, scalars[mean_score_key]['train'],
                args.score, scalars[mean_score_key]['validation'],
                args.score, scalars[mean_score_key]['test']),
         )

LOG = {
    'less':less_log}