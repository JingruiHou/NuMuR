import csv
import pickle
from curses.ascii import isdigit

import numpy as np
import os
import logging
import copy
from datetime import datetime
import pandas as pd
import task_utils


def calculate_forgetting_retaining_performance(prediction_dict, forgetting_dict, qrels):
    retaining_predictions_dict = copy.deepcopy(prediction_dict)
    forgetting_query_removal_mrr = []
    forgetting_docum_removal_mrr = []
    for query, doc, substitutes in forgetting_dict['doc_removal_samples'] + forgetting_dict['common_samples']:
        ranked = sorted(prediction_dict[query], key=prediction_dict[query].get, reverse=True)
        position = ranked.index(doc)
        forgetting_docum_removal_mrr.append(1 / (position + 1))
        retaining_predictions_dict[query].pop(doc)
        for s_doc in substitutes:
            retaining_predictions_dict[query].pop(s_doc)

    for query, doc, substitutes in forgetting_dict['query_removal_samples'] + forgetting_dict['common_samples']:
        retaining_predictions_dict.pop(query)
        ranked = sorted(prediction_dict[query], key=prediction_dict[query].get, reverse=True)
        for i in range(min(len(ranked), 100)):
            if ranked[i] in qrels[query]:
                forgetting_query_removal_mrr.append(1 / (i + 1))
                break

    retaining_mrr = task_utils.calculate_mrr(qrels, retaining_predictions_dict, topx=10)
    return np.mean(forgetting_query_removal_mrr), np.mean(forgetting_docum_removal_mrr), retaining_mrr




def do_single_model_evaluation(task_cfg):
    if task_cfg['strategy'] == 'original':
        stage = 1
        model_name = f"{task_cfg['model']}"
    else:
        stage = 2
        model_name = f"{task_cfg['strategy']}_{task_cfg['ratio']}_{task_cfg['model']}"
    epoch = task_cfg['epoch']
    dev_prediction_path = os.path.join(task_cfg['prediction_output_path'],
                                       '{}_dev_{}_{}.csv'.format(model_name, stage, epoch))

    test_prediction_path = os.path.join(task_cfg['prediction_output_path'],
                                        '{}_trained_{}_{}.csv'.format(model_name, stage, epoch))

    if os.path.exists(test_prediction_path) :
        unlearn_prediction = task_utils.read_performance_csv_by_pandas(test_prediction_path)

        forgetting_query_mrr, forgetting_doc_mrr, retaining_mrr = calculate_forgetting_retaining_performance(unlearn_prediction,
                                                                                             task_cfg['unlearn_data'],
                                                                                             task_cfg['test_qrels'])
    else:
         forgetting_query_mrr, forgetting_doc_mrr, retaining_mrr = 'N/A', 'N/A', 'N/A'
    if os.path.exists(dev_prediction_path):
        prediction = task_utils.read_performance_csv_by_pandas(dev_prediction_path)
        dev_mrr = task_utils.calculate_mrr(task_cfg['dev_qrels'], prediction)
    else:
        dev_mrr = 'N/A'

    print(f"{model_name} {stage} epoch({epoch}): "
          f"query_removal_forgetting_mrr: {forgetting_query_mrr}, document_removal_forgetting_mrr: {forgetting_doc_mrr}, "
          f"retaining_mrr:{retaining_mrr}, dev_mrr: {dev_mrr}")

    if isinstance(forgetting_query_mrr, (int, float)) or  isinstance(dev_mrr, (int, float)):
        return task_cfg['dataset'], task_cfg['model'], task_cfg['strategy'], task_cfg['ratio'], epoch, \
               forgetting_query_mrr, forgetting_doc_mrr, retaining_mrr, dev_mrr
    else:
        return False



def do_evaluation(eval_cfg):
    num_epochs = 100
    meta_data = ['dataset', 'model', 'unlearn_strategy', 'forgetting_ratio', 'epoch',
                 'forgetting_query_mrr', 'forgetting_document_mrr', 'retaining_mrr', 'dev_mrr']
    result_dic = {key: [] for key in meta_data}

    trained_model_epoch = {
        'trec': dict(BERTCat=8, BERTdot=8, ColBERT=4, Parade=8),
        'marco': dict(BERTCat=3, BERTdot=3, ColBERT=3, Parade=3)
    }

    for dataset in eval_cfg['tested_datasets']:
        task_cfg = task_utils.loadyaml(f'config/data.{dataset}.yaml')
        dev_qrels = task_utils.read_qrels(task_cfg['dev_qrel_path'])
        test_qrels = task_utils.read_qrels(task_cfg['test_qrel_path'])
        task_cfg['dev_qrels'] = dev_qrels
        task_cfg['test_qrels'] = test_qrels
        task_cfg['trained_model_epoch'] = trained_model_epoch
        task_cfg['dataset'] = dataset
        with open(task_cfg['forgetting_substitute_samples_path'], 'rb') as f:
            unlearn_data_all = pickle.load(f)
            unlearn_data = unlearn_data_all[f'ratio_{eval_cfg["ratio"]}']
            task_cfg['unlearn_data'] = unlearn_data
            task_cfg['ratio'] = eval_cfg["ratio"]
            for strategy in eval_cfg['tested_strategies']:
                task_cfg['strategy'] = strategy
                for model in eval_cfg['tested_models']:
                    task_cfg['model'] = model

                    for x in range(0, num_epochs + 1):
                        task_cfg['epoch'] = x
                        eval_result = do_single_model_evaluation(task_cfg)
                        if eval_result:
                            for ind, key in enumerate(meta_data):
                                result_dic[key].append(eval_result[ind])

    pd.DataFrame(result_dic).to_csv(f'./results/Unlearn_result_output.csv')


if __name__ == '__main__':
    eval_models = ['BERTCat', 'ColBERT', 'BERTdot', 'Parade']
    eval_datasets = ['trec', 'marco']
    eval_cfg = dict(
        tested_models=eval_models,
        tested_strategies=['catastrophic', 'amnesiac', 'NegGrad', 'BadTeacher', 'SSD',
                           'CoCoL', 'CoCoL_V2'],
        tested_datasets=['marco', 'trec'],
        ratio=0.1
    )
    do_evaluation(eval_cfg)

