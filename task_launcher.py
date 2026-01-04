import task_utils
from unranking_task import UnRankingTask
import torch


def run_unlearn_exp(unlearn_method):
    retraining_epoch = dict(trec=100, marco=100)
    for data_name in data_names:
        for model_cfg_path in model_cfg_paths:
            device_count=torch.cuda.device_count()
            config = dict(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            config['device_count'] = device_count
            model_config = task_utils.loadyaml(f'{model_cfg_path}')
            data_config = task_utils.loadyaml(f'./config/data.{data_name}.yaml')
            config.update(data_config)
            config.update(model_config)
            config['forgetting_ratio'] = 0.1
            config['num_workers'] = 0
            config['task_prefix'] = unlearn_method
            config['unlearn_epoch'] = retraining_epoch[data_name]
            config['num_neg_docs_used'] = 100
            config['amnesiac_num'] = 100
            config['trained_model_epoch'] = trained_model_epoch[data_name][config['model_name']]

            if unlearn_method == 'amnesiac':
                task = UnRankingTask(config)
                task.do_amnesiac_retrain()
            elif unlearn_method == 'retrain':
                config['trained_model_epoch'] = 0
                task = UnRankingTask(config)
                task.do_retrain()
            elif unlearn_method == 'catastrophic':
                task = UnRankingTask(config)
                task.do_retrain()
            elif unlearn_method == 'NegGrad':
                task = UnRankingTask(config)
                task.do_neg_gradient_retrain()
            elif unlearn_method == 'BadTeacher':
                config['teacher_model'] = True
                config['bad_teacher_model'] = True
                task = UnRankingTask(config)
                task.do_bad_teacher_retrain()
            elif unlearn_method == 'SSD':
                config['unlearn_epoch'] = 5
                config['num_neg_docs_used'] = 10
                task = UnRankingTask(config)
                task.do_SSD_retrain()
            elif unlearn_method == 'CoCoL':
                config['batch_size'] = 32
                config['teacher_model'] = True
                config['distillation_margin'] = 0.0
                config['distillation_ratio'] = 1
                config['num_neg_docs_used'] = 5
                config['unlearn_epoch'] = 5
                task = UnRankingTask(config)
                task.do_CoCoL_retrain()
            elif unlearn_method == 'CoCoL_V2':
                config['batch_size'] = 32
                config['teacher_model'] = True
                config['distillation_margin'] = 0.0
                config['distillation_ratio'] = 1
                config['num_neg_docs_used'] = 5
                config['unlearn_epoch'] = 5
                config['CoCoL_learning_from_worst'] = True
                task = UnRankingTask(config)
                task.do_CoCoL_retrain()
            else:
                pass
            for epoch in range(1, config['unlearn_epoch'] + 1):
                task.do_test(test='trained', epoch=epoch)
                # task.do_test(test='dev', epoch=epoch)


if __name__ == '__main__':
    torch.cuda.set_device(0)
    data_names = ['marco', 'trec']
    model_cfg_paths = ['config/ranker.bertcat.yaml', 'config/ranker.bertdot.yaml', 'config/ranker.colbert.yaml',
                        'config/ranker.parade.yaml']
    trained_model_epoch = {
        'trec': dict(BERTCat=8, BERTdot=8, ColBERT=4, Parade=8),
        'marco': dict(BERTCat=3, BERTdot=3, ColBERT=3, Parade=3)
    }
    unlearn_methods = ['amnesiac', 'retrain', 'catastrophic', 'NegGrad', 'BadTeacher', 'SSD', 'CoCoL', 'CoCoL_V2']
    for unlearn_method in unlearn_methods:
        run_unlearn_exp(unlearn_method)
