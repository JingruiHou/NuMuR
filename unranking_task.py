import random

import pandas as pd
import pickle
import task_utils
import models
import torch
import copy
from tqdm import tqdm
from torch.utils.data import DataLoader
from train import train_ranking_model, test_ranking_model, unlearn_neg_teacher_hinge, unlearn_bad_good_teacher, cocol_distil_train
from ranking_dataset import TSVTextDataset, SimpleStringDataset
import torch.nn as nn
import os


class UnRankingTask:
    def __init__(self, config):
        self.config = config
        # loading original trained model
        self.config['step'] = 1
        self.config['stage'] = 1
        self.config['model_save_name'] = self.config['model_name']
        # retain from zero
        self.model = self.load_model(epoch=self.config['trained_model_epoch'], freezing_parameters=True)
        if self.config.get('teacher_model') and self.config['teacher_model']:
            self.teacher_model = self.load_model(epoch=self.config['trained_model_epoch'], freezing_parameters=False)
        if self.config.get('bad_teacher_model') and self.config['bad_teacher_model']:
            self.bad_teacher = self.load_model(epoch=0, freezing_parameters=False)

        # do training....
        self.config['stage'] = 2
        self.config['start_epoch'] = 0
        self.config['epochs'] = self.config['unlearn_epoch']
        self.config[
            'model_save_name'] = f"{self.config['task_prefix']}_{self.config['forgetting_ratio']}_{self.config['model_name']}"

    def load_model(self, load_state=True, epoch=1, freezing_parameters=False):
        model = models.__dict__[self.config['model_type']].__dict__[self.config['model_name']].from_config(
            self.config)
        model = model.to(self.config['device'])
        if load_state:
            saved_model_path_stage = self.config['model_save_path'].format(self.config['model_save_name'],
                                                                           self.config['stage'], epoch)
            print(saved_model_path_stage)
            if os.path.exists(saved_model_path_stage):
                model_state = torch.load(saved_model_path_stage, map_location=lambda storage, loc: storage)
                model.load_state_dict(model_state)
        if freezing_parameters and self.config.get('parameter_name_filter'):
            print(self.config.get('parameter_name_filter'))
            for name, param in model.named_parameters():
                param.requires_grad = False
                for _filter in self.config['parameter_name_filter']:
                    if _filter in name:
                        param.requires_grad = True
                        break
        return model

    def do_test(self, test='test', epoch=1):
        path = os.path.join(self.config['prediction_output_path'],
                            '{}_{}_{}_{}.csv'.format(self.config['model_save_name'], test, self.config['stage'], epoch))
        if os.path.exists(path):
            print(f'{path} already exits, pass...')
            return
        saved_model_path_stage = self.config['model_save_path'].format(self.config['model_save_name'],
                                                                       self.config['stage'], epoch)
        if os.path.exists(saved_model_path_stage):
            model = self.load_model(load_state=True, epoch=epoch)
            if test == 'dev':
                dataset = TSVTextDataset(self.config['dev_data_path'])
            if test == 'trained':
                dataset = TSVTextDataset(self.config['test_data_path'])

            data_loader = DataLoader(dataset, batch_size=self.config['prediction_batch_size'], shuffle=False,
                                     num_workers=self.config['num_workers'])
            q_ids, d_ids, total_scores = test_ranking_model(data_loader, model, self.config)
            pd.DataFrame({'q_id': q_ids, 'd_id': d_ids, 'rel': total_scores}).to_csv(
                os.path.join(self.config['prediction_output_path'],
                             '{}_{}_{}_{}.csv'.format(self.config['model_save_name'], test, self.config['stage'], epoch)))

    def do_retrain(self):
        retain_dataset = UnRankingDataConstructor(self.config).build_retrain_dataset()
        data_loader = DataLoader(retain_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                 num_workers=self.config['num_workers'])
        train_ranking_model(data_loader, self.model, self.config)

    def do_amnesiac_retrain(self):
        amnesiac_dataset = UnRankingDataConstructor(self.config).build_amnesiac_dataset()
        data_loader = DataLoader(amnesiac_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                 num_workers=self.config['num_workers'])
        train_ranking_model(data_loader, self.model, self.config)

    def do_retrain_correction(self):
        correction_dataset = UnRankingDataConstructor(self.config).build_correction_dataset()
        data_loader = DataLoader(correction_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                 num_workers=self.config['num_workers'])
        unlearn_neg_teacher_hinge(data_loader, self.teacher_model, self.model, self.config)

    def do_neg_gradient_retrain(self):
        forget_dataset, _ = UnRankingDataConstructor(
            self.config).build_split_forget_substitute_dataset()
        forget_loader = DataLoader(forget_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                   num_workers=self.config['num_workers'])
        # substitute_loader = DataLoader(substitute_dataset, batch_size=self.config['batch_size'], shuffle=True,
        #                                num_workers=self.config['num_workers'])
        self.config['epochs'] = 1
        for x in range(self.config['unlearn_epoch']):
            self.config['start_epoch'] = x
            self.config['neg_GRAD'] = True
            self.model = train_ranking_model(forget_loader, self.model, self.config)
            # self.config['neg_GRAD'] = False
            # self.model = train_ranking_model(substitute_loader, self.model, self.config)

    def do_bad_teacher_retrain(self):
        pos_pair_dataset, retain_dataset = UnRankingDataConstructor(self.config).build_bad_teacher_dataset()
        bat_teacher_loader = DataLoader(pos_pair_dataset, batch_size=self.config['batch_size'], shuffle=True,
                                        num_workers=self.config['num_workers'])
        # substitute_loader = DataLoader(retain_dataset, batch_size=self.config['batch_size'], shuffle=True,
        #                                num_workers=self.config['num_workers'])
        self.config['epochs'] = 1
        for x in range(self.config['unlearn_epoch']):
            self.config['start_epoch'] = x
            self.model = unlearn_bad_good_teacher(bat_teacher_loader, self.bad_teacher, self.teacher_model, self.model,
                                                  self.config)
            # self.model = train_ranking_model(substitute_loader, self.model, self.config)

    def do_SSD_retrain(self):
        import ssd
        forget_data, substitute_data, retain_data = UnRankingDataConstructor(self.config).build_ssd_dataset()
        forgetting_loader = DataLoader(forget_data, batch_size=self.config['batch_size'], shuffle=True,
                                       num_workers=self.config['num_workers'])
        retrain_loader = DataLoader(retain_data, batch_size=self.config['batch_size'], shuffle=True,
                                    num_workers=self.config['num_workers'])
        # substitute_loader = DataLoader(substitute_data, batch_size=self.config['batch_size'], shuffle=True,
        #                                num_workers=self.config['num_workers'])
        SSD_parameters = {
            "lower_bound": 1,
            "exponent": 1,
            "magnitude_diff": None,
            "min_layer": -1,
            "max_layer": -1,
            "forget_threshold": 1,
            "dampening_constant": 1,
            "selection_weighting": 10,
        }
        self.config['epochs'] = 1
        for x in range(self.config['unlearn_epoch']):
            self.config['start_epoch'] = x
            optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
            ssd_instance = ssd.ParameterPerturber(self.config, self.model, optimizer, self.config['device'],
                                                  SSD_parameters)
            original_importances = ssd_instance.calc_ranking_importance(retrain_loader, dl_name='retaining')
            sample_importances = ssd_instance.calc_ranking_importance(forgetting_loader, dl_name='forgetting')
            self.model = ssd_instance.modify_weight(original_importances, sample_importances)
            # self.model = train_ranking_model(substitute_loader, self.model, self.config)

    def do_CoCoL_retrain(self):
        forget_entangled_set, disjoint_set, substitute_set = UnRankingDataConstructor(self.config).build_cocol_dataset()
        forgetting_loader = DataLoader(forget_entangled_set, batch_size=self.config['batch_size'], shuffle=True,
                                       num_workers=self.config['num_workers'])
        disjoint_loader = DataLoader(disjoint_set, batch_size=self.config['batch_size'], shuffle=True,
                                    num_workers=self.config['num_workers'])
        # substitute_loader = DataLoader(substitute_set, batch_size=self.config['batch_size'], shuffle=True,
        #                                num_workers=self.config['num_workers'])
        if self.config.get('reload_epoch'):
            self.model = self.load_model(epoch=self.config['reload_epoch'])
            start_epoch = self.config['reload_epoch']
        else:
            start_epoch = 0
        self.config['epochs'] = 1
        for x in range(self.config['unlearn_epoch']):
            self.config['start_epoch'] = start_epoch + x
            self.model = cocol_distil_train(forgetting_loader, disjoint_loader, self.teacher_model, self.model, self.config)
            # self.model = train_ranking_model(substitute_loader, self.model, self.config)
            self.config['model_save_step'] = 2


class UnRankingDataConstructor:
    def __init__(self, config):
        self.config = config

    def build_qid_did_map(self):
        dataset = TSVTextDataset(self.config['test_data_path']).get_all_samples()
        # dataset index  qid, did, line_number
        qid_did_index = {(sample[0], sample[1]): i for i, sample in enumerate(dataset)}
        # qid_did maps in dataset
        qid_did_map = {}
        for qid_did, ind in qid_did_index.items():
            qid, did = qid_did
            if qid not in qid_did_map:
                qid_did_map[qid] = []
            qid_did_map[qid].append(did)
        return qid_did_map, qid_did_index, dataset

    def get_unlearn_info(self):
        with open(self.config['forgetting_substitute_samples_path'], 'rb') as f:
            data = pickle.load(f)
        forgetting_ind = self.config['forgetting_ratio']
        correcting_samples = data[f'ratio_{forgetting_ind}']
        qrels = task_utils.read_qrels(self.config['test_qrel_path'])
        qrels_retain = copy.deepcopy(qrels)
        qid_did_map, qid_did_index, dataset = self.build_qid_did_map()
        qid_did_map_retain = copy.deepcopy(qid_did_map)
        correcting_qrels = {}
        for sample_type, samples in correcting_samples.items():
            for sample in samples:
                q, d, sb = sample
                if q not in correcting_qrels:
                    correcting_qrels[q] = []
                correcting_qrels[q].append({'forget': d, 'substitute': sb})

        for query, doc, substitute in correcting_samples['query_removal_samples'] + correcting_samples[
            'common_samples']:
            qid_did_map_retain.pop(query)
            qrels_retain.pop(query)
        for query, doc, substitutes in correcting_samples['doc_removal_samples']:
            if query in qid_did_map_retain:
                while doc in qid_did_map_retain[query]:
                    qid_did_map_retain[query].remove(doc)
                while doc in substitutes:
                    qid_did_map_retain[query].remove(doc)
            if query in qrels_retain:
                while doc in qrels_retain[query]:
                    qrels_retain[query].remove(doc)
        return dict(dataset=dataset, qrels=qrels, qrels_retain=qrels_retain,
                    qid_did_map=qid_did_map, qid_did_index=qid_did_index,
                    qid_did_map_retain=qid_did_map_retain,
                    correcting_qrels=correcting_qrels,
                    correcting_samples=correcting_samples)

    def build_forget_substitute_dataset(self, unlearn_info):
        dataset = unlearn_info['dataset']
        all_samples = []
        for sample_type, samples in unlearn_info['correcting_samples'].items():
            all_samples += samples
        forget_queries = []
        forget_pos_documents = []
        forget_neg_documents = []
        substitute_queries = []
        substitute_pos_documents = []
        substitute_neg_documents = []
        # sample:(qid, did)
        for sample in tqdm(all_samples):
            sample_q_id, sample_pos_d_ID, sample_subtitute_ids = sample[0], sample[1], sample[2]
            assert sample_pos_d_ID in unlearn_info['qrels'][sample_q_id]
            assert sample_pos_d_ID in unlearn_info['qid_did_map'][sample_q_id]
            sample_position = unlearn_info['qid_did_index'][(sample_q_id, sample_pos_d_ID)]
            q_text = dataset[sample_position][2]
            pos_text = dataset[sample_position][3]
            q_neg_samples = [(sample_q_id, s) for s in unlearn_info['qid_did_map'][sample_q_id]
                             if s not in unlearn_info['qrels'][sample_q_id] and s not in sample_subtitute_ids]
            q_neg_samples_sampling = task_utils.random_sample(q_neg_samples, self.config['num_neg_docs_used'],
                                                              _seed=self.config['seed'])
            neg_texts = [dataset[unlearn_info['qid_did_index'][q_d]][3] for q_d in q_neg_samples_sampling]
            forget_queries += [q_text for _ in range(len(neg_texts))]
            forget_pos_documents += [pos_text for _ in range(len(neg_texts))]
            forget_neg_documents += neg_texts

            for substitute_id in sample_subtitute_ids:
                sample_position = unlearn_info['qid_did_index'][(sample_q_id, substitute_id)]
                substitute_txt = dataset[sample_position][3]
                substitute_pos_documents += [substitute_txt for _ in range(len(neg_texts))]
                substitute_queries += [q_text for _ in range(len(neg_texts))]
                substitute_neg_documents += neg_texts
        return (forget_queries, forget_pos_documents, forget_neg_documents), \
               (substitute_queries, substitute_pos_documents, substitute_neg_documents)

    def build_retain_dataset(self, unlearn_info, merge_neg_docs=True, retain_ratio=None):
        qid_did_map_retain = unlearn_info['qid_did_map_retain']
        qrels_retain = unlearn_info['qrels_retain']
        qid_did_index = unlearn_info['qid_did_index']
        qrels = unlearn_info['qrels']
        dataset = unlearn_info['dataset']
        retain_queries, retain_pos_docs, retain_neg_docs = [], [], []
        if retain_ratio and retain_ratio < 1:
            sampled_queries_len = int(retain_ratio*len(list(qid_did_map_retain.keys())))
            if sampled_queries_len > 0:
                random.seed(self.config['seed'])
                sampled_keys = random.sample(list(qid_did_map_retain.keys()), sampled_queries_len)
                qid_did_map_retain = {key: qid_did_map_retain[key] for key in sampled_keys}
        for query, docs in tqdm(qid_did_map_retain.items(), desc='building retain_dataset...'):
            pos_docs = qrels_retain[query]
            neg_docs = [doc for doc in docs if doc not in qrels[query]]
            selected_neg_docs = task_utils.random_sample(neg_docs, self.config['num_neg_docs_used'],
                                                         _seed=self.config['seed'])
            for pos in pos_docs:
                idx = qid_did_index[(query, pos)]
                _qid, _did, _q_txt, pos_txt = dataset[idx]
                neg_txts = []
                for neg in selected_neg_docs:
                    n_idx = qid_did_index[(query, neg)]
                    _qid, _did, _, neg_txt = dataset[n_idx]
                    neg_txts.append(neg_txt)
                if merge_neg_docs:
                    retain_queries.append(_q_txt)
                    retain_pos_docs.append(pos_txt)
                    retain_neg_docs.append(neg_txts)
                else:
                    retain_queries += [_q_txt] * len(neg_txts)
                    retain_pos_docs += [pos_txt] * len(neg_txts)
                    retain_neg_docs += neg_txts
        return retain_queries, retain_pos_docs, retain_neg_docs

    def build_amnesiac_dataset(self):
        unlearn_info = self.get_unlearn_info()
        self.config['num_neg_docs_used'] = self.config['amnesiac_num']
        forget_data, substitute_data = self.build_forget_substitute_dataset(unlearn_info)
        forget_queries, forget_pos_documents, forget_neg_documents = forget_data
        # substitute_queries, substitute_pos_documents, substitute_neg_documents = substitute_data
        amnesiac_dataset = SimpleStringDataset(forget_queries,
                                               forget_neg_documents,
                                               forget_pos_documents)
        return amnesiac_dataset

    def build_split_forget_substitute_dataset(self):
        unlearn_info = self.get_unlearn_info()
        if not self.config.get('amnesiac_num'):
            self.config['amnesiac_num'] = 100
        self.config['num_neg_docs_used'] = self.config['amnesiac_num']
        forget_data, substitute_data = self.build_forget_substitute_dataset(unlearn_info)
        forget_dataset = SimpleStringDataset(*forget_data)
        substitute_dataset = SimpleStringDataset(*substitute_data)
        return forget_dataset, substitute_dataset

    def build_bad_teacher_dataset(self):
        unlearn_info = self.get_unlearn_info()
        data_retain = unlearn_info['qrels_retain']
        qid_did_index = unlearn_info['qid_did_index']
        dataset = unlearn_info['dataset']
        correcting_qrels = unlearn_info['correcting_qrels']
        data_forget = {}
        data_substitute = {}
        for query, data in correcting_qrels.items():
            data_forget[query] = [pair['forget'] for pair in data]
            data_substitute[query] = [pair['substitute'][0] for pair in data]

        _queries = []
        _pos_docs = []
        _labels = []

        def _processing_docs(docs, _label):
            for pos in docs:
                idx = qid_did_index[(query, pos)]
                _qid, _did, _q_txt, pos_txt = dataset[idx]
                _queries.append(_q_txt)
                _pos_docs.append(pos_txt)
                _labels.append(_label)

        for query, docs in tqdm(data_retain.items(), desc='building retain_dataset...'):
            _processing_docs(docs, 1)

        for query, docs in tqdm(data_forget.items(), desc='building forget_dataset...'):
            _processing_docs(docs, 0)

        for query, docs in tqdm(data_substitute.items(), desc='building substitute_dataset...'):
            _processing_docs(docs, 1)
        pos_pair_dataset = SimpleStringDataset(_queries, _pos_docs, _labels)

        if not self.config.get('num_neg_docs_used'):
            self.config['num_neg_docs_used'] = 100
        _, substitute_data = self.build_forget_substitute_dataset(unlearn_info)
        substitute_queries, substitute_pos_documents, substitute_neg_documents = substitute_data
        retrain_dataset = SimpleStringDataset(substitute_queries, substitute_pos_documents, substitute_neg_documents)
        return pos_pair_dataset, retrain_dataset

    def build_correction_dataset(self):
        unlearn_info = self.get_unlearn_info()
        qid_did_index = unlearn_info['qid_did_index']
        qrels = unlearn_info['qrels']
        dataset = unlearn_info['dataset']
        correcting_qrels = unlearn_info['correcting_qrels']
        qid_did_map = unlearn_info['qid_did_map']

        _queries, _pos_docs, _neg_docs = self.build_retain_dataset(unlearn_info, retain_ratio=self.config.get('retain_ratio'))
        _labels = [1 for x in range(len(_queries))]
        _subs_docs = _pos_docs[:]

        for q, ds in correcting_qrels.items():
            all_docs = qid_did_map[q]
            pos_docs = qrels[q]
            for pair in ds:
                forget_ind = pair['forget']
                substitute_ind = pair['substitute']
                neg_docs = [doc for doc in all_docs if doc not in pos_docs and doc not in substitute_ind]
                selected_neg_docs = task_utils.random_sample(neg_docs, self.config['num_neg_docs_used'],
                                                             _seed=self.config['seed'])
                f_idx = qid_did_index[(q, forget_ind)]
                s_idx = qid_did_index[(q, substitute_ind[0])]
                _, _, _q_txt, forget_txt = dataset[f_idx]
                _, _, _, substitute_txt = dataset[s_idx]
                neg_txts = []
                for neg in selected_neg_docs:
                    n_idx = qid_did_index[(q, neg)]
                    _qid, _did, q_txt, neg_txt = dataset[n_idx]
                    neg_txts.append(neg_txt)
                _queries.append(_q_txt)
                _pos_docs.append(forget_txt)
                _subs_docs.append(substitute_txt)
                _neg_docs.append(neg_txts)
                _labels.append(0)
        list_wise_dataset = SimpleStringDataset(_queries, _pos_docs, _neg_docs, _subs_docs, _labels)
        return list_wise_dataset

    def build_retrain_dataset(self):
        unlearn_info = self.get_unlearn_info()
        if not self.config.get('num_neg_docs_used'):
            self.config['num_neg_docs_used'] = 100
        _, substitute_data = self.build_forget_substitute_dataset(unlearn_info)
        retain_data = self.build_retain_dataset(unlearn_info, merge_neg_docs=False)
        retain_queries, retain_pos_documents, retain_neg_documents = retain_data
        substitute_queries, substitute_pos_documents, substitute_neg_documents = substitute_data
        retrain_dataset = SimpleStringDataset(retain_queries + substitute_queries,
                                              retain_pos_documents + substitute_pos_documents,
                                              retain_neg_documents + substitute_neg_documents)
        return retrain_dataset

    def build_ssd_dataset(self):
        unlearn_info = self.get_unlearn_info()
        if not self.config.get('num_neg_docs_used'):
            self.config['num_neg_docs_used'] = 100
        forget_data, substitute_data = self.build_forget_substitute_dataset(unlearn_info)
        retain_data = self.build_retain_dataset(unlearn_info, merge_neg_docs=False)
        retain_queries, retain_pos_documents, retain_neg_documents = retain_data
        forget_queries, forget_pos_documents, forget_neg_documents = forget_data
        substitute_queries, substitute_pos_documents, substitute_neg_documents = substitute_data
        forget_dataset = SimpleStringDataset(forget_queries, forget_pos_documents, forget_neg_documents)
        retain_dataset = SimpleStringDataset(retain_queries, retain_pos_documents, retain_neg_documents)
        substitute_dataset = SimpleStringDataset(substitute_queries, substitute_pos_documents, substitute_neg_documents)
        return forget_dataset, substitute_dataset, retain_dataset

    def build_cocol_dataset(self):
        unlearn_info = self.get_unlearn_info()
        qrels = unlearn_info['qrels']
        qid_did_index = unlearn_info['qid_did_index']
        dataset = unlearn_info['dataset']
        correcting_qrels = unlearn_info['correcting_qrels']
        qrels_retain = unlearn_info['qrels_retain']
        data_forget = {}
        data_substitute = {}

        for query, data in correcting_qrels.items():
            data_forget[query] = [pair['forget'] for pair in data]
            data_substitute[query] = [pair['substitute'][0] for pair in data]

        correcting_samples = unlearn_info['correcting_samples']
        doc_removal_samples = correcting_samples['doc_removal_samples']
        query_removal_samples = correcting_samples['query_removal_samples']
        common_samples = correcting_samples['common_samples']
        qrels_disjoint = copy.deepcopy(qrels_retain)

        forget_entangled_list = []
        for sample in doc_removal_samples:
            q, pos, _ = sample
            if q in qrels_retain:
                entangled_docs = qrels_retain[q]
                assert len(entangled_docs) > 0 and pos not in entangled_docs
                for _doc in entangled_docs:
                    forget_entangled_list.append((q, pos, q, _doc))
                    qrels_disjoint[q].remove(_doc)
            else:
                forget_entangled_list.append((q, pos, None, None))
        for sample in query_removal_samples:
            q, pos, _ = sample
            entangled_queries = [key_query for key_query in qrels_retain.keys() if pos in qrels_retain[key_query]]
            assert len(entangled_queries) > 0 and q not in entangled_queries
            for _q in entangled_queries:
                forget_entangled_list.append((q, pos, _q, pos))
                qrels_disjoint.pop(_q)
        for sample in common_samples:
            q, pos, _ = sample
            forget_entangled_list.append((q, pos, None, None))

        disjoint_q, disjoint_pos, disjoint_neg = [], [], []
        for query, docs in tqdm(qrels_disjoint.items(), desc='building disjoint_dataset...'):
            q_neg_samples = [_n_doc for _n_doc in unlearn_info['qid_did_map'][query]
                 if _n_doc not in unlearn_info['qrels'][query]]
            selected_neg_docs = task_utils.random_sample(q_neg_samples, self.config['num_neg_docs_used'],
                                                         _seed=self.config['seed'])
            for pos in docs:
                idx = qid_did_index[(query, pos)]
                _qid, _did, _q_txt, pos_txt = dataset[idx]
                neg_txts = []
                for neg in selected_neg_docs:
                    n_idx = qid_did_index[(query, neg)]
                    _qid, _did, _, neg_txt = dataset[n_idx]
                    neg_txts.append(neg_txt)
                disjoint_q += [_q_txt] * len(neg_txts)
                disjoint_pos += [pos_txt] * len(neg_txts)
                disjoint_neg += neg_txts

        substitute_q, substitute_pos, substitute_neg = [], [], []
        for query, docs in tqdm(data_substitute.items(), desc='building retain_dataset...'):
            q_neg_samples = [_n_doc for _n_doc in unlearn_info['qid_did_map'][query]
                 if _n_doc not in unlearn_info['qrels'][query] and  _n_doc not in docs]
            selected_neg_docs = task_utils.random_sample(q_neg_samples, self.config['num_neg_docs_used'],
                                                         _seed=self.config['seed'])
            for pos in docs:
                idx = qid_did_index[(query, pos)]
                _qid, _did, _q_txt, pos_txt = dataset[idx]
                neg_txts = []
                for neg in selected_neg_docs:
                    n_idx = qid_did_index[(query, neg)]
                    _qid, _did, _, neg_txt = dataset[n_idx]
                    neg_txts.append(neg_txt)
                substitute_q += [_q_txt] * len(neg_txts)
                substitute_pos += [pos_txt] * len(neg_txts)
                substitute_neg += neg_txts

        q_forgetting, d_forgetting, q_entangled, d_entangled = [], [], [], []
        for q, pos, _q, _pos in forget_entangled_list:
            idx = qid_did_index[(q, pos)]
            _qid, _did, q_txt, pos_txt = dataset[idx]
            if _q and _pos:
                _qid, _did, _q_txt, _pos_txt = dataset[qid_did_index[(_q, _pos)]]
            else:
                _q_txt, _pos_txt = "", ""
            q_forgetting.append(q_txt)
            d_forgetting.append(pos_txt)
            q_entangled.append(_q_txt)
            d_entangled.append(_pos_txt)

        forget_entangled_dataset = SimpleStringDataset(q_forgetting, d_forgetting, q_entangled, d_entangled)
        disjoint_dataset = SimpleStringDataset(disjoint_q, disjoint_pos, disjoint_neg)
        substitute_dataset = SimpleStringDataset(substitute_q, substitute_pos, substitute_neg)
        return forget_entangled_dataset, disjoint_dataset, substitute_dataset
