# encoding=utf-8
import numpy as np
import torch
import datetime
from torch.optim import *
import random
import pickle
from tqdm import tqdm
from transformers import AutoTokenizer
from models.model_utils import separate_parameters_by_type
from nltk.tokenize import word_tokenize
from torch.nn import functional as F
from unranking_methods import bad_ranking_teacher_loss, ranking_teacher_loss, RankingDistilLoss, CoCoLDistillationLoss


def print_message(s):
    print("[{}] {}".format(datetime.datetime.utcnow().strftime("%b %d, %H:%M:%S"), s), flush=True)


def save_model(model, file_path):
    model_state = model.state_dict()
    if isinstance(model, torch.nn.DataParallel):
        # Get rid of 'module' before the name of states
        model_state = model.module.state_dict()
    for key in model_state.keys():  # Always save it to cpu
        model_state[key] = model_state[key].cpu()
    torch.save(model_state, file_path)


def unlearn_neg_teacher_hinge(train_iter, teacher_model, student_model, config):
    def _process_traning_bath(batch):
        q, pos, negs_batch, substitute, labels = batch
        if config.get('concatenate_query_doc') and config['concatenate_query_doc']:
            pos = tokenizer(list(q), list(pos), truncation=True, padding='max_length',
                            max_length=config['query_max_len'] + config['target_max_len'], return_tensors='pt')
            substitute = tokenizer(list(q), list(substitute), truncation=True, padding='max_length',
                                   max_length=config['query_max_len'] + config['target_max_len'], return_tensors='pt')
            negs = [tokenizer(list(q), list(n_txt), truncation=True, padding='max_length',
                              max_length=config['query_max_len'] + config['target_max_len'], return_tensors='pt') for
                    n_txt in negs_batch]
            q = pos
        else:
            q = tokenizer(list(q), truncation=True, padding='max_length', max_length=config['query_max_len'],
                          return_tensors='pt')
            pos = tokenizer(list(pos), truncation=True, padding='max_length', max_length=config['target_max_len'],
                            return_tensors='pt')
            substitute = tokenizer(list(substitute), truncation=True, padding='max_length',
                                   max_length=config['target_max_len'],
                                   return_tensors='pt')
            negs = [tokenizer(list(n_txt), truncation=True, padding='max_length',
                              max_length=config['target_max_len'], return_tensors='pt') for n_txt in negs_batch]

        for elems in negs + [q, pos, substitute]:
            for idx, item in elems.items():
                elems[idx] = item.to(config['device'])
        labels = labels.to(config['device'])
        return q, pos, negs, substitute, labels

    if 'step' not in config:
        config['step'] = 1

    all_params = student_model.parameters()
    optimizer = Adam(all_params, lr=config['learning_rate'], weight_decay=0)
    student_model.train()
    teacher_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config['bert_pretrained_model'])
    criterion = RankingDistilLoss(config, quantile_neg=config['quantile_neg'])
    for epoch in range(config['start_epoch'] + 1, config['start_epoch'] + config['epochs'] + 1):
        step = 0
        loss_sum = torch.zeros(1).to(config['device'])
        total = len(train_iter)
        log_slice = total // 5 + 1
        pbar = tqdm(total=total, desc='bad teacher unlearning progress in epoch {}'.format(epoch))
        for batch in train_iter:
            q, pos, negs, substitute, labels = _process_traning_bath(batch)
            optimizer.zero_grad()
            student_pos_out = student_model(q, pos)
            student_subs_out = student_model(q, substitute)
            student_neg_outs = [student_model(q, neg) for neg in negs]
            # Teacher model outputs (detach gradients)
            with torch.no_grad():
                teacher_pos_out = teacher_model(q, pos)
                teacher_neg_outs = [teacher_model(q, neg) for neg in negs]
            # Calculate loss
            loss = criterion(student_pos_out, student_neg_outs, student_subs_out, teacher_pos_out, teacher_neg_outs,
                             labels)
            loss_sum = loss_sum.data + loss.detach().data
            loss.backward()
            optimizer.step()
            pbar.update(1)
            step += 1
            if step > 0 and step % log_slice == 0:
                print_message(str(step / total) + "," + str(loss_sum.item() / log_slice) + "\n")
                loss_sum = torch.zeros(1).to(config['device'])

        if epoch == config['start_epoch'] + 1 or epoch == config['start_epoch'] + config['epochs'] + 1 or epoch % \
                config['step'] == 0:
            save_model(student_model,
                       config['model_save_path'].format(config['model_save_name'], config['stage'], epoch))
        pbar.close()
    return student_model


def unlearn_neg_teacher(train_iter, teacher_model, student_model, config):
    def _process_traning_bath(batch):
        q, pos, negs_batch, labels = batch
        if config.get('concatenate_query_doc') and config['concatenate_query_doc']:
            pos = tokenizer(list(q), list(pos), truncation=True, padding='max_length',
                            max_length=config['query_max_len'] + config['target_max_len'], return_tensors='pt')

            negs = [tokenizer(list(q), list(n_txt), truncation=True, padding='max_length',
                              max_length=config['query_max_len'] + config['target_max_len'], return_tensors='pt') for
                    n_txt in negs_batch]

            q = pos
        else:
            q = tokenizer(list(q), truncation=True, padding='max_length', max_length=config['query_max_len'],
                          return_tensors='pt')
            pos = tokenizer(list(pos), truncation=True, padding='max_length', max_length=config['target_max_len'],
                            return_tensors='pt')
            negs = [tokenizer(list(n_txt), truncation=True, padding='max_length',
                              max_length=config['target_max_len'], return_tensors='pt') for n_txt in negs_batch]

        for idx, item in q.items():
            q[idx] = item.to(config['device'])
        for idx, item in pos.items():
            pos[idx] = item.to(config['device'])
        for neg in negs:
            for idx, item in neg.items():
                neg[idx] = item.to(config['device'])
        labels = labels.to(config['device'])
        return q, pos, negs, labels

    if 'step' not in config:
        config['step'] = 1
    all_params = student_model.parameters()
    optimizer = Adam(all_params, lr=config['learning_rate'], weight_decay=0)
    student_model.train()
    teacher_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config['bert_pretrained_model'])
    for epoch in range(config['start_epoch'] + 1, config['start_epoch'] + config['epochs'] + 1):
        step = 0
        loss_sum = torch.zeros(1).to(config['device'])
        total = len(train_iter)
        log_slice = total // 5 + 1
        pbar = tqdm(total=total, desc='bad teacher unlearning progress in epoch {}'.format(epoch))
        for batch in train_iter:
            q, pos, negs, labels = _process_traning_bath(batch)
            optimizer.zero_grad()
            # Forward pass for positive samples
            student_pos_out = student_model(q, pos)
            # Forward pass for negative samples
            student_neg_outs = [student_model(q, neg) for neg in negs]
            # Teacher model outputs (detach gradients)
            with torch.no_grad():
                teacher_pos_out = teacher_model(q, pos)
                teacher_neg_outs = [teacher_model(q, neg) for neg in negs]
            # Calculate loss
            loss = ranking_teacher_loss(student_pos_out, student_neg_outs, teacher_pos_out, teacher_neg_outs, labels,
                                        config['quantile_neg'])
            loss_sum = loss_sum.data + loss.detach().data
            loss.backward()
            optimizer.step()
            pbar.update(1)
            step += 1
            if step > 0 and step % log_slice == 0:
                print_message(str(step / total) + "," + str(loss_sum.item() / log_slice) + "\n")
                loss_sum = torch.zeros(1).to(config['device'])

        if epoch == config['start_epoch'] + 1 or epoch == config['start_epoch'] + config['epochs'] + 1 or epoch % \
                config['step'] == 0:
            save_model(student_model,
                       config['model_save_path'].format(config['model_save_name'], config['stage'], epoch))
        pbar.close()
    return student_model


def unlearn_bad_teacher(train_iter, teacher_model, student_model, config):
    def _process_traning_bath(batch):
        q, pos, negs_batch, labels = batch
        if config.get('concatenate_query_doc') and config['concatenate_query_doc']:
            pos = tokenizer(list(q), list(pos), truncation=True, padding='max_length',
                            max_length=config['query_max_len'] + config['target_max_len'], return_tensors='pt')

            negs = [tokenizer(list(q), list(n_txt), truncation=True, padding='max_length',
                              max_length=config['query_max_len'] + config['target_max_len'], return_tensors='pt') for
                    n_txt in negs_batch]

            q = pos
        else:
            q = tokenizer(list(q), truncation=True, padding='max_length', max_length=config['query_max_len'],
                          return_tensors='pt')
            pos = tokenizer(list(pos), truncation=True, padding='max_length', max_length=config['target_max_len'],
                            return_tensors='pt')
            negs = [tokenizer(list(n_txt), truncation=True, padding='max_length',
                              max_length=config['target_max_len'], return_tensors='pt') for n_txt in negs_batch]

        for idx, item in q.items():
            q[idx] = item.to(config['device'])
        for idx, item in pos.items():
            pos[idx] = item.to(config['device'])
        for neg in negs:
            for idx, item in neg.items():
                neg[idx] = item.to(config['device'])
        labels = labels.to(config['device'])
        return q, pos, negs, labels

    if 'step' not in config:
        config['step'] = 1
    all_params = student_model.parameters()
    optimizer = Adam(all_params, lr=config['learning_rate'], weight_decay=0)
    student_model.train()
    teacher_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(config['bert_pretrained_model'])
    for epoch in range(config['start_epoch'] + 1, config['start_epoch'] + config['epochs'] + 1):
        step = 0
        loss_sum = torch.zeros(1).to(config['device'])
        total = len(train_iter)
        log_slice = total // 5 + 1
        pbar = tqdm(total=total, desc='bad teacher unlearning progress in epoch {}'.format(epoch))
        for batch in train_iter:
            q, pos, negs, labels = _process_traning_bath(batch)
            optimizer.zero_grad()
            student_pos_out = student_model(q, pos)
            with torch.no_grad():
                teacher_pos_out = teacher_model(q, pos)
                teacher_neg_outs = [teacher_model(q, neg) for neg in negs]
            loss = bad_ranking_teacher_loss(student_pos_out, teacher_pos_out, teacher_neg_outs, labels)
            if torch.isnan(loss):
                print('*')
            loss_sum = loss_sum.data + loss.detach().data
            loss.backward()
            optimizer.step()
            pbar.update(1)
            step += 1
            if step > 0 and step % log_slice == 0:
                print_message(str(step / total) + "," + str(loss_sum.item() / log_slice) + "\n")
                loss_sum = torch.zeros(1).to(config['device'])

        if epoch == config['start_epoch'] + 1 or epoch == config['start_epoch'] + config['epochs'] + 1 or epoch % \
                config['step'] == 0:
            save_model(student_model,
                       config['model_save_path'].format(config['model_save_name'], config['stage'], epoch))
        pbar.close()
    return student_model


def unlearn_bad_good_teacher(train_iter, bad_model, good_model, student_model, config, test_loader=None):
    if 'step' not in config:
        config['step'] = 1
    all_params = student_model.parameters()
    optimizer = Adam(all_params, lr=config['learning_rate'], weight_decay=0)
    # change model mode
    student_model.train()
    bad_model.eval()
    good_model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config['bert_pretrained_model'])
    for epoch in range(config['start_epoch'] + 1, config['start_epoch'] + config['epochs'] + 1):
        step = 0
        loss_sum = torch.zeros(1).to(config['device'])
        pbar = tqdm(total=len(train_iter), desc='training progress in epoch {}'.format(epoch))
        total = len(train_iter)
        for batch in train_iter:
            q, pos, labels = batch
            if config.get('concatenate_query_doc') and config['concatenate_query_doc']:
                pos = tokenizer(list(q), list(pos), truncation=True, padding='max_length',
                                max_length=config['query_max_len'] + config['target_max_len'], return_tensors='pt')
                q = pos
            else:
                q = tokenizer(list(q), truncation=True, padding='max_length', max_length=config['query_max_len'],
                              return_tensors='pt')
                pos = tokenizer(list(pos), truncation=True, padding='max_length', max_length=config['target_max_len'],
                                return_tensors='pt')

            for idx, item in q.items():
                q[idx] = item.to(config['device'])
            for idx, item in pos.items():
                pos[idx] = item.to(config['device'])
            labels = labels.to(config['device'])
            optimizer.zero_grad()
            student_out = student_model(q, pos)
            with torch.no_grad():
                bad_teacher_out = bad_model(q, pos)
                good_teacher_out = good_model(q, pos)
                overall_teacher_out = labels * good_teacher_out + (1 - labels) * bad_teacher_out
            loss = F.kl_div(F.softmax(student_out), F.softmax(overall_teacher_out))
            loss_sum = loss_sum.data + loss.detach().data
            loss.backward()
            optimizer.step()
            step += 1
            step += 1
            pbar.update(1)
            if step > 0 and step % 500 == 0:
                # append loss to loss file
                print_message(str(step / total) + "," + str(loss_sum.item() / 500) + "\n")
                loss_sum = torch.zeros(1).to(config['device'])
        # if epoch == config['start_epoch'] + 1 or epoch == config['start_epoch'] + config['epochs'] + 1 or epoch % \
        #         config['step'] == 0:
        #     save_model(student_model,
        #                config['model_save_path'].format(config['model_save_name'], config['stage'], epoch))
        pbar.close()
    return student_model


def train_ranking_model(train_iter, model, config, test_loader=None):

    all_params = model.parameters()
    optimizer = Adam(all_params, lr=config['learning_rate'], weight_decay=0)
    criterion = torch.nn.MarginRankingLoss(margin=1, reduction='elementwise_mean').cuda(config['device'])
    # change model mode
    model.train()
    tokenizer = AutoTokenizer.from_pretrained(config['bert_pretrained_model'])

    def _process_traning_bath(batch):
        q, pos, neg = batch
        if config.get('concatenate_query_doc') and config['concatenate_query_doc']:
            pos = tokenizer(list(q), list(pos), truncation=True, padding='max_length',
                            max_length=config['query_max_len'] + config['target_max_len'], return_tensors='pt')
            neg = tokenizer(list(q), list(neg), truncation=True, padding='max_length',
                            max_length=config['query_max_len'] + config['target_max_len'], return_tensors='pt')
            q = pos
        else:
            q = tokenizer(list(q), truncation=True, padding='max_length', max_length=config['query_max_len'],
                          return_tensors='pt')
            pos = tokenizer(list(pos), truncation=True, padding='max_length', max_length=config['target_max_len'],
                            return_tensors='pt')
            neg = tokenizer(list(neg), truncation=True, padding='max_length', max_length=config['target_max_len'],
                            return_tensors='pt')

        for idx, item in q.items():
            q[idx] = item.to(config['device'])
        for idx, item in pos.items():
            pos[idx] = item.to(config['device'])
        for idx, item in neg.items():
            neg[idx] = item.to(config['device'])
        return q, pos, neg

    for epoch in range(config['start_epoch'] + 1, config['start_epoch'] + config['epochs'] + 1):
        step = 0
        loss_sum = torch.zeros(1).to(config['device'])
        pbar = tqdm(total=len(train_iter), desc='training progress in epoch {}'.format(epoch))
        total = len(train_iter)
        log_slice = total // 5 + 1
        for batch in train_iter:
            q, pos, neg = _process_traning_bath(batch)
            y1 = torch.from_numpy(np.ones((pos['input_ids'].size(dim=0)), dtype=np.int32)).to(config['device'])
            optimizer.zero_grad()
            # print(pos_local.sum(), neg_local.sum())
            score_pos = model(q, pos)
            score_neg = model(q, neg)
            loss = criterion(score_pos, score_neg, y1)
            loss_sum = loss_sum.data + loss.detach().data
            loss.backward()

            if config.get('neg_GRAD') and config['neg_GRAD']:
                with torch.no_grad():
                    for param in model.parameters():
                        if param.grad is not None:
                            param += config['learning_rate'] * param.grad
                            param.grad.zero_()
                            # param += 0.001 * param.grads
            else:
                optimizer.step()
            step += 1
            if step > 0 and step % log_slice == 0:
                # append loss to loss file
                print_message(str(step / total) + "," + str(loss_sum.item() / log_slice) + "\n")
                loss_sum = torch.zeros(1).to(config['device'])
            pbar.update(1)

        if 'model_save_step' not in config or epoch % config['model_save_step'] == 0:
            save_model(model, config['model_save_path'].format(config['model_save_name'], config['stage'], epoch))
        pbar.close()
    return model


def test_ranking_model(eval_iter, model, config):
    model.eval()
    q_ids = []
    d_ids = []
    total_scores = []
    tokenizer = AutoTokenizer.from_pretrained(config['bert_pretrained_model'])
    pbar = tqdm(total=len(eval_iter), desc='prediction progress')
    for tbatch in eval_iter:
        q_id, doc_id, q_data, doc_data = tbatch
        if config.get('concatenate_query_doc') and config['concatenate_query_doc']:
            doc_data = tokenizer(list(q_data), list(doc_data), truncation=True, padding='max_length',
                                 max_length=config['query_max_len'] + config['target_max_len'], return_tensors='pt')
            q_data = doc_data
        else:
            q_data = tokenizer(list(q_data), truncation=True, padding='max_length', max_length=config['query_max_len'],
                               return_tensors='pt')
            doc_data = tokenizer(list(doc_data), truncation=True, padding='max_length',
                                 max_length=config['target_max_len'],
                                 return_tensors='pt')

        for idx, item in q_data.items():
            q_data[idx] = item.to(config['device'])
        for idx, item in doc_data.items():
            doc_data[idx] = item.to(config['device'])
        score = model(q_data, doc_data)
        score_values = score.cpu().detach().numpy().tolist()
        total_scores += score_values
        q_ids += list(q_id)
        d_ids += list(doc_id)
        pbar.update(1)
    pbar.close()
    model.train()
    return q_ids, d_ids, total_scores


def cocol_distil_train(train_iter, disjoint_loader, model_t, model_s, config, test_loader=None, ):

    model_t.eval()
    model_s.train()
    tokenizer = AutoTokenizer.from_pretrained(config['bert_pretrained_model'])
    all_params = model_s.parameters()
    criterion = CoCoLDistillationLoss(margin=config.get('distillation_margin'), ratio=config.get('distillation_ratio'))
    optimizer = Adam(all_params, lr=config['learning_rate'], weight_decay=0)

    def _process_forget_entangled_pair(q, docs, status='consistent'):
        docs_masked = None
        encoded_docs_masked = []
        if config.get('CoCoL_learning_from_worst') and config['CoCoL_learning_from_worst'] and status == 'contrastive':
            docs_masked = [
                (lambda t: tuple(
                    " ".join(f"[unused{random.randint(1, 99)}]" for _ in text)
                    for text in t
                ))(docs[0])
            ]
        if config.get('concatenate_query_doc') and config['concatenate_query_doc']:
            encoded_docs = [tokenizer(list(q), list(pos), truncation=True, padding='max_length',
                            max_length=config['query_max_len'] + config['target_max_len'], return_tensors='pt') for pos in docs]
            if docs_masked:
                encoded_docs_masked = [tokenizer(list(q), list(pos), truncation=True, padding='max_length',
                                max_length=config['query_max_len'] + config['target_max_len'], return_tensors='pt') for pos in docs_masked]
            q = encoded_docs[0]

        else:
            q = tokenizer(list(q), truncation=True, padding='max_length', max_length=config['query_max_len'],
                          return_tensors='pt')
            encoded_docs = [tokenizer(list(pos), truncation=True, padding='max_length', max_length=config['target_max_len'],
                            return_tensors='pt') for pos in docs]
            if docs_masked:
                encoded_docs_masked = [tokenizer(list(pos), truncation=True, padding='max_length', max_length=config['target_max_len'],
                            return_tensors='pt') for pos in docs]


        for encoded in [q] + encoded_docs +  encoded_docs_masked:
            for idx, item in encoded.items():
                encoded[idx] = item.to(config['device'])

        return tuple([q] + encoded_docs + encoded_docs_masked)

    for epoch in range(config['start_epoch'] + 1, config['start_epoch'] + config['epochs'] + 1):
        loss_sum = torch.zeros(1).to(config['device'])
        pbar = tqdm(total=len(train_iter), desc='training progress in epoch {}'.format(epoch))
        total = len(train_iter)
        step = 0
        for batch in train_iter:
            q_forgetting, d_forgetting, q_retaining, d_retaining = batch
            if config.get('CoCoL_learning_from_worst') and config['CoCoL_learning_from_worst']:
                q_forgetting, d_forgetting, d_forgetting_worst = _process_forget_entangled_pair(q_forgetting, [d_forgetting], status='contrastive')
                with torch.no_grad():
                    score_forgetting_t = model_t(q_forgetting, d_forgetting_worst)
                optimizer.zero_grad()
            else:
                q_forgetting, d_forgetting = _process_forget_entangled_pair(q_forgetting, [d_forgetting])
                with torch.no_grad():
                    score_forgetting_t = model_t(q_forgetting, d_forgetting)
                optimizer.zero_grad()

            score_forgetting_s = model_s(q_forgetting, d_forgetting)
            loss = criterion(score_forgetting_t, score_forgetting_s, loss_type='contrastive')

            q_retaining = [_txt for _txt in q_retaining if len(_txt) > 0]
            d_retaining = [_txt for _txt in d_retaining if len(_txt) > 0]
            if len(q_retaining) > 0 and len(d_retaining) > 0:
                q_retaining, d_retaining = _process_forget_entangled_pair(q_retaining, [d_retaining])
                with torch.no_grad():
                    score_retaining_t = model_t(q_retaining, d_retaining)
                optimizer.zero_grad()
                score_retaining_s = model_s(q_retaining, d_retaining)
                retaining_loss = criterion(score_retaining_t, score_retaining_s, loss_type='consistent')
                loss += retaining_loss
            loss.backward()
            optimizer.step()
            step += 1
            loss_sum = loss_sum.data + loss.detach().data

            pbar.update(1)
            if step >= 0 and step % 100 == 0:
                # append loss to loss file
                print_message(str(step / total) + "," + str(loss_sum.item() / 100) + "\n")
                loss_sum = torch.zeros(1).to(config['device'])
        pbar.close()

        pbar = tqdm(total=len(disjoint_loader), desc='training disjoint progress in epoch {}'.format(epoch))
        total = len(disjoint_loader)
        step = 0

        for batch in disjoint_loader:
            q_disjoint, d_disjoint_pos, d_disjoint_neg = batch
            q_disjoint, d_disjoint_pos, d_disjoint_neg = _process_forget_entangled_pair(q_disjoint, [d_disjoint_pos, d_disjoint_neg])

            with torch.no_grad():
                score_disjoint_neg_t = model_t(q_disjoint, d_disjoint_neg)
                score_disjoint_pos_t = model_t(q_disjoint, d_disjoint_pos)

            optimizer.zero_grad()
            score_disjoint_neg_s = model_s(q_disjoint, d_disjoint_neg)
            score_disjoint_pos_s = model_s(q_disjoint, d_disjoint_pos)

            loss_retaining_p = criterion(score_disjoint_neg_t, score_disjoint_neg_s, loss_type='consistent')
            loss_retaining_n = criterion(score_disjoint_pos_t, score_disjoint_pos_s, loss_type='consistent')
            loss = loss_retaining_p + loss_retaining_n
            loss.backward()
            optimizer.step()
            step += 1
            loss_sum = loss_sum.data + loss.detach().data
            pbar.update(1)

            if step >= 0 and step % 100 == 0:
                print_message(str(step / total) + "," + str(loss_sum.item() / 100) + "\n")
                loss_sum = torch.zeros(1).to(config['device'])

        if 'model_save_step' not in config or epoch % config['model_save_step'] == 0:
            save_model(model_s, config['model_save_path'].format(config['model_save_name'], config['stage'], epoch))
        pbar.close()

    return model_s
