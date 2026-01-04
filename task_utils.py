import codecs
import yaml
import os
from tqdm import tqdm
import pandas as pd
import random


def loadyaml(path):
    doc = []
    if os.path.exists(path):
        with codecs.open(path, 'r') as yf:
            doc = yaml.safe_load(yf)
    return doc


def read_qrels(file_path):
    qrels = {}
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        pbar = tqdm(total=len(lines), desc=f'reading qrel file:{file_path} ...')
        for line in lines:
            arr = line.strip().split('\t')
            if len(arr) == 2:
                q, pos = arr[0], arr[1]
            if len(arr) == 4:
                q, pos = arr[0], arr[2]
            if q not in qrels:
                qrels[q] = []
            qrels[q].append(pos)
            pbar.update(1)
        pbar.close()
        return qrels


def read_performance_csv_by_pandas(result_csv):
    prediction = {}
    df_r = pd.read_csv(result_csv)
    pbar = tqdm(total=df_r.shape[0], desc='loading performance_csv')
    for idx in range(df_r.shape[0]):
        qid, doc_id, score = str(df_r['q_id'][idx]), str(df_r['d_id'][idx]), df_r['rel'][idx]
        if qid not in prediction:
            prediction[qid] = {}
        prediction[qid][doc_id] = float(score)
        pbar.update(1)
    pbar.close()
    return prediction


def calculate_mrr(qrels, predictions, topx=10):
    mrr = 0
    for qid, docs in predictions.items():
        ranked = sorted(docs, key=docs.get, reverse=True)
        for i in range(min(len(ranked), topx)):
            if ranked[i] in qrels[qid]:
                mrr += 1 / (i + 1)
                break
    mrr /= len(predictions)
    return mrr


def random_sample(array, l, _seed=42):
    random.seed(_seed)
    if len(array) >= l:
        return random.sample(array, l)
    else:
        result = array[:]
        while len(result) < l:
            result.append(random.choice(array))
        return result
