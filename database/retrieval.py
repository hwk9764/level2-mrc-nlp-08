import json
import os
import pickle
import time
import random
import logging
from tqdm import tqdm
from contextlib import contextmanager
from typing import List, NoReturn, Optional, Tuple, Union, Callable
from utils.arguments_inference import ModelArguments, DataTrainingArguments, OurTrainingArguments
from datasets import DatasetDict, Features, Sequence, Value
import faiss
import numpy as np
import pandas as pd
from database.sparse_retrieval import SparseRetrieval
from datasets import Dataset, concatenate_datasets, load_from_disk
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from tqdm.auto import tqdm

logger = logging.getLogger("mrc")
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)

seed = 104
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정

########################################
### Sparse 성능 테스트 및 Vector 저장 ###
########################################
class TFIDFModel:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        
    def train(self, model_file):
        print("Training TF-IDF model...")
        self.tfidf_matrix = self.vectorizer.fit_transform(tqdm(self.documents, desc="TF-IDF Training"))        
        with open(model_file, 'wb') as f:
            pickle.dump((self.vectorizer, self.tfidf_matrix), f)
        print(f"TF-IDF model saved to {model_file}")
    
    def load(self, model_file):
        print(f"Loading TF-IDF model from {model_file}...")
        with open(model_file, 'rb') as f:
            self.vectorizer, self.tfidf_matrix = pickle.load(f)
    
    def get_top_k(self, query, k=5):
        query_vec = self.vectorizer.transform([query])
        cosine_similarities = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        top_k_indices = cosine_similarities.argsort()[-k:][::-1]
        top_k_documents = [(self.documents[i], cosine_similarities[i]) for i in top_k_indices]
        return top_k_documents
    
class BM25Model:
    def __init__(self, documents, tokenizer, k1, b):
        self.tokenizer = tokenizer
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.bm25 = None
        
    def train(self, model_file):
        print("Training BM25 model...")
        tokenized_documents = [self.tokenizer.tokenize(doc) for doc in tqdm(self.documents, desc="BM25 Tokenizing")]
        self.bm25 = BM25Okapi(tokenized_documents, k1=self.k1, b=self.b)
        with open(model_file, 'wb') as f:
            pickle.dump(self.bm25, f)
        print(f"BM25 model saved to {model_file}")
        
    def load(self, model_file):
        print(f"Loading BM25 model from {model_file}...")
        with open(model_file, 'rb') as f:
            self.bm25 = pickle.load(f)
            
    def get_top_k(self, query, k=5):
        tokenized_query = self.tokenizer.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
        top_k_documents = [(self.documents[i], bm25_scores[i]) for i in top_k_indices]
        return top_k_documents

#################
### Inference ###
#################
# model_args, training_args, data_args, datasets
def run_sparse_retrieval(
    model_args: ModelArguments,
    data_args: DataTrainingArguments,
    training_args: OurTrainingArguments,
    datasets: DatasetDict,
) -> DatasetDict:

    # Query에 맞는 Passage들을 Retrieval 합니다.
    retriever = SparseRetrieval(model_args)
    retriever.get_sparse_embedding()
    
    if data_args.use_faiss:
        retriever.build_faiss(num_clusters=data_args.num_clusters)
        df = retriever.retrieve_faiss(
            datasets["validation"], topk=data_args.top_k_retrieval
        )
    else:
        df = retriever.retrieve(datasets["validation"], topk=data_args.top_k_retrieval)

    # test data 에 대해선 정답이 없으므로 id question context 로만 데이터셋이 구성됩니다.
    if training_args.do_predict:
        f = Features(
            {
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )

    # train data 에 대해선 정답이 존재하므로 id question context answer 로 데이터셋이 구성됩니다.
    elif training_args.do_eval:
        f = Features(
            {
                "answers": Sequence(
                    feature={
                        "text": Value(dtype="string", id=None),
                        "answer_start": Value(dtype="int32", id=None),
                    },
                    length=-1,
                    id=None,
                ),
                "context": Value(dtype="string", id=None),
                "id": Value(dtype="string", id=None),
                "question": Value(dtype="string", id=None),
            }
        )
    datasets = DatasetDict({"validation": Dataset.from_pandas(df, features=f)})
    return datasets