import json
import os
#import pickle
import sys
from typing import List, Optional, Tuple, Union
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import torch
from datasets import Dataset, concatenate_datasets,load_from_disk
from datasets.arrow_dataset import Dataset
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, HfArgumentParser, TrainingArguments, set_seed

from typing import Optional
#from datasets import DatasetDict, Features, Sequence, Value
#from utils.arguments_dense import ModelArguments, DataTrainingArguments, RetrieverArguments
#from utils.dense_dataloader import DPRDataset
from model.dense_model import BertEncoder
#from dense_trainer import BiEncoderTrainer
#import logging

from typing import Optional

# data_args.use_faiss 값에 따라 FAISS 인덱스 사용 여부 결정
# FAISS 사용 시:
# num_clusters와 top_k_retrieval 값으로 검색 설정
# retrieve_faiss() 메서드로 검색 수행
# FAISS 미사용 시:
# retrieve() 메서드로 검색 수행
# 검색 결과는 df 변수에 데이터프레임 형태로 저장


class DenseRetrieval:
    def __init__(
        self,
        context_path: str = "../data/wikipedia_documents.json",
    ) -> None:
        with open(context_path, 'r', encoding='utf-8') as f:
            wiki = json.load(f)
        self.search_corpus = list(dict.fromkeys([v['text'] for v in wiki.values()]))

    def get_relevant_doc(
        self,
        query_vec: torch.Tensor,
        context_vecs: torch.Tensor,
        top_k: int = 1
    ) -> Tuple[List, List]:
        similarity_scores = torch.matmul(query_vec, context_vecs.t()).squeeze()
        top_k_scores, top_k_indices = similarity_scores.topk(k=top_k)
        return top_k_scores.tolist(), top_k_indices.tolist()

    def retrieve(
        self,
        query_or_dataset: Union[str, Dataset, TensorDataset],
        context_vecs: torch.Tensor,
        tokenizer: AutoTokenizer,
        q_encoder: BertEncoder,
        top_k: Optional[int] = 1,
        device: Optional[str] = 'cuda',
    ) -> Union[Tuple[List, List], pd.DataFrame]:
        
        if isinstance(query_or_dataset, str):
            input_query = tokenizer(
                query_or_dataset, padding='max_length', truncation=True, return_tensors='pt'
            ).to(device)

            with torch.no_grad():
                output_query = q_encoder(**input_query).cpu()

            doc_scores, doc_indices = self.get_relevant_doc(output_query, context_vecs, top_k=top_k)
            return (doc_scores, [self.search_corpus[i] for i in doc_indices])

        elif isinstance(query_or_dataset, (Dataset, TensorDataset)):
            total = []
            with torch.no_grad():
                q_encoder.eval()
                for idx, example in enumerate(tqdm(query_or_dataset, desc='Dense passage retrieval')):
                    if isinstance(query_or_dataset, Dataset):
                        question = example['question']
                    else:  # TensorDataset
                        question = tokenizer.decode(example[0], skip_special_tokens=True)

                    input_query = tokenizer(
                        question, padding='max_length', truncation=True, return_tensors='pt'
                    ).to(device)

                    output_query = q_encoder(**input_query).cpu()
                    doc_scores, doc_indices = self.get_relevant_doc(output_query, context_vecs, top_k=top_k)

                    tmp = {
                        'question': question,
                        'id': example['id'] if isinstance(query_or_dataset, Dataset) else idx,
                        'context': ' '.join([self.search_corpus[i] for i in doc_indices])
                    }

                    if isinstance(query_or_dataset, Dataset) and 'context' in example and 'answers' in example:
                        tmp['original_context'] = example['context']
                        tmp['answers'] = example['answers']

                    total.append(tmp)

            return pd.DataFrame(total)