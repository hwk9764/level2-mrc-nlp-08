from typing import Optional
import numpy as np
from datasets import load_from_disk
from torch.utils.data import Dataset, TensorDataset
from tqdm import tqdm
from transformers.tokenization_utils_base import PreTrainedTokenizerBase


class DPRDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer: PreTrainedTokenizerBase,
        in_batch_negatives: Optional[bool] = True,
        num_negative_samples: int = 3, #negative sample의 수
    ):
        dataset = load_from_disk(dataset_path=dataset_path)

        if in_batch_negatives: #각각의 배치에서 하나의 pos_example / 여러개의 neg_example 사용
            print('Constructing in-batch negatives..')
            # corpus = np.array(list(set([example for example in dataset["context"]])))
            corpus = np.array(list(dict.fromkeys(dataset['context']).keys()))
            p_with_negs = []
            for examples in tqdm(dataset['context']): #'context' 속 각각의 example에 대해서
                while True:
                    neg_idx = np.random.randint(len(corpus), size=num_negative_samples)
                    #네거티브 샘플로 활용할 examples의 index를 랜덤으로 뽑음 (num_negative_samples만큼) 
                    if not examples in corpus[neg_idx]: #랜덤 선택한 문장 != 현재 example인지 확인
                        negative_examples = corpus[neg_idx]
                        p_with_negs.append(examples) #하나의 pos_example로써 활용됨
                        p_with_negs.extend(negative_examples)
                        break
            #여기 bm25로 hard negative 구현하는 코드

        #context 토큰화
        context_seqs = tokenizer(
            p_with_negs if in_batch_negatives else dataset['context'],
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        #question 토큰화
        q_seqs = tokenizer(
            dataset['question'],
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )

        if in_batch_negatives:
            max_len = context_seqs['input_ids'].size(-1)  # context_seqs['input_ids'] : (배치사이즈, num_negs + 1(본인), max_len)
            context_seqs['input_ids'] = context_seqs['input_ids'].view(-1, num_negative_samples + 1, max_len)
            context_seqs['attention_mask'] = context_seqs['attention_mask'].view(-1, num_negative_samples + 1, max_len)
            context_seqs['token_type_ids'] = context_seqs['token_type_ids'].view(-1, num_negative_samples + 1, max_len)

        self.dataset = TensorDataset(
            context_seqs['input_ids'], context_seqs['attention_mask'], context_seqs['token_type_ids'],
            q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
        )
        #context_seq : 배치수 만큼의 질문에 대해서 각 질문당 (pos sample 1개)-(neg sample 여러개) 정보가 포함됨
        #q_seqs에는 배치 수 만큼의 질문 토큰화 정보 저장됨

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
