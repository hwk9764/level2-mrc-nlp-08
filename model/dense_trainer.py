import json
import os
import logging
from datetime import datetime
import sys
from typing import List, Optional, Tuple
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

import torch
import torch.nn.functional as F
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import (
    DataLoader,
    Dataset,
    RandomSampler,
    SequentialSampler,
    TensorDataset,
)
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup, TrainingArguments
from transformers.tokenization_utils_base import PreTrainedTokenizerBase

from model.dense_model import BertEncoder

from utils.arguments_dense import RetrieverArguments


class BiEncoderTrainer:
    def __init__(
        self,
        args: TrainingArguments = None,
        train_dataset: torch.utils.data.Dataset = None,
        eval_dataset: torch.utils.data.Dataset = None,
        tokenizer: PreTrainedTokenizerBase = None,
        context_encoder: BertEncoder = None,
        q_encoder: BertEncoder = None,
        in_batch_negatives: bool = True,
        num_neg_samples: int = 3,
        context_encoder_path: str = 'resources/pq/context_encoder.pth',
        q_encoder_path: str = 'resources/pq/q_encoder.pth',
        context_path: str = 'resources/data/wikipedia_documents.json',
        output_dir: str = 'resources/dense'
    ) -> None:

        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer
        self.in_batch_negatives = in_batch_negatives
        self.num_neg_samples = num_neg_samples
        self.context_encoder = context_encoder
        self.q_encoder = q_encoder
        self.context_encoder_path = context_encoder_path
        self.q_encoder_path = q_encoder_path
        
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        log_file = os.path.join(self.output_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
        file_handler = logging.FileHandler(log_file)
        self.logger.addHandler(file_handler)
        
        self.logger.info("Initializing BiEncoderTrainer")
        self.logger.info(f"Output directory: {self.output_dir}")

        with open(context_path, 'r', encoding='utf-8') as f:
            wiki = json.load(f)

        self.search_corpus = list(dict.fromkeys([v['text'] for v in wiki.values()]))

        print('Start tokenizing wiki docs..')
        self.wiki_tokens = self.tokenizer(
            self.search_corpus, padding='max_length', truncation=True, return_tensors='pt'
        )
        print('Tokenizing wiki docs has been finished.')
        # {
        #     'input_ids': tensor([[ .... ]]),
        #     'attention_mask': tensor([[ .... ]]),
        #     'token_type_ids': tensor([[ .... ]]) 각각 max_length로 패딩됨
        # }

    def train_one_epoch(
        #한 에폭동안 인코더들 학습. 데이터를 배치단위로 처리하고 모델 업데이트
        self,
        epoch_iterator: DataLoader,
        optimizer: Optimizer,
        scheduler: _LRScheduler
    ) -> float:
        batch_size = self.args.per_device_train_batch_size

        batch_loss = 0.0 #현재 에폭에서 발생한 총 손실

        if self.in_batch_negatives:
            for training_step, batch in enumerate(epoch_iterator): 
                #epoch_iterator = tqdm(train_dataloader, desc='Iter', leave=True)
                self.context_encoder.train()
                self.q_encoder.train()
                
                if torch.cuda.is_available():
                    batch = tuple(b.cuda() for b in batch)
                    targets = torch.zeros(batch_size).long()
                    targets = targets.cuda()

                context_inputs = {
                    'input_ids': batch[0].view(batch_size * (self.num_neg_samples + 1), -1),
                    'attention_mask': batch[1].view(batch_size * (self.num_neg_samples + 1), -1),
                    'token_type_ids': batch[2].view(batch_size * (self.num_neg_samples + 1), -1)
                }
                q_inputs = {
                    'input_ids': batch[3],
                    'attention_mask': batch[4],
                    'token_type_ids': batch[5]
                }

                #context랑 q를 각각의 인코더에 통과시켜서 임베딩값을 얻는다
                context_outputs = self.context_encoder(**context_inputs)  # (batch_size, emb_dim)
                q_outputs = self.q_encoder(**q_inputs)

                context_outputs = context_outputs.view(batch_size, (self.num_neg_samples + 1), -1)
                q_outputs = q_outputs.view(batch_size, 1, -1)

                #context, q 사이에 유사도 계산/log_softmax 적용
                similarity_scores = torch.bmm(q_outputs, torch.transpose(context_outputs, 1, 2)).squeeze()
                similarity_scores = similarity_scores.view(batch_size, -1)
                similarity_scores = F.log_softmax(similarity_scores, dim=1)

                loss = F.nll_loss(similarity_scores, targets)

                batch_loss += loss.item() #현재 에폭의 총 손실 계산

                loss.backward()

                if (training_step + 1) % self.args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step()
                    self.context_encoder.zero_grad()
                    self.q_encoder.zero_grad()

                epoch_iterator.set_description(
                    f"Loss {loss:.04f} at step {training_step}"
                )

                del context_inputs, q_inputs

            torch.cuda.empty_cache()

            return batch_loss / len(epoch_iterator) #평균 손실 반환

    def train(self) -> Tuple[BertEncoder, BertEncoder]:
        train_dataloader = self.get_train_dataloader() #배치 단위 학습데이터 로드

        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.context_encoder.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.context_encoder.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0},
            {'params': [p for n, p in self.q_encoder.named_parameters() if not any(nd in n for nd in no_decay)],'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.q_encoder.named_parameters() if any(nd in n for nd in no_decay)],'weight_decay': 0.0}
        ]

        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        training_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs #총 데이터 수 = 배치 크기 * 에폭 수
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=(training_total * self.args.warmup_ratio),
            num_training_steps=training_total
        )
        
        if torch.cuda.is_available():
            self.context_encoder.cuda()
            self.q_encoder.cuda()
        
        self.context_encoder.zero_grad()
        self.q_encoder.zero_grad()
        torch.cuda.empty_cache()

        train_iterator = tqdm(range(int(self.args.num_train_epochs)), desc='Epoch')
        best_score = 0

        for _ in train_iterator: #한 에폭마다 train_one_epoch()로 학습 수행, return되는 평균손실 출력
            epoch_iterator = tqdm(train_dataloader, desc='Iter', leave=True)

            train_loss = self.train_one_epoch(epoch_iterator=epoch_iterator, optimizer=optimizer, scheduler=scheduler)
            print(f'Train loss: {train_loss:.4f}')

            top_1, top_10, top_30, top_40, top_100 = self.evaluate()

            if top_40 > best_score:
                self.save_model_weights()
                self.logger.info("model weights saved..")
                best_score = top_40

        return self.context_encoder, self.q_encoder
        
    def save_model_weights(self) -> None:
        self.logger.info(f"Saving model weights to {self.context_encoder_path} and {self.q_encoder_path}")
        torch.save(self.context_encoder.state_dict(), self.context_encoder_path)
        torch.save(self.q_encoder.state_dict(), self.q_encoder_path)
        self.logger.info("Model weights saved successfully")

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError('Trainer: training requires a train_dataset.')
        else:
            train_sampler = RandomSampler(self.train_dataset)

        data_loader = DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            drop_last=True,
        )

        return data_loader

    def get_eval_dataloader(self, eval_dataset: Optional[Dataset] = None) -> DataLoader:

        if eval_dataset is None and self.eval_dataset is None:
            raise ValueError("Trainer: evaluation requires an eval_dataset.")

        eval_dataset = eval_dataset if eval_dataset is not None else self.eval_dataset

        sampler = SequentialSampler(eval_dataset)

        data_loader = DataLoader(
            eval_dataset,
            sampler=sampler,
            batch_size=self.args.per_device_eval_batch_size,
            drop_last=True,
        )

        return data_loader

    #def get_test_dataloader(self, test_dataset: Dataset) -> DataLoader:

        sampler = SequentialSampler(test_dataset)

        data_loader = DataLoader(
            test_dataset,
            sampler=sampler,
            batch_size=self.args.per_device_eval_batch_size,
            drop_last=True,
        )

        return data_loader

    def evaluate(self) -> List[float]:
        self.logger.info("Starting evaluation")
        batch_size = self.args.per_device_eval_batch_size

        with torch.no_grad():
            self.context_encoder.eval()
            self.q_encoder.eval()

            question = self.eval_dataset['question']
            gold_passage = self.eval_dataset['context']

            q_seqs_eval = self.tokenizer(
                question, padding='max_length', truncation=True, return_tensors='pt'
            ).to('cuda')
            q_emb = self.q_encoder(**q_seqs_eval).to('cpu')  # (num_questions, emb_dim)

            wiki_iterator = TensorDataset(
                self.wiki_tokens['input_ids'],
                self.wiki_tokens['attention_mask'],
                self.wiki_tokens['token_type_ids']
            )
            wiki_dataloader = DataLoader(wiki_iterator, batch_size=batch_size)

            context_embs = []
            for context in tqdm(wiki_dataloader):
                if torch.cuda.is_available():
                    context = tuple(c.cuda() for c in context)

                context_inputs = {
                    'input_ids': context[0],
                    'attention_mask': context[1],
                    'token_type_ids': context[2]
                }

                context_emb = self.context_encoder(**context_inputs)
                context_embs.append(context_emb)

            context_embs = torch.cat(context_embs, dim=0).view(len(wiki_iterator), -1)
            context_embs = context_embs.to('cpu')  # (num_contexts, emb_dim)

            sim_scores = torch.matmul(q_emb, torch.transpose(context_embs, 0, 1))

            rank = torch.argsort(sim_scores, dim=1, descending=True).squeeze()

            def eval_score(k: int = 1) -> float:
                total = len(question)
                cnt = 0

                for i in range(total):
                    self.logger.info(f"\nCurrent question: {question[i]}")
                    
                    gold_index = self.search_corpus.index(gold_passage[i])
                    self.logger.info(f"Gold passage index: {gold_index}")
                    
                    top_k = rank[i][:k]
                    self.logger.info(f"Top-{k} predicted passage indices: {top_k}")
                    
                    pred_corpus = []
                    for top in top_k:
                        pred_corpus.append(self.search_corpus[top])

                    if gold_passage[i] in pred_corpus:
                        cnt += 1

                res = cnt / total
                self.logger.info(f'Top-{k} score is {res:.4f}')

                return res
            
        self.logger.info(f"Sample question embedding: {q_emb[0][:10]}")
        self.logger.info(f"Sample context embedding: {context_embs[0][:10]}")
        self.logger.info(f"Sample similarity score: {sim_scores[0][:10]}")

        self.logger.info('********** Evaluation **********')
        
        return [
            eval_score(1),
            eval_score(10),
            eval_score(30),
            eval_score(40),
            eval_score(100),
        ]