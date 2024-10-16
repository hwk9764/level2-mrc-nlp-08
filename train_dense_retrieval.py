import torch
import numpy as np
import random
import json
import os
import logging
from datetime import datetime

from datasets import load_from_disk
from transformers import HfArgumentParser
from transformers import BertTokenizerFast, TrainingArguments

from utils.dense_dataloader import DPRDataset
from utils.arguments_dense import DataTrainingArguments, RetrieverArguments
from model.dense_trainer import BiEncoderTrainer
from model.dense_retrieval import BertEncoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

seed = 104
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정
torch.manual_seed(seed) # torch random seed 고정
torch.cuda.manual_seed_all(seed)

def save_args(args, output_dir):
     with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)

def main():
    parser = HfArgumentParser(
        (DataTrainingArguments, RetrieverArguments)
    )
    data_args, retriever_args  = parser.parse_args_into_dataclasses() 
    # 기본 설정
    
    #output_dir = os.path.join("resources", "dense")
    output_dir = os.path.join(retriever_args.dpr_encoder_save_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    log_file = os.path.join(output_dir, f"train_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    file_handler = logging.FileHandler(log_file)
    logger.addHandler(file_handler)
    
    save_args(data_args, output_dir)
    save_args(retriever_args, output_dir)
    
    model_name = "bert-base-uncased"
    train_file = "resources/data/train_dataset/train"
    #eval_file = "resources/data/train_dataset/validation"

    # 토크나이저 및 인코더 초기화
    tokenizer = BertTokenizerFast.from_pretrained(model_name)
    context_encoder = BertEncoder.from_pretrained(model_name)
    q_encoder = BertEncoder.from_pretrained(model_name)

    # 학습 인자 설정
    training_args = TrainingArguments(
        output_dir='resources/dense',
        num_train_epochs=3,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=os.path.join(output_dir, "logs"),
    )

    # DPRDataset을 사용하여 데이터 로드
    train_dataset = DPRDataset(train_file, tokenizer)
    #eval_dataset = DPRDataset(eval_file, tokenizer)
    eval_dataset = load_from_disk(dataset_path=data_args.eval_dataset_path)

    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Eval dataset size: {len(eval_dataset)}")
    logger.info(f"First item in train dataset: {train_dataset[0]}") #전처리가 끝난 데이터셋의 첫번째 값
            # context_seqs['input_ids'], context_seqs['attention_mask'], context_seqs['token_type_ids'],
            # q_seqs['input_ids'], q_seqs['attention_mask'], q_seqs['token_type_ids']
    logger.info(f"First item in eval dataset: {eval_dataset[0]}")
    
    # BiEncoderTrainer 초기화 및 학습
    trainer = BiEncoderTrainer(
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        context_encoder=context_encoder,
        q_encoder=q_encoder,
        in_batch_negatives=retriever_args.dpr_in_batch_negatives,
        num_neg_samples=retriever_args.dpr_num_neg_samples,
        context_encoder_path=retriever_args.dpr_context_encoder_path,
        q_encoder_path=retriever_args.dpr_q_encoder_path,
        context_path=data_args.context_path,
        
    )

    # 학습 실행
    logger.info("Starting training...")
    context_encoder, q_encoder = trainer.train()
    logger.info("Training completed.")

    # 모델 가중치 저장
    trainer.save_model_weights()
    logger.info("Model weights saved.")

    # 평가 실행
    logger.info("Starting evaluation...")
    evaluation_results = trainer.evaluate()
    logger.info("Evaluation completed.")
    
    with open(os.path.join(output_dir, 'evaluation_results.json'), 'w') as f:
        json.dump(evaluation_results, f, indent=2)

    logger.info(f"Evaluation results: {evaluation_results}")


if __name__ == "__main__":
    main()