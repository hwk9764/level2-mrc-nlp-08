import os
import numpy as np
import random
import json
import logging
import logging.config
import torch
from utils.arguments_reader import ModelArguments, DataTrainingArguments, OurTrainingArguments
from utils.data_processing import ExtracionDataModule
from utils.metric_extraction import compute_metrics
from transformers import HfArgumentParser, set_seed, AutoTokenizer, AutoModelForQuestionAnswering, DataCollatorWithPadding
from model.qat_custom import QuestionAnsweringTrainer

seed = 104
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정
torch.manual_seed(seed) # torch random seed 고정
torch.cuda.manual_seed_all(seed)

config = json.load(open("./utils/log/logger.json"))
logging.config.dictConfig(config)
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    # 학습 파라미터 로깅
    logger.info(f"Model is from {model_args.model_name_or_path}")
    logger.info(f"Data is from {data_args.dataset_name}")
    logger.info("Training/evaluation parameters %s", training_args)

    # 모델을 초기화하기 전에 난수를 고정
    set_seed(training_args.seed)

    # pretrained model 과 tokenizer를 불러오기
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
    )
    logger.info(model)
    
    # 데이터 불러오기 및 전처리 data_args, training_args, tokenizer
    dm = ExtracionDataModule(data_args, training_args, tokenizer)
    train_dataset, eval_dataset = dm.get_processing_data()
    # pad_to_max_length가 True이면 이미 max length로 padding된 상태 / 그렇지 않다면 data collator에서 padding을 진행해야 함
    data_collator = DataCollatorWithPadding(
        tokenizer, pad_to_multiple_of=8 if training_args.fp16 else None
    )

    # Trainer 초기화
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset,
        eval_examples=dm.datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        post_process_function=dm._post_processing_function,
        compute_metrics=compute_metrics,
    )
    
    # Training
    train_result = trainer.train()
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    metrics["train_samples"] = len(train_dataset)

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    output_train_file = os.path.join(training_args.output_dir, "train_results.txt")

    with open(output_train_file, "w") as writer:
        logger.info("***** Train results *****")
        for key, value in sorted(train_result.metrics.items()):
            logger.info(f"  {key} = {value}")
            writer.write(f"{key} = {value}\n")

    # Training state 저장
    trainer.state.save_to_json(
        os.path.join(training_args.output_dir, "trainer_state.json")
    )

    # Evaluation
    logger.info("***** Evaluate *****")
    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(eval_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()
