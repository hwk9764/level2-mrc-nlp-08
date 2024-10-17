import os
import numpy as np
import random
import json
import logging
import logging.config
import torch
from utils.arguments_gen_reader import ModelArguments, DataTrainingArguments, OurTrainingArguments
from utils.dataloader_reader import GenerationDataModule
from model.extraction_trainer import GenerationTrainer
from transformers import HfArgumentParser, set_seed, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig
from datasets import load_metric

seed = 104
random.seed(seed) # python random seed 고정
np.random.seed(seed) # numpy random seed 고정
torch.manual_seed(seed) # torch random seed 고정
torch.cuda.manual_seed_all(seed)

config = json.load(open("./utils/log/logger.json"))
config['handlers']['file_debug']['filename'] = "./utils/log/generation/debug.log"
config['handlers']['file_error']['filename'] = "./utils/log/generation/error.log"
logging.config.dictConfig(config)
logger = logging.getLogger(__name__)


def main():
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, OurTrainingArguments) # arguement 쭉 읽어보면서 이해하기
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
    quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            # bnb_4bit_compute_dtype=torch.float16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        quantization_config=quantization_config
    )
    lora_config = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.1, bias="none", task_type="CAUSAL_LM", modules_to_save=None,)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    logger.info(model)
    
    # 데이터 불러오기 및 전처리
    dm = GenerationDataModule(data_args, training_args, tokenizer) 
    train_dataset, eval_dataset = dm.get_processing_data()
    # Trainer 초기화
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side='right'
    metric = load_metric('squad')
    # SFTTrainer는 trainer가 알아서 dataset을 tokenize함
    trainer = GenerationTrainer(
        model=model,
        dataset_text_field="prompt",
        max_seq_length=training_args.max_seq_length,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=lora_config,
        tokenizer=tokenizer,
        packing= training_args.packing,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        metrics=metric
        # preprocess_logits_for_metrics=post_processing_function, # metric을 계산하기 위한 후처리
        # compute_metrics=compute_metrics, # metric 계산 코드
        # class 안에 있으니까 굳이 인자로 넣을 필요가 없음
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

    # Evaluation -> 나중에 구현
    logger.info("***** Evaluate *****")
    metrics = trainer.evaluate()

    metrics["eval_samples"] = len(eval_dataset)

    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)


if __name__ == "__main__":
    main()