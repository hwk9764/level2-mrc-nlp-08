# coding=utf-8
# Copyright 2020 The HuggingFace Team All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Question-Answering task와 관련된 'Trainer'의 subclass 코드 입니다.
"""
import torch
import datasets
from transformers import Trainer
from transformers.trainer_utils import PredictionOutput, EvalLoopOutput
from trl import SFTTrainer

# Huggingface의 Trainer를 상속받아 QuestionAnswering을 위한 Trainer를 생성합니다.
class QuestionAnsweringTrainer(Trainer):
    def __init__(self, *args, eval_examples=None, post_process_function=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.post_process_function = post_process_function

    def evaluate(self, eval_dataset=None, eval_examples=None, ignore_keys=None):
        # preprocess한 데이터
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        # 원본 데이터?
        eval_examples = self.eval_examples if eval_examples is None else eval_examples
    
        # 일시적으로 metric computation를 불가능하게 한 상태이며, 해당 코드에서는 loop 내에서 metric 계산을 수행합니다.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
                # self.args.prediction_loss_only
                # metrics가 없으면 loss만 내보내고 있으면 pred, label, metric까지 함께 내보냄
                # 여기서 metric 계산 안하고 eval_loss만 받음(metric key에 존재)
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(
                eval_examples, eval_dataset, output.predictions, self.args
            )
            metrics = self.compute_metrics(eval_preds)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics

    def predict(self, test_dataset, test_examples, ignore_keys=None):
        test_dataloader = self.get_test_dataloader(test_dataset)

        # 일시적으로 metric computation를 불가능하게 한 상태이며, 해당 코드에서는 loop 내에서 metric 계산을 수행합니다.
        # evaluate 함수와 동일하게 구성되어있습니다
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        try:
            output = self.prediction_loop(
                test_dataloader,
                description="Evaluation",
                # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is None or self.compute_metrics is None:
            return output

        if isinstance(test_dataset, datasets.Dataset):
            test_dataset.set_format(
                type=test_dataset.format["type"],
                columns=list(test_dataset.features.keys()),
            )

        predictions = self.post_process_function(
            test_examples, test_dataset, output.predictions, self.args
        )
        return predictions
    
    
class GenerationTrainer(SFTTrainer):
    def __init__(self, *args, metrics=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.metric_func = metrics
    
    # 학습 중 또는 학습 후 squad 매트릭으로 평가
    def evaluate(self):
        print('---------------evaluate---------------')
        # eval metric을 return. eval loss와 추가 metric
        eval_dataset = self.eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        
        ''' eval loss 쓰려면 '''
        # valid_dataset의 prediction들 뽑기 -> predictions, labels, metrics 각각을 logit 형태로 받음
        try:
            output = self.prediction_loop(
                eval_dataloader,
                description="Evaluation",
                # metric이 없으면 예측값을 모으는 이유가 없으므로 아래의 코드를 따르게 됩니다.
                # self.args.prediction_loss_only
                # metrics가 없으면 loss만 내보내고 있으면 metric까지 함께 내보내는?
                prediction_loss_only=True if self.metrics_func is None else None,
            )
            # metrics가 있으면 pred_ids, label_ids, metrics 없으면 
        except Exception as e:
            print(e)
            
        # 왜 하지
        if isinstance(eval_dataset, datasets.Dataset):
            eval_dataset.set_format(
                type=eval_dataset.format["type"],
                columns=list(eval_dataset.features.keys()),
            )

        if self.metric_func is not None:    # metric이 존재할 때
            # decoding
            eval_preds, eval_labels = self.post_process_function(output)
            # get eval score -> dict 형태로 return
            metrics = self.compute_metrics(eval_preds, eval_labels)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(
            self.args, self.state, self.control, metrics
        )
        return metrics
    
        '''eval_loss 없이 EM, f1만 쓰려면'''
        EM, f1 = [], []
        for i, batch in enumerate(eval_dataloader):
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=self.args.max_seq_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            print(inputs)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=30)
            preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            score = self.compute_metrics(preds, batch['answers'])
            
            EM.extend(score['exact_match'])
            f1.extend(scroe['f1'])
        
        return (sum(EM)/len(EM), sum(f1)/len(f1))
    
    # 모델 아웃풋을 decoding이나 뭘 해서 텍스트로 변환
    # 1) eval이나 inference에는 model.generate로 해도 될 것 같음
    # 2) train에서 eval을 하려면 어떻게 해야할지 고민해보기
    def post_process_function(self, output:EvalLoopOutput):
        print('---------------post_process_function---------------')
        preds = output['predictions']
        label_ids = output['label_ids']
        # logits to token_ids
        print(preds.shape, label_ids.shape)
        ...
        # trainer에서 loss 계산을 위해 pad token을 -100으로 해놓았는데 이를 다시 pad token으로 변환
        # 냅두면 decoding 불가
        preds[preds == -100] = 141643
        label_ids[label_ids == -100] = 141643
        
        # decoding -> token_ids to text
        pred_texts = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        label_texts = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        return pred_texts, label_texts
    
    # 텍스트화한 애들로 squad metric에 넣어서 값 구하기
    def compute_metrics(self, preds, labels):
        print('---------------compute_metrics---------------')
        # squad metric이 지원하는 형식에 맞추어 줌.
        # id : question-answer 쌍을 구별하기 위한 id
        # answer_start : generation에서는 비워두어도 됨.
        predictions = [{'prediction_text': pred, 'id': str(i)} for i, pred in enumerate(preds)]
        references = [{'answers': {'answer_start': [], 'text': [label]}, 'id': str(i)} for i, label in enumerate(labels)]
    
        result = self.metric.compute(predictions=predictions, references=references)
        return {
            "exact_match": result["exact_match"],
            "f1": result["f1"]
        }
    
    # predict -> infernece. 그냥 generate해서 내뱉기.
    # prediction_loop로 가능할까
    def predict(self, question):
        # question과 passage 모음 잘 합치기
        inputs = ...
        # 
        inputs = self.tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=self.args.max_seq_length)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(**inputs, max_new_tokens=30)
        pred = self.tokenizer.decode(outputs, skip_special_tokens=True)
        return pred