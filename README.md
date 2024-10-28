#🔥 네이버 AI Tech NLP 8조 The AIluminator 🌟
## Level 2 Project - Open-Domain Question Answering

## 목차
1. [프로젝트 소개](#1-프로젝트-소개)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [Installation and Quick Start](#3-installation-and-quick-start)
4. [팀원 소개](#4-팀원-소개)
5. [프로젝트 진행](#5-프로젝트-진행)
6. [리더보드 결과](#6-리더보드-결과)

## 1. 프로젝트 소개
(1) 주제 및 목표
- 부스트캠프 AI Tech NLP 트랙 level 2 MRC
- 주제 : ODQA (Open-Domain Question Answering)    
      ODQA 데이터셋을 활용해 질문에 맞는 정답을 예측  <br>

(2) 평가지표
- 주 평가지표 : Exact Match (모델의 예측과 실제 답이 정확하게 일치할 때만 점수가 주어짐) <br>
- 참고용 : F1 score (모델의 예측과 실제 답에 겹치는 부분이 있으면 부분점수가 주어짐) <br>

(3) 개발 환경 <br>
- GPU : Tesla V100 * 4 <br>

(4) 협업 환경
|**Tool**|**Description**|
|:-:|-|
|**GitHub**|- Task 별 issue 생성<br>- 담당한 issue에 대한 branch 생성 후 PR & main에 merge|
|**Slack**| - GitHub과 연동해서 레포지토리에 업데이트 되는 내용 실시간으로 확인<br>- 허들을 이용한 회의 및 결과 공유 |
|**Notion**| - 타임라인 정리<br>- 칸반보드를 이용한 task 관리 |
|**Zoom**| - 진행상황 공유 |
|**WandB**| - Sweep을 통한 하이퍼 파라미터 최적화 |

## 2. 프로젝트 구조
```sh
.
├── model
│   ├── fine_tune_gnn.py
│   ├── fine_tune_sts.py
│   └── SimCSE.py
├── preprocessing
│   ├── modeling
│   │   └── Clustering.ipynb
│   ├── DataCleaning.ipynb
│   ├── EDA.ipynb
│   ├── v1_downsampling.ipynb
│   ├── v2_augmentation_biassed.ipynb
│   ├── v3_augmentation_uniform.ipynb
│   └── v4_augmentation_spellcheck.ipynb
├── resources
│   ├── log
│   └── raw
│       ├── dev.csv
│       ├── sample_submission.csv
│       ├── test.csv
│       └── train.csv
├── utils
│   ├── data_module.py
│   ├── ensemble_module.py
│   └── helpers.py
├── inference.py
├── run_ensemble.py
├── train_graph.py
├── train.py
├── train_unsup_CL.py
```

## 3. Installation and Quick Start

## Installation and Quick Start
**Step 1.** 프로젝트에 필요한 모든 dependencies는 `requirements.txt`에 있고, 이에 대한 가상환경을 생성해서 프로젝트를 실행
```sh
# 가상환경 만들기
$ python -m venv .venv

# 가상환경 켜기
$ . .venv/bin/activate

# 제공되는 서버 환경에 따라 선택적 사용
$ export TMPDIR=/data/ephemeral/tmp 
$ mkdir -p $TMPDIR

# 필요 라이브러리 설치
$ pip install --upgrade pip
$ pip install -r requirements.txt
```

**Step 2.** Sparse embedding + extracion based reader model 실행 방법
 ```sh
# 먼저 reader를 학습 / parameter를 변경 하고 싶다면 utils/arguments_reader.py 수정
$ python train_extraction_reader.py

# 학습이 완료되면 utils/arguments_inference.py의 model_name_or_path를 수정 후 실행
$ python inference.py
```

**Step 3.** Inference 실행 방법
``` sh
# 먼저 sparse/dense 임베딩 벡터 저장
$ cd/database
$ python get_embedding_vec.py

# inference 실행
$ python inference.py
```

**Optional.** 원격 연결 끊어졌을 때도 돌아갈 수 있도록 Tmux 사용을 권장
```sh
# 새로운 세션 생성
$ tmux new -s (session_name)

# 세션 목록
$ tmux ls

# 세션 시작하기 (다시 불러오기)
tmux attach -t (session_name)

# 세션에서 나가기
(ctrl + b) d

# 특정 세션 강제 종료
$ tmux kill-session -t (session_name)
```


## 4. 팀원 소개
|김동한|김성훈|김수아|김현욱|송수빈|신수환|
|:--:|:--:|:--:|:--:|:--:|:--:|
|<img src="https://github.com/user-attachments/assets/c7d1807e-ef20-4c82-9a88-bc0eb5a700f4" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/62829d6a-13c9-40dd-807a-116347c1de11" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/5933a9e6-b5b8-41df-b050-c0a89ec19607" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/c90f4226-3bea-41d9-8b28-4d6227c1d254" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/65a7e762-b018-41fc-88f0-45d959c0effa" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/8d806852-764d-499b-a780-018b6cf32b8d" width="100" height="100" />|
<!-- |[Github]()|[Github]()|[Github](https://github.com/tndkkim)|[Github](https://github.com/hwk9764)|[Github](https://github.com/suvinn)|[Github]()| -->

### 맡은 역할
|**Member**|**Team**|**Role**|
|:--|--|--|
|**김수아**|Model|**EDA**(label 분포 및 문장 길이 분석), **Data Cleanling**|
|**김현욱**|Data, Model|- **Generation Reader Modeling**(학습 및 추론)<br>- **EDA**(데이터 텍스트 퀄리티 분석)|
|**송수빈**|Model|**데이터 증강**(Downsampling/Sentence Swap/BERT-Mask Insertion/hanspell)|
|**김동한**|Data, Model|- **Extraction Reader Modeling**(학습 및 추론)<br>- **Extraction Reader 아키텍처 수정**(CNN Head)<br>- **Sparse Passage Retrieval**(Retrieval 결과 분석)<br>- **EDA**(데이터 토큰 개수 분포 분석)|
|**김성훈**|Data, Model|**Model Exploration & Training**, **Modeling**(Second-stream with GNN, Contrastive Learning, Soft Voting Ensemble), **Model Tuning**(deliciouscat/kf-deberta-base-cross-sts, snunlp/KR-ELECTRA-discriminator), **코드 모듈화**|
|**신수환**|Data, Model|**Model Training & Tuning**(RoBERTa, T5, SBERT), **모델 경량화**(Roberta-large with deepspeed), **Modeling**(Clustering)|
<br>

## 5. 프로젝트 진행
<img src="https://github.com/user-attachments/assets/daf4c4ea-2463-426c-9964-939b5c793937"/>
framework 설명<br><br>

| Task | **Task Description** |
| --- | --- |
| **EDA** | 데이터의 특성을 살펴보기 위해 중복 데이터 확인, 토큰 개수 분포, 데이터 퀄리티 체크 등 시각화 및 분석 |
| **Retrieval** | BM25, DPR Retrieval 기법 구현 및 실험 |
| **Reader Model Exploration** | Reader Model로 사용하기 적합한 pre-trained model 실험 및 선정 |
| **Reader Model** | Transfer Learning <br> CNN Head <br> Cleaning|
| **Post-Processing** | 후처리 <br> 모델 다양성 체크 <br> 앙상블 |


## 6. 원본 데이터 탐색
### 사용한 데이터셋
데이터는 train_dataset, test_dataset의 2개의 DatasetDict로 되어있으며 각 파일의 구성은 다음과 같다. <br>
| 분류(디렉토리 명)|세부 분류|샘플 수|용도|공개여부|
|:-:|:-:|:-:|:-:|:-:|
|train|train|3,952|학습용|모든 정보 공개(id, question, context, answers, document_id, title)|
|-|valid|240|학습용|모든 정보 공개(id, question, context, answers, document_id, title)|
|test|valid|240 (Public)|제출용|id, question 만 공개|
|-|-|360 (Private)|제출용|id, question 만 공개|
|Wiki, Doc|corpus|60,613|제출용|모든 정보 공개 (text, corpus_source, url, title, document_id)


**title** : context 제목 <br>
**context** : 문단 <br>
**question** : 질문 <br>
**id** : context, question 쌍 고유 id <br>
**answers** : {answer_start: 문단 내 시작위치, text: 정답} <br>
**document_id** : 문단 id <br>

### 중복 데이터 확인
- Query-Passage 쌍 데이터 : context 기준, Train 2,761개(1,191↓) / Valid 230개(10↓)
- Wiki. Doc : 56,737개(3,801↓) <br> <br>

### 토큰 별 분포
- Query-Passage 쌍 데이터의 Text들에 대한 토큰을 세 종류의 Tokenizer (BPE, SentenecPiece, WordPiece)를 통해 확인
<img src="https://github.com/user-attachments/assets/4dfd39a3-d18d-483c-b1f4-9fe0fe3ba02f"/>
- Wiki. Doc 의 Text들에 대한 토큰을 세 종류의 토크나이저 (BPE, SentenecPiece, WordPiece)를 통해 확인
<img src="https://github.com/user-attachments/assets/eb02949a-6a3b-4668-b1db-2c9e9b434702"/>

- Train, Validation set 모두 최대 1,500 이하이며 비슷한 분포를 가짐
- Wiki set : boxplot을 통해 outlier가 존재함을 확인 <br><br>
<img src="https://github.com/user-attachments/assets/a0a0c7e7-6983-4a5d-b73b-2a10779e202d"/> <br>
<br>

### 데이터 퀄리티 체크
- 각 데이터셋에서 한글이 아닌 문자(영어, 한자, url, html 태그, 특수문자 등) 개수 파악, text에 한번이라도 포함되면 count <br>
<img src="https://github.com/user-attachments/assets/27ff50e7-5bc5-495c-8f4f-b2947a39e14c"/>

## 7. Modeling
### Model Exploration
- 베이스라인으로 주어진 코드를 기반으로 HuggingFace의 model 허브에서 “question answering”으로 필터링한 후 Klue로 사전학습 되지 않은 모델에 대해 기초 성능 평가를 진행. <br>
Reader : Query-Passage 쌍 데이터로 학습 한 Extraction based Reader
<img src="https://github.com/user-attachments/assets/6c82984b-76b0-4e02-964a-6e3b67cc977c"/>



### Model Tuning
Retrieval Tuning : TF-IDF <br>
- Retrieval 단계에서 가져오는 문서의 개수(k)가 모델이 정답을 맞출 확률을 높이는 데 미치는 영향을 분석
<img src="https://github.com/user-attachments/assets/6c82984b-76b0-4e02-964a-6e3b67cc977c"/>
<br>
Parameter Tuning <br>
- 과적합을 방지하기 위해 Loss function에 L2 penalty를 좀 더 많이 적용
- 적절한 답변의 길이를 찾기
<img src="https://github.com/user-attachments/assets/ead0a1f5-0945-4a4b-a361-6c6e626ca63f"/>

### Contrastive Learning
**모델링 설명**
- SimCSE (Gao et al., 2021)은 Contrastive Learning을 sentence embedding에 적용한 최초의 논문으로 그 해 STS Task에서 SOTA의 성능을 달성
- Contrastive Learning이 negative instance를 분리하므로 uniformity를 향상시켜 anisotropy를 완화 시키는 것을 실험적으로 증명하였고, 이 요소들이 sentence embedding에 중요한 factor임을 확인
- 이에 SimCSE의 Contrastive Learning을 본 프로젝트의 적용하고자 하였으며, 시간 관계 상 Unsupervised SimCSE를 적용하였음

**결과 분석**
- 정량적 평가 : Validation Pearson 기준으로는 Second-stream with GNN과 동일하지만 Public Pearson 결과는 더 향상된 것을 확인
- 정성적 평가 : deberta with CL 모델은 5.0이상으로 라벨을 예측하는 경우가 84개 더 많은 것을 확인할 수 있음, 이를 통해 Contrastive Learning이 효과적으로 적용되어 보다 잘 분류하는 것을 확인 
- 학습 cost와 성능 향상 폭을 생각했을 때는 Contrastive Learning이 유의미한 것으로 판단되어 제안하는 모델을 사용하는 것으로 결정

| Model | Validation Pearson | Public Pearson |
| --- | --- | --- |
| deliciouscat/kf-deberta-base-cross-sts | 0.926 | 0.9110 |
| deliciouscat/kf-deberta-base-cross-sts + GNN | 0.929 | 0.9164 |
| deliciouscat/kf-deberta-base-cross-sts + CL | 0.929 | 0.9190 |

### Clustering
**모델링 설명**
- STS(Semantic Textual Similarity) 문제에서 모델은 (1) 두 벡터의 코사인 유사도 값이 크나 문장 간의 label이 작은 경우나 (2) 코사인 유사도 값이 작으나 문장간의 label이 큰 경우에 대해 어려워 할 것이라고 가정
- 이러한 문제들을 잘 해결할 수 있도록 위 두가지 케이스를 고려한 클러스터를 만들어 모델을 학습하고자 함
- `train.csv`의 `sentence_1`, `sentence_2` 를 사전 학습된 encoder 모델로 임베딩 후, 각 임베딩 벡터 간의 코사인 유사도를 계산하고 label과 코사인 유사도 간 상관관계를 기준으로 두개의 클러스터로 분류

**결과 분석**
- 정량적 평가 : Public Pearson 측정 결과, 0.9177의 성능을 확인
- 정성적 평가 : 맞춤법, 불용어 등에 따라 코사인 유사도의 영향을 많이 받는 것을 확인하여 Cleaning 전처리를 추가하기로 결정

### Soft Voting Ensemble
**모델링 설명**
- Soft Voting은 앙상블 학습에서 사용되는 기법으로, 여러 개의 분류 모델의 예측 결과를 평균하여 최종 예측을 만드는 방법
- 각 모델이 예측한 logit을 평균하거나 가중 평균하여 최종 logit 결정
- Valid score 기반 가중 평균
    - 앙상블할 모델의 valid score만큼 비율로 곱하여 가중 평균
    - e.g) model A : 0.9 / model B : 0.8 인 경우
        
        $$
        \frac {A_i \times0.9+B_i\times 0.8} {0.9+0.8}
        $$
        
- Min-Max 정규화 가중 평균
    - 단순 Valid score 기반 가중 평균시, 대부분의 앙상블 대상 모델이 92~93의 유사한 valid score를 가짐
    - 더 좋은 성능을 가진 모델과 그렇지 않은 모델간 차이를 줄 수 있게 적합한 가중치 정규화의 필요성을 느낌
    - 앙상블할 모델의 valid score를 0.8~1.2 값으로 scaling하여 가중평균
    - 0.8~1.2로 scaling할때의 min-max 정규화 수식

$$
0.8+\frac {x-x_{min}} {x_{max}-x_{min}}\times(1.2-0.8)
$$


**결과 분석**
- Data Aaugmentation 진행한 결과에 따른 4가지 version의 train data와 Model exploration&Modeling을 거쳐 선정된 model에 다양한 조합으로 실험하여 최적의 성능 도출
- **각 기법마다 best case에 대해서 비교해본 결과 min-max 평균을 취한 case가 가장 높은 92.98의 public pearson 값을 가지는 것을 확인하고 이를 최종 리더보드에 제출**

| 모델 | 활용 기법 | Validation Pearson | Min-Max 정규화 가중 평균 |
| --- | --- | --- | --- |
| deliciouscat/kf-deberta-base-cross-sts | raw + Contrastive Learning | 0.930 | 1.111 |
| deliciouscat/kf-deberta-base-cross-sts | raw + Cleaning | 0.930 | 1.111 |
| sorryhyun/sentence-embedding-klue-large | Augmentation v2 | 0.923 | 0.800 |
| snunlp/KR-ELECTRA-discriminator | Augmentation v2 | 0.932 | 1.200 |
| snunlp/KR-ELECTRA-discriminator | Augmentation v3 | 0.930 | 1.111 |

## 6. 리더보드 결과
Leader Board에서 Pearson을 비교하였을 때 0.105가 올라, 대회에 참여한 16팀 중 가장 많이 순위가 상승하였고 최종 4위를 기록함

**Public Leader Board 순위**

<img src="https://github.com/user-attachments/assets/d2d828ff-e443-4a9a-a111-d8e4b8453cc8"/>


**Private Leader Board 순위**

<img src="https://github.com/user-attachments/assets/3c87d052-9734-4c90-8c81-9c186c877bdf"/>

