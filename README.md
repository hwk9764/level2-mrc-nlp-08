# 🔥 네이버 AI Tech NLP 8조 The AIluminator 🌟
## Level 2 Project - Open-Domain Question Answering

## 목차
1. [프로젝트 소개](#1-프로젝트-소개)
2. [Installation and Quick Start](#2-installation-and-quick-start)
3. [프로젝트 진행](#3-프로젝트-진행)
4. [최종 결과]()

# 1. 프로젝트 소개
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

(5) 팀원 소개
|김동한|김성훈|김수아|김현욱|송수빈|신수환|
|:--:|:--:|:--:|:--:|:--:|:--:|
|<img src="https://github.com/user-attachments/assets/c7d1807e-ef20-4c82-9a88-bc0eb5a700f4" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/62829d6a-13c9-40dd-807a-116347c1de11" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/5933a9e6-b5b8-41df-b050-c0a89ec19607" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/c90f4226-3bea-41d9-8b28-4d6227c1d254" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/65a7e762-b018-41fc-88f0-45d959c0effa" width="100" height="100" />|<img src="https://github.com/user-attachments/assets/8d806852-764d-499b-a780-018b6cf32b8d" width="100" height="100" />|
<!-- |[Github]()|[Github]()|[Github](https://github.com/tndkkim)|[Github](https://github.com/hwk9764)|[Github](https://github.com/suvinn)|[Github]()| -->

|**Member**|**Team**|**Role**|
|:--|--|--|
|**김수아**|Model|**EDA**(label 분포 및 문장 길이 분석), **Data Cleanling**|
|**김현욱**|Data, Model|- **Generation Reader Modeling**(학습 및 추론)<br>- **EDA**(데이터 텍스트 퀄리티 분석)|
|**송수빈**|Model|**데이터 증강**(Downsampling/Sentence Swap/BERT-Mask Insertion/hanspell)|
|**김동한**|Data, Model|- **Extraction Reader Modeling**(학습 및 추론)<br>- **Extraction Reader 아키텍처 수정**(CNN Head)<br>- **Sparse Passage Retrieval**(Retrieval 결과 분석)<br>- **EDA**(데이터 토큰 개수 분포 분석)|
|**김성훈**|Data, Model|**Model Exploration & Training**, **Modeling**(Second-stream with GNN, Contrastive Learning, Soft Voting Ensemble), **Model Tuning**(deliciouscat/kf-deberta-base-cross-sts, snunlp/KR-ELECTRA-discriminator), **코드 모듈화**|
|**신수환**|Data, Model|**Model Training & Tuning**(RoBERTa, T5, SBERT), **모델 경량화**(Roberta-large with deepspeed), **Modeling**(Clustering)|
<br>


# 2. Installation and Quick Start
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

**Step 2.** Pre Processing 실행
 ```sh
# 작업환경 변경
$ cd pre_process

# 다음 주피터를 따라가며 KorQuAD 1.0 데이터 증강
$ data_augment_korquadv1.ipynb

# 다음 주피터를 따라가며 AIHub 데이터 증강
$ data_augment_aihub.ipynb

# 다음 주피터를 따라가며 DPR retrieval을 학습하기 위한 데이터 만들기
$ generate_DPR_dataset_korquad.ipynb
```

**Step 3.** DPR 모델 학습

**utils/arguments_dpr.py** 에서 DPR 학습을 위한 파라미터 변경
- model : 원하는 사전 학습된 모델 불러오기
- train_data : generate_DPR_dataset_korquad.ipynb 에서 생성한 데이터 경로
- valid_data : generate_DPR_dataset_korquad.ipynb 에서 생성한 데이터 경로
- q_output_path : Query embedding 모델 저장할 경로
- c_output_path : Context embedding 모델 저장할 경로
```sh
# ./level2-mrc-nlp-08 경로에서 실행
$ python train_dpr.py
```



**Step 4.** Retrieval를 위한 사전처리 진행

**database/python get_embedding_vec.csv** : BM25 모델 및 DPR의 embedding vector 저장
- model : 학습된 context embedding 모델 경로
- wiki_path : Wiki.doc 데이터 경로 
- valid_data : Query-Passage 쌍 데이터의 검증 데이터 경로
- save_path : Embedding vector 저장 경로

**test_retrieval.py**
- model : 학습된 query embedding 모델 경로
- valid_data : Query-Passage 쌍 데이터의 검증 데이터 경로
- faiss_path : **database/python get_embedding_vec.csv** 에서 실행한 save_path 경로
- bm25_path : **database/python get_embedding_vec.csv** 에서 실행한 save_path 경로
- context_path : **database/python get_embedding_vec.csv** 에서 실행한 save_path 경로

**test_retrieval_inference.py**
- model : 학습된 query embedding 모델 경로
- test_dataset : Query-Passage 쌍 데이터의 테스트 데이터 경로
- faiss_path : 위와 동일
- bm25_path : 위와 동일
- context_path : 위와 동일

```sh
# 작업환경 변경
$ cd database

# 다음 코드를 실행하여 embedding vector 추출
$ python get_embedding_vec.csv

# BM25 및 DPR 성능 확인
$ cd ..
$ python test_retrieval.py

# Inference 시 사용할 retireve 된 데이터 생성
$ python test_retrieval_inference.py
```


**Step 5.** Reader 학습

**utils/arguments_extraction_reader.py**에서 extracion based model 학습을 위한 파라미터 변경
- model_name_or_path : 사전 학습된 모델 불러오기
- dataset_name : Query-Passage 쌍 데이터나 증강된 데이터 경로로 변경
- output_dir : 학습된 모델 및 평가 결과 저장 경로

```sh
# 다음 코드를 실행하여 extraction based model 학습
$ python train_extraction_reader.py

# 프로젝트 때는 사용하지 않았지만 generation based model 학습, 파라미터 변경은 위와 동일
$ python train_generation_reader_Seq2SeqLM,.py
$ python train_generation_reader_CausalLM,.py
```



**Step 6.** Inference 실행

**utils/arguments_inference.py**에서 inference 할 extraction based 모델의 파라미터 변경
- model_name_or_path : 학습이 완료된 모델 불러오기
- output_dir : Inference 결과 저장 경로

```sh
# 코드 50번째 줄에서 retireve 된 데이터 불러오는 경로 원하는 것으로 변경하면서 사용
$ python inference.py
```


**Step 7.** 앙상블 실행
```sh
# train_extraction_reader 실행 시 생성되는 predictions.json 값들로 상관분석 분석
$ correlation_exp.ipynb

# 상관분석을 통해 사용할 모델 선택 되었다면 그 모델들로 inference 시 생성된 nbest_predictions.json 파일들로 앙상블 진행 / 두가지 버전 모두 사용 가능
$ ensemble_v1.ipynb
$ ensemble_v2.ipynb

```

# 3. 프로젝트 진행
<img src="https://github.com/user-attachments/assets/daf4c4ea-2463-426c-9964-939b5c793937"/>


| Task | **Task Description** |
| --- | --- |
| **EDA** | 데이터의 특성을 살펴보기 위해 중복 데이터 확인, 토큰 개수 분포, 데이터 퀄리티 체크 등 시각화 및 분석 |
| **Retrieval** | BM25, DPR Retrieval 기법 구현 및 실험 |
| **Reader Model Exploration** | Reader Model로 사용하기 적합한 pre-trained model 실험 및 선정 |
| **Reader Model** | Transfer Learning <br> CNN Head <br> Cleaning|
| **Post-Processing** | 후처리 <br> 모델 다양성 체크 <br> 앙상블 |


## EDA
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

## 베이스라인 모델
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

## Reader 개선
### Transfer Learning
- **배경**
    - 2009년 ICML에서 발표된 논문의 "Curriculum learning"에 따라 Transfer Learning을 효과적으로 적용하고자 함
- **Curriculum Learning**
    - 사람과 유사하게 기계에 Curriculum을 만들어 주어 더 쉬운 데이터부터 학습시키고 차츰 어려운 데이터를 순차적으로 학습시키는 방식을 의미
    - ex) Shape Recognition Task를 학습시킬 때, 쉬운 데이터로 원, 정사각형 등을 학습시키고 어려운 데이터로 타원, 직사각형 등 학습 <br>
       <img src="https://github.com/user-attachments/assets/36e44c0c-432f-4099-abc6-d2eecab27826"/>
- **오픈 데이터셋 추가**
    -  KoQuAD 1.0 : 한국어 MRC를 위해 만든 데이터셋
        -  해당 데이터셋은 Wikipedia article의 일부 하위 영역에 대한 질문과 답변으로 이루어져 있음
        -  Stanford Question Answering Dataset(SQuAD) v1.0과 동일한 방식으로 구성
        ```
            Context = "덴노지 역 구내에서는 야마토지 선과 평면 교차하여 단선으로 운전하는 한와 선과의 단락선을 복선화하는 공사가 이루어져
         2008년 3월 15일 시간표 개정시부터 사용을 개시하였다. 대낮 시간대의 덴노지 역 ~ 와카야마 역 간의 쾌속과 JR 난바 역
         발착의 간쿠 쾌속이 오사카 순환선과 직통 운전하는 간쿠·기슈지 쾌속에 통합되어 같은 시간대의 간쿠·기슈지 쾌속이 매시
         3편으로 증발되었다. 이 복선화는 야마토지 선의 시간표 혼란이 큰 영향을 끼치는 것을 막는 효과를 가지고 있다.
        또 아침 출근 시간대에 오사카 순환선 내 각역에 정차하는 직통 쾌속도 운전이 개시되었다.
        전반적으로 히네노 역 이북에서는 8량 편성의 쾌속 열차가 대폭 증가되었기 때문에 난카이 전기 철도 본선과 수송력에서 차이가 나게 되었다."
            
            Question = "야마토지 선의 시간표가 더 간단해지게 된 이유는 무엇일까?",  
            Answer = "복선화"
        ```
    - ai-hub 데이터 추가 : 뉴스 기사 기계독해 데이터
      - 해당 데이터셋은 뉴스 기사 context와 Question과 Answer로 구성
        - 추출형 : 지문 내에서 답을 도출할 수 잇는 질문과 그 답변으로 이루어진 질의응답
        - **추론형** : 지문 내 정보를 이용하여 답을 유추하도록 하는 질문과 근거, 답변으로 이루어진 질의응답 셋으로, 답에 대해 지문에 직접적으로 명시되지 않은 내용으로 질문을 제작한다는 점에서 추출형과 구분, 추론형은 일반 상식의 조합이나 연역 추론 등의 방식을 활용하여 제작 <br>

    → 이 중 Curriculum Learning에 반영할 난이도를 생각했을 때, 추론형이 더 학습하기 어려운 데이터이므로 추론형을 선택하여 전이 학습에 사용

- **평가**
    - 증강된 데이터를 활용하였을 때, validation에서는 크게 개선되지 않았지만, **F1-score와 Public Score가 많이 향상된 것을 확인**
<img src="https://github.com/user-attachments/assets/197ea107-b49b-45e7-b0ff-aab25ae0dff6"/>

### CNN Layer
- **배경**
    - 기존 Pre-trained 모델의 출력은 주로 Linear Layer를 거쳐 처리되지만, 이는 인접한 토큰들 간의 연관된 정보를 충분히 학습하기에 부족
    - 삼성 SDS에서 제안한 KorQuAD 1.0 성능 개선 방안1) 에 따르면, 1D Convolution을 추가하면 Kernel Size에 해당하는 인접 토큰들의 임베딩 벡터 간의 연관성을 학습하여 보다 풍부한 문맥 정보를 얻을 수 있다는 것을 확인
- **1차원 Convolution**
    - 1D Convolution은 2D Convolution 와 달리 커널의 높이가 임베딩 차원과 같고, 너비만 kernel size로 설정
    - 결과적으로, 시퀀스의 각 위치에서 해당 범위 내 단어들의 정보를 종합한 특징 벡터가 생성
      <img src="https://github.com/user-attachments/assets/2d0dc473-19c9-49d3-99eb-a107ed14e974"/> 
- **CNN Layer 적용 구조**
    - Kernel Size 3과 1을 갖는 Conv 레이어와 Residual Connection을 통해 인접 토큰 간의 연관성 학습과 동시에 모델이 깊어짐에 따른 학습 부담을 완화
      <img src="https://github.com/user-attachments/assets/d7016bf4-fc64-48a0-8432-c88a184995d5"/>
- **평가**
    - kpfbert-korquad-1 : Eval_F1 : 4.66 , Eval_EM : 5.00 상승
    - roberta-large-qa-korquad-v1: Eval_F1 : 2.84 , Eval_EM : 2.9 상승, Public_EM : 7.08, public_F1 : 7.5 감소 
    
    → 학습 성능이 높아졌지만 오버피팅 가능성이 높아짐을 확인
<img src="https://github.com/user-attachments/assets/fdf81d6e-0bd4-4077-8fc6-cb36f77c15e9"/>

### Cleaning
- **배경**
    - 대회에서 주어진 Query-Passage 쌍 데이터의 context가 KQUAD, AI Hub 등 데이터셋에 비해 노이즈가 많이 끼어 있어, 해당 부분에서 노이즈를 줄일 필요성을 느낌
    - 해당 데이터의 context가 증강된 데이터와 유사해질수록, 증강된 데이터로 학습이 더 용이해 질 것이라고 가정
- **방안**
    - Context에서 답이 지워지지 않도록 노이즈를 제거하는 코드를 작성
    - 해당 코드를 대회에서 주어진 train 데이터셋, validation 데이터셋에 적용하고, retriever를 통해 불러오는 context 들에 대해서도 적용함
- **예시**
    - **날짜 정보 제거**:
        - 원본: "낸다. 이러한 방식으로 단원제가 빠지기 쉬운 함정을 미리 방지하는 것이다.날짜=2017-02-05"
        - 처리 후: "낸다. 이러한 방식으로 단원제가 빠지기 쉬운 함정을 미리 방지하는 것이다."
    
    - **인용문 제거**:
        - 원본: "연장시킬 뿐이라고 했다.문학의 숲 고전의 바다 元老의 誤判] 조선일보 2004.03.05</ref>"
        - 처리 후: "연장시킬 뿐이라고 했다."
    
    - **섬네일 정보 제거**:
        - 원본: "섬네일|left|페데리코 다 몬테펠트로의 구비오 스투디올로의 모습.\n\n"
        - 처리 후: "페데리코 다 몬테펠트로의 구비오 스투디올로의 모습."
    
    - **질문-답변 불일치 수정**:
        - 원본 질문: "스타필드 고양점이 개업한 날은?"
        - 원본 답변: "3호점인 스타필드 고양"
        - 수정 후 답변: "2017년 8월 24일" (본문에서 실제 개업일을 찾아 수정)

    - **답변 형식 통일**:
        - 이외에 Answer 형식을 통일하려고 시도하였으나, 규칙성을 찾아내지 못하였음

- **평가**
    - Clean 되기 이전의 데이터를 사용했을 때에 비해 Clean 데이터를 사용하였을 때 모델의 성능이 전반적으로 오른다는 것을 확인
<img src="https://github.com/user-attachments/assets/ab4f25b9-5571-477e-9aed-bf0234154905"/>

- **결과 분석**
    - Data Aaugmentation 진행한 결과에 따른 4가지 version의 train data와 Model exploration&Modeling을 거쳐 선정된 model에 다양한 조합으로 실험하여 최적의 성능 도출
    - **각 기법마다 best case에 대해서 비교해본 결과 min-max 평균을 취한 case가 가장 높은 92.98의 public pearson 값을 가지는 것을 확인하고 이를 최종 리더보드에 제출**

