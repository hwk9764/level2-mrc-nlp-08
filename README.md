# 🔥 네이버 AI Tech NLP 8조 The AIluminator 🌟
## Level 2 Project - Open-Domain Question Answering

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
 ```sh
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
