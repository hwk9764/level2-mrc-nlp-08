import pickle
import logging
import argparse
import pandas as pd
from tqdm import tqdm
from kiwipiepy import Kiwi
from transformers import AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi

logger = logging.getLogger("mrc")
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s: [ %(message)s ]','%m/%d/%Y %I:%M:%S %p')
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


def main(args):
    if args.retrieval == 'tf-idf':
        # wiki 데이터 불러오기
        if args.wiki_data_path.split('.')[-1] == 'csv':
            wiki = pd.read_csv(args.wiki_data_path)
            logger.info(f"Load wiki dataset : {len(wiki)}개")
            logger.info(f"Item in wiki {wiki}")
        else:
            raise Exception("wiki 데이터 경로를 다시 확인 해주세요")
        
        # tokenizer 불러오기
        if args.tokenizer == 'kiwi':
            tokenizer = Kiwi()
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        
        # context 불러오기
        if args.tokenizer == 'kiwi':
            NotImplemented
        else:
            wiki['title_text'] = wiki['title'] + ' ' + wiki['text']
            contexts = wiki['title_text'].tolist()

        # Vectorizer 불러오고 실행
        tfidfv = TfidfVectorizer(tokenizer=tokenizer.tokenize, ngram_range=(1, 2), max_features=50000)
        p_embedding = tfidfv.fit_transform(contexts)
        
        # 저장
        with open("sparse_embedding_tfidv.bin", "wb") as file:
            pickle.dump(p_embedding, file)
        with open("tfidv.bin", "wb") as file:
            pickle.dump(tfidfv, file)
        logger.info("tf-idf embedding pickle saved")
        
    elif args.retrieval == 'bm25':
        # wiki 데이터 불러오기
        if args.wiki_data_path.split('.')[-1] == 'csv':
            wiki = pd.read_csv(args.wiki_data_path)
            logger.info(f"Load wiki dataset : {len(wiki)}개")
            logger.info(f"Item in wiki {wiki}")
        else:
            raise Exception("wiki 데이터 경로를 다시 확인 해주세요")
        
        # tokenizer 불러오기
        if args.tokenizer == 'kiwi':
            tokenizer = Kiwi()
        else:
            tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        
        # context 불러오기
        if args.tokenizer == 'kiwi':
            NotImplemented
        else:
            wiki['title_text'] = wiki['title'] + ' ' + wiki['text']
            contexts = wiki['title_text'].tolist()
            contexts = [tokenizer.tokenize(doc) for doc in tqdm(contexts, desc="BM25 Tokenizing")]
            
        # Vectorizer 불러오고 실행 / bm25 파라미터 자유롭게 변경
        bm25 = BM25Okapi(contexts, k1=1.5, b=0.75)
        with open("bm25.bin", 'wb') as f:
            pickle.dump(bm25, f)
        
    elif args.retrieval == 'dense':
        NotImplemented


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval", type=str, default="bm25", help="tfidf, bm25, dense")
    parser.add_argument("--wiki_data_path", type=str, default="../resources/wiki/remove_duplicates.csv")
    parser.add_argument("--tokenizer", type=str, default="monologg/kobigbird-bert-base", help="huggingface 모델명 혹은 kiwi(한국어 형태소 분석기) 선택")    
    args = parser.parse_args()
    
    main(args)