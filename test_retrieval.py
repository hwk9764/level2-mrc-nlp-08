import os
import argparse
import pandas as pd
import pickle
import csv
from transformers import AutoTokenizer
from database.retrienval import TFIDFModel, BM25Model
from tqdm import tqdm

# Argument Parsing
def parse_arguments():
    parser = argparse.ArgumentParser(description="TF-IDF and BM25 Model Experiments")
    
    parser.add_argument("--tfidf_model_file", type=str, default="model/tfidf_model.pkl", help="TF-IDF model file path")
    parser.add_argument("--bm25_model_file", type=str, default="model/bm25_model.pkl", help="BM25 model file path")
    parser.add_argument("--tokenizer_file", type=str, default="model/bert_based_tokenizer.pkl", help="Tokenizer file path")
    parser.add_argument("--test_file", type=str, default="validation.csv", help="Test file path")
    parser.add_argument("--corpus_file", type=str, default="remove_duplicates.csv", help="Corpus file path")
    parser.add_argument("--k", type=int, default=5, help="Top-k documents to retrieve")
    parser.add_argument("--k1", type=float, default=1.5, help="bm25 k1 value")
    parser.add_argument("--b", type=float, default=0.75, help="bm25 b value")
    parser.add_argument("--model", type=str, choices=["TF-IDF", "BM25", "ALL"], default="ALL", help="Choose model to run: TF-IDF, BM25, or ALL")
    parser.add_argument("--output_file", type=str, default="results.csv", help="Output file to save the results")
    
    return parser.parse_args()

# 데이터 로드
def load_data(test_file, corpus_file):
    validation_df = pd.read_csv(test_file)
    questions = validation_df['morphs_question'].tolist()
    correct_document_ids = validation_df['document_id'].tolist()
    
    documents_df = pd.read_csv(corpus_file)
    documents = documents_df['morphs_context'].tolist()
    
    return documents, questions, correct_document_ids, documents_df

# 토크나이저 로드
def load_tokenizer(tokenizer_file):
    if os.path.exists(tokenizer_file):
        print(f"Loading tokenizer from {tokenizer_file}...")
        with open(tokenizer_file, 'rb') as f:
            tokenizer = pickle.load(f)
    else:
        print("Initializing new BertTokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("monologg/kobigbird-bert-base")
        with open(tokenizer_file, 'wb') as f:
            pickle.dump(tokenizer, f)
        print(f"Tokenizer saved to {tokenizer_file}")
    
    return tokenizer

def save_results_to_csv(results, output_file):
    keys = results[0].keys()  # CSV의 헤더로 사용할 키들
    with open(output_file, 'w', newline='', encoding='utf-8') as output_csv:
        dict_writer = csv.DictWriter(output_csv, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
    print(f"Results saved to {output_file}")

def run_experiments(args, documents, questions, correct_document_ids, documents_df):
    # 토크나이저 로드
    tokenizer = load_tokenizer(args.tokenizer_file)
    
    # TF-IDF 불러오기
    if args.model in ["TF-IDF", "ALL"]:
        tfidf_model = TFIDFModel(documents)
        if os.path.exists(args.tfidf_model_file):
            tfidf_model.load(args.tfidf_model_file)
        else:
            tfidf_model.train(args.tfidf_model_file)
        
    # BM25 불러오기
    if args.model in ["BM25", "ALL"]:
        bm25_model = BM25Model(documents, tokenizer, args.k1, args.b)
        if os.path.exists(args.bm25_model_file):
            bm25_model.load(args.bm25_model_file)
        else:
            bm25_model.train(args.bm25_model_file)
            
    bm25_correct_predictions = 0
    tfidf_correct_predictions = 0
    total_questions = len(questions)
    
    results = []
    
    # 실험 진행
    for i, question in enumerate(tqdm(questions, desc="Processing Questions")):
        correct_doc_id = correct_document_ids[i]
        # TF-IDF 실험
        if args.model in ["TF-IDF", "ALL"]:
            tfidf_top_k_documents = tfidf_model.get_top_k(query=question, k=args.k)
            top_k_predicted_doc_ids = [documents_df['document_id'][documents.index(doc)] for doc, _ in tfidf_top_k_documents]
            is_correct = correct_doc_id in top_k_predicted_doc_ids  # top-k 중에 정답이 있는지 확인
            if is_correct:
                tfidf_correct_predictions += 1

            for rank, (doc, score) in enumerate(tfidf_top_k_documents):
                predicted_doc_id = documents_df['document_id'][documents.index(doc)]
                results.append({
                    'model': 'TF-IDF',
                    'question': question,
                    'predicted_document_id': predicted_doc_id,
                    'score': score,
                    'correct_document_id': correct_doc_id,
                    'is_correct': is_correct
                })
                
        # BM25 실험
        if args.model in ["BM25", "ALL"]:
            bm25_top_k_documents = bm25_model.get_top_k(query=question, k=args.k)
            top_k_predicted_doc_ids = [documents_df['document_id'][documents.index(doc)] for doc, _ in bm25_top_k_documents]
            is_correct = correct_doc_id in top_k_predicted_doc_ids  # top-k 중에 정답이 있는지 확인
            if is_correct:
                bm25_correct_predictions += 1

            for rank, (doc, score) in enumerate(bm25_top_k_documents):
                predicted_doc_id = documents_df['document_id'][documents.index(doc)]
                results.append({
                    'model': 'BM25',
                    'question': question,
                    'predicted_document_id': predicted_doc_id,
                    'score': score,
                    'correct_document_id': correct_doc_id,
                    'is_correct': is_correct
                })
                
    save_results_to_csv(results, args.output_file)
                
    # 결과 출력
    if args.model in ["BM25", "ALL"]:
        bm25_accuracy = bm25_correct_predictions / total_questions * 100
        print(f"BM25 Top-{args.k} 정확도: {bm25_accuracy:.2f}%")

    if args.model in ["TF-IDF", "ALL"]:
        tfidf_accuracy = tfidf_correct_predictions / total_questions * 100
        print(f"TF-IDF Top-{args.k} 정확도: {tfidf_accuracy:.2f}%")
        
def main():
    args = parse_arguments()
    
    documents, questions, correct_document_ids, documents_df = load_data(args.test_file, args.corpus_file)
    
    run_experiments(args, documents, questions, correct_document_ids, documents_df)
    
if __name__ == "__main__":
    main()