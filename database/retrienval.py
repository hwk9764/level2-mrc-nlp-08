import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from rank_bm25 import BM25Okapi
from tqdm import tqdm


class TFIDFModel:
    def __init__(self, documents):
        self.documents = documents
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        
    def train(self, model_file):
        print("Training TF-IDF model...")
        self.tfidf_matrix = self.vectorizer.fit_transform(tqdm(self.documents, desc="TF-IDF Training"))
        with open(model_file, 'wb') as f:
            pickle.dump((self.vectorizer, self.tfidf_matrix), f)
        print(f"TF-IDF model saved to {model_file}")
    
    def load(self, model_file):
        print(f"Loading TF-IDF model from {model_file}...")
        with open(model_file, 'rb') as f:
            self.vectorizer, self.tfidf_matrix = pickle.load(f)
    
    def get_top_k(self, query, k=5):
        query_vec = self.vectorizer.transform([query])
        cosine_similarities = (self.tfidf_matrix * query_vec.T).toarray().flatten()
        top_k_indices = cosine_similarities.argsort()[-k:][::-1]
        top_k_documents = [(self.documents[i], cosine_similarities[i]) for i in top_k_indices]
        return top_k_documents
    
class BM25Model:
    def __init__(self, documents, tokenizer, k1, b):
        self.tokenizer = tokenizer
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.bm25 = None
        
    def train(self, model_file):
        print("Training BM25 model...")
        tokenized_documents = [self.tokenizer.tokenize(doc) for doc in tqdm(self.documents, desc="BM25 Tokenizing")]
        self.bm25 = BM25Okapi(tokenized_documents, k1=self.k1, b=self.b)
        with open(model_file, 'wb') as f:
            pickle.dump(self.bm25, f)
        print(f"BM25 model saved to {model_file}")
        
    def load(self, model_file):
        print(f"Loading BM25 model from {model_file}...")
        with open(model_file, 'rb') as f:
            self.bm25 = pickle.load(f)
            
    def get_top_k(self, query, k=5):
        tokenized_query = self.tokenizer.tokenize(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_k_indices = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:k]
        top_k_documents = [(self.documents[i], bm25_scores[i]) for i in top_k_indices]
        return top_k_documents