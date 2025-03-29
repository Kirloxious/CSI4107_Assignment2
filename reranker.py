import json
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from collections import defaultdict

RESULTS_PATH = "saved/results.tsv"
CORPUS_PATH = "scifact/corpus.jsonl"
QUERIES_PATH = "scifact/queries.jsonl"


def read_jsonl(path):
    d = {}
    with open(path, "r") as f:
        for line in f:
            data = json.loads(line)
            d[data["_id"]] = data
    return d 

def load_results(path):
    results = {}

    with open(path, "r") as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) != 6:
                continue  # Skip malformed lines

            query_id, _, doc_id, rank, score, run_id = parts
            query_id = int(query_id)
            doc_id = int(doc_id)
            rank = int(rank)
            score = float(score)
            run_id = int(run_id)

            if query_id not in results:
                results[query_id] = []

            results[query_id].append((doc_id, rank, score, run_id))

    return results

initial_results = load_results(RESULTS_PATH)
corpus = read_jsonl(CORPUS_PATH)
queries = read_jsonl(QUERIES_PATH)


def compute_similarity(query_embedding, doc_embedding):
    query_embedding_tensor = torch.tensor(query_embedding)
    doc_embedding_tensor = torch.tensor(doc_embedding)
    
    # Ensure the tensors are of float type (necessary for cosine similarity)
    query_embedding_tensor = query_embedding_tensor.float()
    doc_embedding_tensor = doc_embedding_tensor.float()
    
    # Ensure both embeddings are of the same shape
    if query_embedding_tensor.shape != doc_embedding_tensor.shape:
        raise ValueError(f"Embeddings have mismatched shapes: {query_embedding_tensor.shape} vs {doc_embedding_tensor.shape}")

    # Compute cosine similarity
    similarity = torch.cosine_similarity(query_embedding_tensor, doc_embedding_tensor)
    
    return similarity.item()  # Convert the result to a Python scalar

# Load a pretrained neural reranker
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "dslim/bert-base-NER" # 0.5241
model_name = 'sentence-transformers/all-mpnet-base-v2' # 0.6447

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to(device)
# tokenizer.pad_token = tokenizer.eos_token

def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    # mean pooling
    embeddings = outputs[0].mean(dim=1)
    return embeddings.cpu().numpy()


def save_embeddings(embeddings, filename):
    np.save(filename, embeddings)

def load_embeddings(filename):
    return np.load(filename)

def rerank_documents(results_data, embedding_lookup_table):
    reranked_results = {}

    for query_id, docs in results_data.items():
        print(f"Processing Query {query_id}")
        
        # Check if the query embedding already exists in the lookup table
        if query_id in embedding_lookup_table:
            query_embedding = embedding_lookup_table[query_id]
        else:
            # Get query embedding and store it in the lookup table
            query_text = queries[str(query_id)]["text"]
            query_embedding = get_embedding(query_text)
            embedding_lookup_table[query_id] = query_embedding

        doc_scores = []
        for doc_id, rank, score, run_id in docs:
            # Check if the document embedding exists in the lookup table
            if doc_id in embedding_lookup_table:
                doc_embedding = embedding_lookup_table[doc_id]
            else:
                # Get document embedding and store it in the lookup table
                doc_text = corpus[str(doc_id)]["text"]
                doc_embedding = get_embedding(doc_text)
                embedding_lookup_table[doc_id] = doc_embedding

            # Compute similarity between query and document
            similarity = compute_similarity(query_embedding, doc_embedding)
            combined_score = 0.5 * score + 0.5 * similarity
            doc_scores.append((doc_id, combined_score, rank, score, run_id))

        # Sort documents based on similarity score (highest first)
        reranked_results[query_id] = sorted(doc_scores, key=lambda x: x[1], reverse=True)

    return reranked_results

embedding_lookup_table = {}  # A dictionary to store embeddings
reranked_results = rerank_documents(initial_results, embedding_lookup_table)


with open("reranked_results.tsv", "w") as f:
    for query_id, docs in reranked_results.items():
        rank = 1
        for doc_id, similarity, prev_rank, score, run_id in docs:
            f.write(f"{query_id}\tQ0\t{doc_id}\t{rank}\t{similarity:.8f}\t{run_id}\n")
            rank += 1