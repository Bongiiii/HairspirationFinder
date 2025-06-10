# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 16:55:58 2025

@author: bongi
"""

# backend for embeddings script
import json
from sentence_transformers import SentenceTransformer, util

#Load dataset
def load_tutorials(path=r"C:\Users\bongi\Desktop\projects\HairspirationFinder\tutorials_dataset.json"):
    with open(path, 'r') as file:
        data = json.load(file)
        
    #Combine title and description to create richer, more descriptive text embeddings for semantic search
    combined = [f"{item['title']}. {item['description']}" for item in data]
    return data, combined

#embedding generator helper func
def generate_embeddings(text_list, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(text_list, convert_to_tensor=True)
    return model, embeddings


#search helper func
def semantic_search(query, dataset, dataset_embeddings, model, top_k=3):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.cos_sim(query_embedding, dataset_embeddings)[0]
    top_results = similarities.argsort(descending=True)[:top_k]

    results = []
    for idx in top_results:
        results.append({
            "score": float(similarities[idx]),
            "title": dataset[idx]['title'],
            "description": dataset[idx]['description'],
            "tags": dataset[idx]['tags'],
            "link": dataset[idx]['link']
        })
    return results

if __name__ == "__main__":
    dataset, combined_texts = load_tutorials()
    model, embeddings = generate_embeddings(combined_texts)

    query = "diy sew in"
    top_matches = semantic_search(query, dataset, embeddings, model)

    for res in top_matches:
        print(f"{res['score']:.2f} - {res['title']}")
