# -*- coding: utf-8 -*-
"""
Created on Mon Jun 16 13:22:54 2025

@author: bongi
"""

import streamlit as st
from search import load_tutorials, generate_embeddings, semantic_search

st.set_page_config(page_title="Hairspiration Finder", layout="centered")

st.title("Hairspiration Finder")
st.subheader("Find hairstyles that match your vibe ✨")

# Load dataset and embeddings once
@st.cache_resource
def init_search():
    data, combined = load_tutorials()
    model, embeddings = generate_embeddings(combined)
    return data, model, embeddings

dataset, model, embeddings = init_search()

# Input form
query = st.text_input("🔍 What hairdo are you looking for? (e.g. 'scalp-friendly styles for summer')")

if query:
    results = semantic_search(query, dataset, embeddings, model)

    st.markdown("---")
    st.write(f"### 🔎 Results for: *{query}*")
    
    for res in results:
        st.markdown(f"### {res['title']}")
        st.write(res['description'])
        if "youtube" in res['link']:
            st.video(res['link'])
        else:
            st.markdown(f"[Watch tutorial here]({res['link']})")
        st.caption(f"🎯 Tags: {', '.join(res['tags'])}")
        st.markdown("---")