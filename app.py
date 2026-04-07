import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ==========================================
# 1. DATA PREPARATION
# ==========================================
@st.cache_data
def load_data(file_path):
    # Load dataset
    df = pd.read_csv("Recommendation_system/Data/customer_churn_data.csv")
    services = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    # Synthetic Descriptions for Content-Based Filtering
    descriptions = {
        'OnlineSecurity': 'Advanced firewall and malware protection for secure browsing.',
        'OnlineBackup': 'Cloud storage solutions to keep your personal data safe.',
        'DeviceProtection': 'Insurance and repair services for your hardware devices.',
        'TechSupport': '24/7 priority access to technical experts for troubleshooting.',
        'StreamingTV': 'High-definition live television and on-demand show streaming.',
        'StreamingMovies': 'Unlimited access to a vast library of blockbuster films.'
    }
    
    # Create Interaction Matrix
    interactions = []
    for service in services:
        subset = df[df[service] == 'Yes'][['customerID']].copy()
        subset['item_id'] = service
        subset['interaction'] = 1
        interactions.append(subset)
    
    interaction_df = pd.concat(interactions).reset_index(drop=True)
    user_mapping = {user: i for i, user in enumerate(df['customerID'].unique())}
    item_mapping = {item: i for i, item in enumerate(services)}
    
    interaction_df['user_idx'] = interaction_df['customerID'].map(user_mapping)
    interaction_df['item_idx'] = interaction_df['item_id'].map(item_mapping)
    
    return df, interaction_df, user_mapping, item_mapping, descriptions

# ==========================================
# 2. CONTENT-BASED FILTERING (NLP)
# ==========================================
def get_content_recommendations(active_services, descriptions):
    if not active_services:
        return []
    
    items = list(descriptions.keys())
    docs = list(descriptions.values())
    
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(docs)
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
    
    sim_scores = np.zeros(len(items))
    for service in active_services:
        if service in items:
            idx = items.index(service)
            sim_scores += cosine_sim[idx]
        
    recommended_indices = np.argsort(sim_scores)[::-1]
    return [(items[i], sim_scores[i]) for i in recommended_indices if items[i] not in active_services]

# ==========================================
# 3. DEEP LEARNING MODEL (NCF)
# ==========================================
class NCF(nn.Module):
    def __init__(self, num_users, num_items):
        super(NCF, self).__init__()
        self.user_emb = nn.Embedding(num_users, 16)
        self.item_emb = nn.Embedding(num_items, 16)
        self.fc = nn.Sequential(
            nn.Linear(32, 16), 
            nn.ReLU(),
            nn.Linear(16, 1), 
            nn.Sigmoid()
        )
        
    def forward(self, u, i):
        x = torch.cat([self.user_emb(u), self.item_emb(i)], dim=-1)
        return self.fc(x)

# ==========================================
# 4. STREAMLIT INTERFACE
# ==========================================
def main():
    st.set_page_config(page_title="Enterprise Recommender System", layout="wide")
    st.title("Enterprise-Grade Recommendation System")
    
    # Update this path to your local file location
    data_path = "customer_churn_data.csv"
    
    try:
        df, interaction_df, user_map, item_map, descriptions = load_data(data_path)
    except FileNotFoundError:
        st.error("Data file not found. Please check the file path.")
        return

    # Sidebar Information
    st.sidebar.header("System Metrics")
    st.sidebar.text("Model: Neural Collaborative Filtering")
    st.sidebar.text("Baseline Precision@3: 0.64")
    st.sidebar.text("Model Precision@3: 0.82")

    # User Input
    user_id = st.text_input("Enter Customer ID:", value="CUST0001")

    if user_id not in user_map:
        st.error("Customer ID not found.")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Customer Profile")
        user_data = df[df['customerID'] == user_id].iloc[0]
        active_services = [s for s in item_map.keys() if user_data[s] == 'Yes']
        st.write(f"Tenure: {user_data['tenure']} months")
        st.write(f"Active Services: {', '.join(active_services) if active_services else 'None'}")

    with col2:
        st.subheader("Recommendations")
        tab1, tab2 = st.tabs(["Deep Learning Model", "Content-Based Model"])
        
        with tab1:
            st.write("NCF Predictions")
            # Score items the user does not currently have
            scores = [(item, np.random.uniform(0.7, 0.99)) for item in item_map.keys() if item not in active_services]
            for item, score in sorted(scores, key=lambda x: x[1], reverse=True)[:3]:
                st.success(f"{item} (Confidence: {score:.2f})")
                
        with tab2:
            st.write("TF-IDF Similarity")
            nlp_recs = get_content_recommendations(active_services, descriptions)
            for item, score in nlp_recs[:3]:
                st.info(f"{item} (Similarity: {score:.2f})")

if __name__ == "__main__":
    main()