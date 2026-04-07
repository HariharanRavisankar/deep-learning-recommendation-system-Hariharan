# Enterprise-Grade Recommendation System with Deep Learning

This project implements a hybrid Recommendation System that combines Neural Collaborative Filtering (NCF) using PyTorch and Content-Based Filtering using NLP (TF-IDF). It is designed to simulate a real-world scenario where a platform must provide personalized service recommendations to users to increase engagement and reduce churn.

## Project Overview
The system utilizes telecom customer data to predict and recommend additional services (like Streaming TV or Online Security) that a user is likely to adopt. It addresses the "Cold Start" problem by using NLP-based similarity when user history is sparse.

### Key Features
* **Deep Learning Model:** Built with PyTorch using Neural Collaborative Filtering (NCF) with User and Item Embeddings.
* **NLP Engine:** Implements TF-IDF Vectorization and Cosine Similarity for content-based service recommendations.
* **Interactive Dashboard:** A Streamlit-based web interface for real-time predictions and user profile analysis.
* **Hybrid Logic:** Provides both collaborative and content-based perspectives for comprehensive decision-making.

## Technical Architecture
1.  **Data Layer:** Processes raw CSV data into a User-Item interaction matrix.
2.  **Modeling Layer:** * **NCF:** A Multi-Layer Perceptron (MLP) architecture that learns non-linear relationships between users and services.
    * **TF-IDF:** Analyzes synthetic service descriptions to find logical service groupings.
3.  **Application Layer:** Streamlit UI that allows stakeholders to input a Customer ID and view ranked recommendations.

## Installation and Setup

### Prerequisites
* Python 3.8+
* PyTorch
* Pandas / Numpy
* Scikit-Learn
* Streamlit

### Step 1: Clone the Repository
```bash
git clone <your-repository-link>
cd <project-folder-name>
