"""
Hospital RAG Indexing Script
Generates FAISS vector index and structured JSON for hospital lookups.
Run this ONCE before starting the Flask app.
"""

import pandas as pd
import numpy as np
import json
import pickle
from sentence_transformers import SentenceTransformer
import faiss
import random
import os

# Configuration
CSV_FILE = "List of GIPSA Hospitals - Sheet1.csv"
FAISS_INDEX_PATH = "hospital_index.faiss"
HOSPITAL_DATA_PATH = "hospital_data.pkl"
NETWORK_JSON_PATH = "network_status.json"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def load_and_prepare_data():
    """Load CSV and prepare hospital data"""
    print("📄 Loading CSV file...")
    df = pd.read_csv(CSV_FILE)
    
    # Clean data
    df = df.dropna(subset=['HOSPITAL NAME', 'CITY'])
    df['HOSPITAL NAME'] = df['HOSPITAL NAME'].str.strip()
    df['CITY'] = df['CITY'].str.strip()
    df['Address'] = df['Address'].fillna('').str.strip()
    
    print(f"✅ Loaded {len(df)} hospitals")
    return df

def generate_embeddings(df):
    """Generate embeddings for semantic search"""
    print("🧠 Loading embedding model (this may take a minute)...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    # Create rich text representations for better semantic matching
    print("🔄 Generating embeddings...")
    texts = []
    for _, row in df.iterrows():
        # Combine hospital name, city, and address for context-rich embeddings
        text = f"{row['HOSPITAL NAME']} located in {row['CITY']}, {row['Address']}"
        texts.append(text)
    
    embeddings = model.encode(texts, show_progress_bar=True)
    print(f"✅ Generated {len(embeddings)} embeddings (dimension: {embeddings.shape[1]})")
    
    return embeddings, texts

def create_faiss_index(embeddings):
    """Create FAISS index for fast similarity search"""
    print("🔍 Building FAISS index...")
    dimension = embeddings.shape[1]
    
    # Use L2 distance for similarity
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings.astype('float32'))
    
    print(f"✅ FAISS index created with {index.ntotal} vectors")
    return index

def simulate_network_status(df):
    """
    Simulate network status for each hospital.
    In a real system, this would come from a database.
    """
    print("🏥 Simulating network status...")
    
    network_data = {}
    for _, row in df.iterrows():
        # Create a unique key: lowercase hospital name + city
        key = f"{row['HOSPITAL NAME'].lower()}|{row['CITY'].lower()}"
        
        # Randomly assign network status (70% in-network, 30% out-of-network)
        status = "In Network" if random.random() < 0.7 else "Out of Network"
        
        network_data[key] = {
            "hospital_name": row['HOSPITAL NAME'],
            "city": row['CITY'],
            "address": row['Address'],
            "network_status": status
        }
    
    print(f"✅ Generated network status for {len(network_data)} hospitals")
    return network_data

def save_artifacts(index, df, network_data):
    """Save all generated artifacts to disk"""
    print("💾 Saving artifacts...")
    
    # Save FAISS index
    faiss.write_index(index, FAISS_INDEX_PATH)
    print(f"✅ Saved FAISS index: {FAISS_INDEX_PATH}")
    
    # Save hospital DataFrame for metadata lookup
    with open(HOSPITAL_DATA_PATH, 'wb') as f:
        pickle.dump(df, f)
    print(f"✅ Saved hospital data: {HOSPITAL_DATA_PATH}")
    
    # Save network status JSON
    with open(NETWORK_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(network_data, f, indent=2, ensure_ascii=False)
    print(f"✅ Saved network status: {NETWORK_JSON_PATH}")

def verify_setup():
    """Verify all required files exist"""
    print("\n🔍 Verifying setup...")
    required_files = [FAISS_INDEX_PATH, HOSPITAL_DATA_PATH, NETWORK_JSON_PATH]
    
    all_exist = True
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} exists")
        else:
            print(f"❌ {file} missing")
            all_exist = False
    
    return all_exist

def main():
    """Main indexing pipeline"""
    print("=" * 60)
    print("🏥 Hospital RAG Indexing Pipeline")
    print("=" * 60)
    
    try:
        # Step 1: Load data
        df = load_and_prepare_data()
        
        # Step 2: Generate embeddings
        embeddings, texts = generate_embeddings(df)
        
        # Step 3: Create FAISS index
        index = create_faiss_index(embeddings)
        
        # Step 4: Simulate network status
        network_data = simulate_network_status(df)
        
        # Step 5: Save everything
        save_artifacts(index, df, network_data)
        
        # Step 6: Verify
        if verify_setup():
            print("\n" + "=" * 60)
            print("✅ SUCCESS! RAG indexing complete.")
            print("=" * 60)
            print("\n📝 Next steps:")
            print("1. Set up your .env file with GEMINI_API_KEY")
            print("2. Run: python app.py")
            print("3. Open: http://localhost:5000")
        else:
            print("\n❌ Setup verification failed. Please check errors above.")
            
    except Exception as e:
        print(f"\n❌ Error during indexing: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
