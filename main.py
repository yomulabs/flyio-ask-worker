from dotenv import load_dotenv
import pinecone
import os
from sentence_transformers import SentenceTransformer

load_dotenv()

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

index = pinecone.Index("fly-io-docs")

model = SentenceTransformer('multi-qa-mpnet-base-dot-v1')

print(model)

def insert_vector():
    f = open("doc.txt", "r")
    document = f.read()

    embedding = model.encode(document)

    last_id  = index.describe_index_stats()['total_vector_count']

    upserted_data = []
    vector = (str(last_id + 1), embedding.tolist(), { 'content': document })
    upserted_data.append(vector)

    index.upsert(vectors=upserted_data)

def query_vector():
    query = "how to backup"
    query_em = model.encode(query).tolist()
    result = index.query(query_em, top_k=1, includeMetadata=True)
    print(result)

query_vector()
