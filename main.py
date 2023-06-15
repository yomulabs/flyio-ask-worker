
from dotenv import load_dotenv
from typing import Optional

import pinecone
import os
from sentence_transformers import SentenceTransformer
import openai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from timeit import default_timer as timer
import requests
from bs4 import BeautifulSoup
import math

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Msg(BaseModel):
    msg: str


@app.get("/")
async def root():
    return {"message": "Hello World. Welcome to FastAPI!"}

from pymilvus import (
    MilvusClient,
    connections,
    utility,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
)


load_dotenv()

milvus_client = MilvusClient(
    uri=os.getenv("MILVUS_ENDPOINT"),
    token=os.getenv("MILVUS_API_KEY")
)

milvus_collection_name = 'flyio_ada'

# ada has max input size of 8191, and gpt-3.5 turbo new has 16k context window. we're gonna set chunk size to 8k
MODEL_CHUNK_SIZE = 8192

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

index = pinecone.Index("fly-io-docs")

# https://huggingface.co/spaces/mteb/leaderboard
# https://huggingface.co/embaas/sentence-transformers-e5-large-v2
sentence_transformer_model = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')

def add_html_to_vectordb(content, path):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = MODEL_CHUNK_SIZE,
        chunk_overlap  = math.floor(MODEL_CHUNK_SIZE/10)
    )

    docs = text_splitter.create_documents([content])


    for doc in docs:
        embedding = create_embedding(doc.page_content)
        insert_embedding(embedding, doc.page_content, path)

def create_embedding(text):
    start = timer()
    # embedding = sentence_transformer_model.encode(text)
    # embedding = sentence_transformer_model.encode(text)
    embedding_model = 'text-embedding-ada-002'
    embedding = openai.Embedding.create(input = [text], model=embedding_model)['data'][0]['embedding']

    end = timer()
    print(f'encode took {end - start} seconds')
    return embedding

def insert_embedding_to_milvus(embedding, text, path):
    row = {
        'vector': embedding,
        'text': text,
        'path': path
    }

    start = timer()
    milvus_client.insert("flyio_ada", data=[row])
    end = timer()

    print(f'insert_vector {len(text)} took {end - start} seconds')

def insert_embedding(embedding, text, path):
    insert_embedding_to_milvus(embedding, text, path)

def insert_embeeding_to_pinecone(embedding, text, path):
    last_id  = index.describe_index_stats()['total_vector_count']

    upserted_data = []
    vector = (str(last_id + 1), embedding.tolist(), { 'content': text, 'path': path })
    upserted_data.append(vector)

    start = timer()
    index.upsert(vectors=upserted_data)
    end = timer()

    print(f'insert_vector {len(text)} took {end - start} seconds')

def list_vectors():
    results = index.fetch(ids=[])
    print(results)

def query_milvus(embedding):
    result_count = 1

    result = milvus_client.search(
        collection_name=milvus_collection_name,
        data=[embedding],
        limit=result_count,
        output_fields=["path", "text"])

    list_of_knowledge_base = map(lambda match: match['entity']['text'], result[0])
    list_of_sources = map(lambda match: match['entity']['path'], result[0])

    return {
        'list_of_knowledge_base': list_of_knowledge_base,
        'list_of_sources': list_of_sources
    }

def query_vector_db(embedding):
    return query_milvus(embedding)

def query_pinecone(embedding):
    result = index.query(embedding, top_k=3, includeMetadata=True)
    return result

def ask_chatgpt(knowledge_base, user_query):
    system_content = """You are an AI coding assistant designed to help users with their programming needs based on the Knowledge Base provided.
    If you dont know the answer, say that you dont know the answer. You will only answer questions related to fly.io, any other questions, you should say that its out of your responsibilities.
    Only answer questions using data from knowledge base and nothing else.
    """

    user_content = f"""
        Knowledge Base:
        ---
        {knowledge_base}
        ---
        User Query: {user_query}
        Answer:
    """
    system_message = {"role": "system", "content": system_content}
    user_message = {"role": "user", "content": user_content}
    print(user_content)
    chatgpt_response = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=[system_message, user_message])
    print(chatgpt_response)

def run(query):
    #user_query = "how do i scale cpu?"
    user_query = query
    embedding = create_embedding(user_query)
    result = query_vector_db(embedding)

    print("sources")
    for source in result['list_of_sources']:
        print(source)

    knowledge_base = "\n".join(result['list_of_knowledge_base'])
    ask_chatgpt(knowledge_base, user_query)


def get_html_body_content(url):
    response = requests.get(url)

    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the body element and extract its inner text
    body = soup.body
    inner_text = body.get_text()
    return inner_text

def get_html_sitemap(url):
    response = requests.get(url)

    # Parse the HTML content
    soup = BeautifulSoup(response.content, "lxml")

    # Find the body element and extract its inner text
    links = []

    locations = soup.find_all("loc")
    for location in locations:
        url = location.get_text()
        if "fly.io/docs" not in url:
            links.append(url)

    return links


def index_website():
    links = get_html_sitemap("https://fly.io/sitemap.xml")
    for link in links[7:]:
        print(link)
        try:
            content = get_html_body_content(link)
            add_html_to_vectordb(content, link)
        except:
            print("unable to process: " + link)

def print_vector_count_milvus():
    stats = milvus_client.describe_collection(collection_name=milvus_collection_name)
    print(stats)

    result = milvus_client.query(collection_name=milvus_collection_name, filter='id > 3', output_fields=["id", "path"], limit=200)
    print(len(result))

#index_website()
#run('how do i add turbo streams')
#print_vector_count_milvus()

