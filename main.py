from dotenv import load_dotenv
import pinecone
import os
from sentence_transformers import SentenceTransformer
import openai
from langchain.text_splitter import MarkdownTextSplitter
from timeit import default_timer as timer
import requests
from bs4 import BeautifulSoup


load_dotenv()

MODEL_CHUNK_SIZE = 512

pinecone.init(api_key=os.getenv("PINECONE_API_KEY"), environment=os.getenv("PINECONE_ENV"))

index = pinecone.Index("fly-io-docs")

# https://huggingface.co/spaces/mteb/leaderboard
# https://huggingface.co/embaas/sentence-transformers-e5-large-v2
model = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')

def add_markdown_document(filename):
    f = open(filename, "r")
    document = f.read()

    markdown_splitter = MarkdownTextSplitter(chunk_size=MODEL_CHUNK_SIZE, chunk_overlap=0)
    docs = markdown_splitter.create_documents([document])
    for doc in docs:
        insert_vector(doc.page_content, filename)

def rebuild_vector_for_path(path):
    print("eh")

def insert_vector(text, path):
    start = timer()
    embedding = model.encode(text)
    end = timer()
    print(f'encode took {end - start} seconds')

    last_id  = index.describe_index_stats()['total_vector_count']

    upserted_data = []
    vector = (str(last_id + 1), embedding.tolist(), { 'content': text, 'path': path })
    upserted_data.append(vector)

    index.upsert(vectors=upserted_data)
    end = timer()

    print(f'insert_vector {len(text)} took {end - start} seconds')

def list_vectors():
    results = index.fetch(ids=[])
    print(results)

def query_vector(user_query):
    query_em = model.encode(user_query).tolist()
    result = index.query(query_em, top_k=1, includeMetadata=True)
    return result

def ask_chatgpt(knowledge_base, user_query):
    system_content = """You are an AI coding assistant designed to help users with their programming needs. You follow the CodeCompanion Ruleset to ensure a helpful and polite interaction. Please provide assistance in accordance with the following rules:
    1. Create code when requested by the user.
    2. Provide explanations for the code snippets and suggest alternative approaches when applicable.
    3. Offer debugging support by identifying potential issues in the user's code and suggesting solutions.
    4. Recommend relevant resources or tutorials for further learning and improvement.
    5. If you dont know the answer, say that you dont know the answer.
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
    chatgpt_response = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[system_message, user_message])
    print(chatgpt_response)

def run():
    user_query = "how do i backup an index in pinecone?"
    result = query_vector(user_query)
    knowledge_base = result.matches[0].metadata['content']
    ask_chatgpt(knowledge_base, user_query)


# f = open("pinecone-cli.txt", "r")
# document = f.read()

#add_markdown_document("pinecone-cli.txt")

def get_html_body_content(url):
    response = requests.get(url)

    # Parse the HTML content
    soup = BeautifulSoup(response.content, "html.parser")

    # Find the body element and extract its inner text
    body = soup.body
    inner_text = body.get_text(strip=True)
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
        if "fly.io/docs" in url:
            links.append(url)

    return links


links = get_html_sitemap("https://fly.io/sitemap.xml")
count = 3
for link in links[:count]:
    content = get_html_body_content(link)
    insert_vector(content, link)

