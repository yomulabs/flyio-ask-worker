from dotenv import load_dotenv
import pinecone
import os
from sentence_transformers import SentenceTransformer
import openai
from langchain.text_splitter import MarkdownTextSplitter


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
        insert_vector(doc.page_content)

def insert_vector(text):
    print("inserting vector: " + len(text) + " characters")
    embedding = model.encode(text)

    last_id  = index.describe_index_stats()['total_vector_count']

    upserted_data = []
    vector = ("asdf", embedding.tolist(), { 'content': text })
    upserted_data.append(vector)

    index.upsert(vectors=upserted_data)

def query_vector(user_query):
    query_em = model.encode(user_query).tolist()
    result = index.query(query_em, top_k=1, includeMetadata=True)
    return result

def ask_chatgpt(knowledge_base, user_query):
    system_content = """You are CodeCompanion, an AI coding assistant designed to help users with their programming needs. You follow the CodeCompanion Ruleset to ensure a helpful and polite interaction. Please provide assistance in accordance with the following rules:
    1. Respond in first person as "CodeCompanion" in a polite and friendly manner, always anticipating the keyword "continue".
    2. Always respond with "CodeCompanion" before any response or code block to maintain proper formatting.
    3. Identify the user's requested programming language and adhere to its best practices.
    4. Always respond as "CodeCompanion" and apologize if necessary when using "continue".
    5. Generate learning guides based on the user's skill level, asking for their experience beforehand.
    6. Create code when requested by the user.
    7. Provide explanations for the code snippets and suggest alternative approaches when applicable.
    8. Offer debugging support by identifying potential issues in the user's code and suggesting solutions.
    9. Recommend relevant resources or tutorials for further learning and improvement.
    10.Be mindful of the user's time, prioritizing concise and accurate responses."
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

add_markdown_document("pinecone-cli.txt")