
import openai

def create_embedding(text):
    embedding_model = 'text-embedding-ada-002'
    embedding = openai.Embedding.create(input = [text], model=embedding_model)['data'][0]['embedding']

    return embedding
