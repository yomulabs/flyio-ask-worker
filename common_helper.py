
import openai
from timeit import default_timer as timer

def create_embedding(text):
    start = timer()
    embedding_model = 'text-embedding-ada-002'
    embedding = openai.Embedding.create(input = [text], model=embedding_model)['data'][0]['embedding']

    end = timer()
    print(f'encode took {end - start} seconds')
    return embedding
