import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import math
from timeit import default_timer as timer
from common_helper import create_embedding

class Indexer:
  def __init__(self, milvus_client, milvus_collection_name):
    self.milvus_client = milvus_client
    self.milvus_collection_name = milvus_collection_name

  MODEL_CHUNK_SIZE = 8192

  def get_html_sitemap(self, url):
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

  def get_html_body_content(self, url):
      response = requests.get(url)

      # Parse the HTML content
      soup = BeautifulSoup(response.content, "html.parser")

      # Find the body element and extract its inner text
      body = soup.body
      inner_text = body.get_text()
      return inner_text

  def index_website(self, website_url):
      links = self.get_html_sitemap(website_url)
      for link in links[7:]:
          print(link)
          try:
              content = self.get_html_body_content(link)
              self.add_html_to_vectordb(content, link)
          except:
              print("unable to process: " + link)

  # https://huggingface.co/spaces/mteb/leaderboard
  # https://huggingface.co/embaas/sentence-transformers-e5-large-v2
  sentence_transformer_model = SentenceTransformer('embaas/sentence-transformers-e5-large-v2')

  def add_html_to_vectordb(self, content, path):
      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size = self.MODEL_CHUNK_SIZE,
          chunk_overlap  = math.floor(self.MODEL_CHUNK_SIZE/10)
      )

      docs = text_splitter.create_documents([content])

      for doc in docs:
          embedding = create_embedding(doc.page_content)
          self.insert_embedding(embedding, doc.page_content, path)

  def insert_embedding(self, embedding, text, path):
      row = {
          'vector': embedding,
          'text': text,
          'path': path
      }

      start = timer()
      self.milvus_client.insert(self.milvus_collection_name, data=[row])
      end = timer()

      print(f'insert_vector {len(text)} took {end - start} seconds')
