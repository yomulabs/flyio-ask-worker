from common_helper import create_embedding
import openai

class SearchEngine:
  def __init__(self, milvus_client, milvus_collection_name):
    self.milvus_client = milvus_client
    self.milvus_collection_name = milvus_collection_name

  def query_milvus(self, embedding):
      result_count = 3

      result = self.milvus_client.search(
          collection_name=self.milvus_collection_name,
          data=[embedding],
          limit=result_count,
          output_fields=["path", "text"])

      list_of_knowledge_base = list(map(lambda match: match['entity']['text'], result[0]))
      list_of_sources = list(map(lambda match: match['entity']['path'], result[0]))

      return {
          'list_of_knowledge_base': list_of_knowledge_base,
          'list_of_sources': list_of_sources
      }

  def query_vector_db(self, embedding):
      return self.query_milvus(embedding)

  def ask_chatgpt(self, knowledge_base, user_query):
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
      
      chatgpt_response = openai.ChatCompletion.create(model="gpt-3.5-turbo-0613", messages=[system_message, user_message])
      return chatgpt_response.choices[0].message.content

  def search(self, user_query):
      embedding = create_embedding(user_query)
      result = self.query_vector_db(embedding)

      print("sources")
      for source in result['list_of_sources']:
          print(source)

      knowledge_base = "\n".join(result['list_of_knowledge_base'])
      response = self.ask_chatgpt(knowledge_base, user_query)

      return {
          'sources': result['list_of_sources'],
          'response': response
      }
