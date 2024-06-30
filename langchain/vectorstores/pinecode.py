# In langchain/vectorstores/pinecone.py

class LangchainPinecone:
    def __init__(self, index, embedding_function, text_key="text"):
        self.index = index
        self.embedding_function = embedding_function
        self.text_key = text_key

    def similarity_search(self, query, **kwargs):
        query_vector = self.embedding_function(query)
        results = self.index.query(vector=query_vector, **kwargs)  # Ensure to pass kwargs properly
        return results

    def similarity_search_with_score(self, query, **kwargs):
        query_vector = self.embedding_function(query)
        results = self.index.query(vector=query_vector, **kwargs)  # Ensure to pass kwargs properly
        return results
