import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

class SimpleRAG:
    def __init__(self):
        # Initialize embeddings model
        self.embeddings_model = SentenceTransformer('all-MiniLM-L6-v2')
        # Initialize text generation model
        self.generator = pipeline('text-generation', model='gpt2')
        # Initialize empty knowledge base
        self.knowledge_base = []
        self.knowledge_embeddings = []
    
    def add_to_knowledge_base(self, texts):
        """Add documents to the knowledge base"""
        self.knowledge_base.extend(texts)
        # Create embeddings for new documents
        new_embeddings = self.embeddings_model.encode(texts)
        if len(self.knowledge_embeddings) == 0:
            self.knowledge_embeddings = new_embeddings
        else:
            self.knowledge_embeddings = np.vstack([self.knowledge_embeddings, new_embeddings])
    
    def retrieve(self, query, k=2):
        """Retrieve k most relevant documents for the query"""
        # Get query embedding
        query_embedding = self.embeddings_model.encode([query])[0]
        
        # Calculate similarity scores
        similarities = cosine_similarity([query_embedding], self.knowledge_embeddings)[0]
        
        # Get top k most similar documents
        top_k_indices = np.argsort(similarities)[-k:][::-1]
        return [self.knowledge_base[i] for i in top_k_indices]
    
    def generate_response(self, query):
        """Generate a response based on retrieved documents"""
        # Retrieve relevant documents
        relevant_docs = self.retrieve(query)
        
        # Create prompt by combining query and retrieved documents
        context = " ".join(relevant_docs)
        prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
        
        # Generate response
        response = self.generator(prompt, max_length=150, num_return_sequences=1)[0]['generated_text']
        return response

# Example usage
def main():
    # Initialize RAG system
    rag = SimpleRAG()
    
    # Add some sample documents to the knowledge base
    # documents = [
    #     "The capital of France is Paris. It is known for the Eiffel Tower.",
    #     "Paris is the largest city in France and a global center for art and culture.",
    #     "The Eiffel Tower was completed in 1889 and stands 324 meters tall.",
    # ]
    documents = [
        "Alger est la capitale de l'Algérie.",
        "Alger est la plus grande ville d'Algérie, et une capitale mondiale de l'art et de la culture.",
        "Makam echahid est le monument le plus connu d'Algérie. Il honnore la mémoire des martyrs.",
    ]
    rag.add_to_knowledge_base(documents)
    
    # Test the system
    query = "Tell me about Paris and the Eiffel Tower"
    response = rag.generate_response(query)
    print(f"Query: {query}")
    print(f"Response: {response}")

if __name__ == "__main__":
    main()
