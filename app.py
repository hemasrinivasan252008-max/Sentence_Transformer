from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load the embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Get two sentences from the user
sentence1 = input("Enter Sentence 1: ")
sentence2 = input("Enter Sentence 2: ")

# Convert sentences into embeddings
embedding1 = model.encode([sentence1])
embedding2 = model.encode([sentence2])

# Display Sentence 1 embedding
print("\nSentence 1 Embedding:")
print(embedding1)

# Display Sentence 2 embedding
print("\nSentence 2 Embedding:")
print(embedding2)

# Display embedding dimension
print("\nEmbedding Dimension:", len(embedding1[0]))

# Calculate cosine similarity
similarity = cosine_similarity(embedding1, embedding2)

# Display similarity results
print("\nCosine Similarity:", similarity[0][0])
print("Similarity Percentage:", round(similarity[0][0] * 100, 2), "%")