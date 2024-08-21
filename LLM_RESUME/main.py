from read_pdf import chunk_pdf_text
from distilbert_embedding import get_distilbert_embeddings
from Store_and_search_embedding import store_embeddings, search_embeddings
from cover import generate_cover_letter
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

pdf_path = r'C:\Users\CGSPL\Desktop\LLM\LLM_RESUME\IYNESHDURAI_RESUME.pdf'

# Set chunk size and overlap
chunk_size = 700  # Adjust the size according to your needs (in characters)
overlap = 100  # Number of overlapping characters between chunks

# Get the chunks
text_chunks = chunk_pdf_text(pdf_path, chunk_size, overlap)

print(text_chunks,"\n\n\n")
# Display the chunks
for i, chunk in enumerate(text_chunks):
    print(f"Chunk {i+1}:\n{chunk}\n")

# Step 3: Generate embeddings using DistilBERT
distilbert_embeddings = get_distilbert_embeddings(text_chunks)
#print(distilbert_embeddings)
#print(distilbert_embeddings)

# Ensure embeddings are 2D
distilbert_embeddings = np.vstack(distilbert_embeddings)
print(distilbert_embeddings)

jd_embedding = " Job Description- We are seeking a highly skilled Machine Learning Engineer to join our team and contribute to the development and deployment of cutting-edge AI solutions. The ideal candidate will have a strong foundation in machine learning, deep learning, and natural language processing, with a proven track record of success in applying these techniques to real-world problems. "

# Step 4: Generate embeddings using DistilBERT for JD
distilbert_embeddings_JD = get_distilbert_embeddings([jd_embedding])

# Ensure JD embedding is 2D
distilbert_embeddings_JD = np.vstack(distilbert_embeddings_JD)
print("Embedding for JD : - \n\n ", distilbert_embeddings_JD)

#Step 5: Store embedding
store_embeddings(distilbert_embeddings)

#step 6: Search for similar embeddings
distances, indices = search_embeddings(np.array(distilbert_embeddings_JD))

 # Print the results
print("Nearest chunks to the JD embedding:")
for i, idx in enumerate(indices[0]):
    print(f"Chunk {idx + 1} with distance {distances[0][i]}")

# Use the closest chunk for cover letter generation
most_similar_chunk = text_chunks[indices[0][0]]

# Step 7: Generate the cover letter using the most similar chunk and JD
cover_letter = generate_cover_letter(jd_embedding, most_similar_chunk)

# Print the generated cover letter
print("\nGenerated Cover Letter:\n", cover_letter)