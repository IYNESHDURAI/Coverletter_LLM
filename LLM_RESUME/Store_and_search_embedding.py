import faiss
import numpy as np

def store_embeddings(embeddings, file_path='faiss_index.index'):
    """
    Store embeddings in a FAISS index and save it to a file.
    
    Args:
    - embeddings (list of numpy arrays): The embeddings to store.
    - file_path (str): The path where the FAISS index will be saved.
    """
    # Convert the list of embeddings to a NumPy array
    embeddings_array = np.array(embeddings)

    # Initialize the FAISS index
    dimension = embeddings_array.shape[1]  # Dimensionality of your embeddings
    index = faiss.IndexFlatL2(dimension)   # L2 distance (Euclidean)

    # Add embeddings to the index
    index.add(embeddings_array)

    # Save the index to a file
    faiss.write_index(index, file_path)
    print(f"Embeddings stored in {file_path}")


def search_embeddings(query_embedding, index_file='faiss_index.index', k=5):
    """
    Search for the most similar embeddings in a FAISS index.
    
    Args:
    - query_embedding (numpy array): The embedding to search for.
    - index_file (str): The path to the FAISS index file.
    - k (int): Number of nearest neighbors to return.
    
    Returns:
    - distances (numpy array): The distances of the nearest neighbors.
    - indices (numpy array): The indices of the nearest neighbors.
    """
    # Load the FAISS index
    index = faiss.read_index(index_file)

    # Perform a search
    distances, indices = index.search(query_embedding, k)
    
    return distances, indices
