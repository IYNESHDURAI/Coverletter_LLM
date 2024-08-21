print("distilbert_embedding.py is being loaded")


# Your function definition here
from transformers import DistilBertTokenizer, DistilBertModel
import torch

def get_distilbert_embeddings(text_chunks):
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    model = DistilBertModel.from_pretrained('distilbert-base-uncased')

    embeddings = []
    for chunk in text_chunks:
        inputs = tokenizer(chunk, return_tensors='pt', truncation=True, padding=True)
        outputs = model(**inputs)
        chunk_embedding = torch.mean(outputs.last_hidden_state, dim=1).detach().numpy()
        embeddings.append(chunk_embedding)
    
    return embeddings
