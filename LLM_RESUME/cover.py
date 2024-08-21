from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

def generate_cover_letter(jd, chunk, model_name='gpt2'):
    """
    Generate a cover letter based on the job description and relevant chunk of the resume.
    
    Args:
    - jd (str): The job description.
    - chunk (str): The relevant chunk from the resume.
    - model_name (str): The name of the Hugging Face model to use.
    
    Returns:
    - str: The generated cover letter.
    """
    # Load the pre-trained model and tokenizer from Hugging Face
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)
    
    # Prepare the input text
    input_text = f"Job Description: {jd}\n\nRelevant Experience: {chunk}\n\nCover Letter:"
    
    # Tokenize the input text
    inputs = tokenizer.encode(input_text, return_tensors='pt', max_length=1024, truncation=True)

    # Generate the cover letter
    outputs = model.generate(
        inputs,
        max_length=500,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        do_sample=True
    )
    
    # Decode the output and return the cover letter
    cover_letter = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return cover_letter

# Example usage
jd = "We are seeking a highly skilled Machine Learning Engineer to join our team..."
chunk = "Relevant experience in machine learning, including work with deep learning frameworks such as TensorFlow and PyTorch..."

cover_letter = generate_cover_letter(jd, chunk)
print("Generated Cover Letter:\n", cover_letter)
