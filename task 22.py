import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Step 1: Extract text from specific pages (Page 2 and Page 6)
def extract_pdf_text_from_specific_pages(pdf_path, pages_to_extract=[2, 6]):
    """
    Extract text from specific pages of a PDF file.
    """
    try:
        doc = fitz.open(pdf_path)
        page_count = doc.page_count
        if page_count == 0:
            raise ValueError(f"The PDF file '{pdf_path}' is empty or invalid.")

        extracted_text = {}
        for page_num in pages_to_extract:
            if page_num <= page_count:
                page = doc.load_page(page_num - 1)  # PyMuPDF uses 0-indexing for pages
                extracted_text[page_num] = page.get_text("text")  # Extract text as plain text
            else:
                print(f"Warning: Page {page_num} does not exist in the document.")
        
        return extracted_text

    except Exception as e:
        raise ValueError(f"Error processing PDF file '{pdf_path}': {e}")

# Step 2: Segment text into chunks
def segment_text_into_chunks(text, chunk_size=5):
    """
    Segment text into chunks for better granularity.
    """
    sentences = [sentence.strip() for sentence in text.replace("\n", " ").split(".") if sentence.strip()]
    chunks = [' '.join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]
    return chunks

# Step 3: Convert chunks into vector embeddings
def convert_to_embeddings(chunks, model):
    """
    Convert text chunks into embeddings using a pre-trained Sentence-Transformer model.
    """
    embeddings = model.encode(chunks, convert_to_tensor=False)
    return np.array(embeddings)

# Step 4: Store embeddings in a vector database
def create_faiss_index(embeddings):
    """
    Create a FAISS index for efficient similarity-based retrieval.
    """
    index = faiss.IndexFlatL2(embeddings.shape[1])  # L2 distance for similarity search
    index.add(embeddings)  # Add embeddings to the index
    return index

# Step 5: Convert the user's query into embeddings
def query_to_embedding(query, model):
    """
    Convert the user's query into an embedding.
    """
    return model.encode([query], convert_to_tensor=False)

# Step 6: Perform similarity search in the vector database
def search_similar_chunks(query_embedding, index, k=5):
    """
    Perform similarity search in the FAISS index to retrieve the most relevant chunks.
    """
    distances, indices = index.search(np.array(query_embedding), k)
    return indices[0], distances[0]

# Step 7: Generate a response using GPT-2
def generate_response(query, context_chunks, tokenizer, gpt2_model, max_input_length=1024, max_new_tokens=100):
    """
    Generate a response using GPT-2 given the query and relevant context chunks.
    """
    input_text = f"Question: {query}\nContext: {' '.join(context_chunks)}\nAnswer:"
    
    # Truncate input text if it exceeds the model's maximum input length
    input_ids = tokenizer.encode(input_text, truncation=True, max_length=max_input_length, return_tensors="pt")
    
    # Create attention mask (1 for real tokens, 0 for padding tokens)
    attention_mask = (input_ids != tokenizer.pad_token_id).type(input_ids.dtype)

    # Generate response
    outputs = gpt2_model.generate(
        input_ids,
        attention_mask=attention_mask,  # Pass the attention mask explicitly
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        no_repeat_ngram_size=2
    )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

# Step 8: Main function to handle the RAG pipeline
def rag_pipeline(pdf_path, user_query, pages_to_extract=[2, 6]):
    """
    Full RAG pipeline to handle user queries on PDF data.
    """
    print("Extracting PDF content from specific pages...")
    try:
        text_data = extract_pdf_text_from_specific_pages(pdf_path, pages_to_extract)
    except ValueError as e:
        return f"Error: {e}"

    print("Segmenting text into chunks...")
    chunks = []
    for page_num, text in text_data.items():
        chunks.extend(segment_text_into_chunks(text))

    print("Loading Sentence-Transformer model...")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    print("Generating embeddings for content chunks...")
    chunk_embeddings = convert_to_embeddings(chunks, embedding_model)

    print("Creating FAISS index...")
    faiss_index = create_faiss_index(chunk_embeddings)

    print("Generating embedding for user query...")
    query_embedding = query_to_embedding(user_query, embedding_model)

    print("Performing similarity search...")
    indices, _ = search_similar_chunks(query_embedding, faiss_index)
    relevant_chunks = [chunks[i] for i in indices]

    print("Loading GPT-2 model and tokenizer...")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")

    # Set the pad_token to eos_token to handle padding
    tokenizer.pad_token = tokenizer.eos_token

    print("Generating response using GPT-2...")
    response = generate_response(user_query, relevant_chunks, tokenizer, gpt2_model)
    return response

# Example Usage
if __name__ == "__main__":
    pdf_path = "E:/greeshma/sitafal/pdf.pdf"  # Replace with the path to your PDF file
    user_query = "What is the unemployment rate based on degree type?"

    print("Running RAG pipeline...")
    try:
        final_response = rag_pipeline(pdf_path, user_query, pages_to_extract=[2, 6])
        print("\nGenerated Response:\n", final_response)
    except Exception as e:
        print("An error occurred:", e)
