import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import google.generativeai as genai
import PyPDF2

# Step 3: Create an embedding model object
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
load_dotenv(override=True)

# Step 4: Define the cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, np.newaxis] * np.linalg.norm(b, axis=1))

# Step 5: Read external raw data from a PDF and embed

# Function to read and process PDF file
def load_data_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
    return text

# Load and process the PDF data
file_path_pdf = "University_of_Texas_at_Austin.pdf"  # Ensure this is the correct path to your PDF file
pdf_text = load_data_from_pdf(file_path_pdf)
embedded_data = model.encode([pdf_text])

# Step 6: Get user input and embed
user_query = input("Enter your question: ")
queries = [user_query]
embedded_queries = model.encode(queries)

# Step 7: Configure Gemini
api_key = os.getenv("GEMINI_API")
print("API Key:", api_key)
genai.configure(api_key=api_key)

generation_config = {
  "temperature": 0.9,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 6000,
}

Gemini = genai.GenerativeModel(model_name="gemini-1.5-flash-latest",
                                generation_config=generation_config,
                                )

# Step 8: Get the result
for i, query_vec in enumerate(embedded_queries):
    # Compute similarities
    similarities = cosine_similarity(query_vec[np.newaxis, :], embedded_data)
    
    # Get top 3 indices based on similarities
    top_indices = np.argsort(similarities[0])[::-1][:3]
    top_docs = [pdf_text for _ in range(len(top_indices))]
    
    # Create the augmented prompt
    augmented_prompt = f"You are an expert question answering system designed to help people learn more about the University of Texas. I'll give you a question and context based on UT history and you'll return the answer. Be very polite and have a texan slang way of speaking with y'alls and other words like that. Query: {queries[i]} Contexts: {top_docs[0]}"
    
    # Generate the model output
    model_output = Gemini.generate_content(augmented_prompt)
    
    # Print the output
    print(model_output.text)









