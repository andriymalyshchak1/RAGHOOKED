import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import os
import google.generativeai as genai

# Step 3: Create an embedding model object
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
load_dotenv(override=True)

# Step 4: Define the cosine similarity function
def cosine_similarity(a, b):
    return np.dot(a, b.T) / (np.linalg.norm(a, axis=1)[:, np.newaxis] * np.linalg.norm(b, axis=1))

# Step 5: Read external raw data from a CSV and embed

# Function to read and process CSV file
def load_data_from_csv(file_path, text_columns):
    df = pd.read_csv(file_path)
    combined_text = df[text_columns].apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1)
    return combined_text.dropna().tolist()

# Load and process the CSV data
file_path = "output.csv"  # Ensure this is the correct path to your CSV file
text_columns = ["Academic Year Span", "Semester", "Section Number", "Course Prefix", "Course Number", "Course Title", "Course", "Letter Grade", "Count of letter grade", "Department/Program"]  # Replace with the actual column name in your CSV file that contains the text data
external_data = load_data_from_csv(file_path, text_columns)
chunks = [chunk for chunk in external_data if chunk.strip()]  # Remove empty chunks
embedded_data = model.encode(chunks)

# Step 6: Get user input and embed
queries = ["Name a finannce class that you know about?"]
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
    top_docs = [chunks[index] for index in top_indices]
    
    # Create the augmented prompt
    augmented_prompt = f"You are an expert question answering system. I'll give you a question and context and you'll return the answer.Never mention that the data is only from one semester Never important. Query: {queries[i]} Contexts: {top_docs[0]}"
    
    # Generate the model output
    model_output = Gemini.generate_content(augmented_prompt)
    
    # Print the output
    print(model_output.text)

