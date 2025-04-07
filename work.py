!pip install streamlit spacy pdfplumber transformers
import streamlit as st
import spacy
import pdfplumber
from transformers import pipeline
import json

# Load a pre-trained Named Entity Recognition model (you can replace this with the Gemini or DeepSeek model if they are available locally)
@st.cache_resource
def load_ner_model():
    # This uses Huggingface's `transformers` for NER; you can swap this out with Gemini/DeepSeek models
    model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    return model

# Function to extract text from a PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
        return text

# Function to extract named entities using NER model
def extract_named_entities(text, ner_model):
    entities = ner_model(text)
    # Limit to the first 20 named entities
    top_entities = entities[:20]
    
    # Format entities as JSON
    entities_json = []
    for entity in top_entities:
        entities_json.append({
            "word": entity["word"],
            "entity": entity["entity"],
            "score": entity["score"]
        })
    
    return entities_json

# Streamlit interface
def main():
    st.title("Named Entity Recognition (NER) Extraction")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a document (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        # Load the model
        ner_model = load_ner_model()

        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(uploaded_file)

        # Display the extracted text
        st.subheader("Extracted Text")
        st.write(text[:1000])  # Display only the first 1000 characters for preview

        # Extract named entities
        named_entities = extract_named_entities(text, ner_model)

        # Display named entities as JSON
        st.subheader("Extracted Named Entities")
        st.json(named_entities)

# Run the app
if __name__ == "__main__":
    main()
from transformers import pipeline
import json

# Load a pre-trained Named Entity Recognition model (you can replace this with the Gemini or DeepSeek model if they are available locally)
@st.cache_resource
def load_ner_model():
    # This uses Huggingface's `transformers` for NER; you can swap this out with Gemini/DeepSeek models
    model = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english")
    return model

# Function to extract text from a PDF
def extract_text_from_pdf(file):
    with pdfplumber.open(file) as pdf:
        text = ''
        for page in pdf.pages:
            text += page.extract_text()
        return text

# Function to extract named entities using NER model
def extract_named_entities(text, ner_model):
    entities = ner_model(text)
    # Limit to the first 20 named entities
    top_entities = entities[:20]
    
    # Format entities as JSON
    entities_json = []
    for entity in top_entities:
        entities_json.append({
            "word": entity["word"],
            "entity": entity["entity"],
            "score": entity["score"]
        })
    
    return entities_json

# Streamlit interface
def main():
    st.title("Named Entity Recognition (NER) Extraction")
    
    # File uploader
    uploaded_file = st.file_uploader("Upload a document (PDF)", type=["pdf"])
    
    if uploaded_file is not None:
        # Load the model
        ner_model = load_ner_model()

        # Extract text from the uploaded PDF
        text = extract_text_from_pdf(uploaded_file)

        # Display the extracted text
        st.subheader("Extracted Text")
        st.write(text[:1000])  # Display only the first 1000 characters for preview

        # Extract named entities
        named_entities = extract_named_entities(text, ner_model)

        # Display named entities as JSON
        st.subheader("Extracted Named Entities")
        st.json(named_entities)

# Run the app
if __name__ == "__main__":
    main()
