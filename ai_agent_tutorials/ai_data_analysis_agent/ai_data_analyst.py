import json
import tempfile
import csv
import streamlit as st
import pandas as pd
from phi.model.openai import OpenAIChat
from phi.agent.duckdb import DuckDbAgent
from phi.tools.pandas import PandasTools
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from langchain_text_splitters import MarkdownTextSplitter
import re

# Function to load and process documentation
def load_and_process_docs():
    # Load the documentation
    with open('data/duckdb_docs.json', 'r') as f:
        docs_data = json.load(f)
    
    # Extract markdown content
    docs_content = []
    for doc in docs_data:
        if 'markdown' in doc:
            docs_content.append(doc['markdown'])
    
    # Combine all docs
    combined_docs = "\n\n".join(docs_content)
    
    # Split into chunks
    text_splitter = MarkdownTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_text(combined_docs)
    
    return chunks

def create_similarity_index(chunks):
    # Load the model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Create embeddings
    embeddings = model.encode(chunks)
    
    return model, embeddings, chunks

def get_relevant_docs(query, model, stored_embeddings, chunks, k=3):
    # Get query embedding and find similar chunks
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, stored_embeddings)[0]
    top_k_indices = similarities.argsort()[-k:][::-1]
    
    # Return concatenated relevant chunks
    return "\n\n".join([chunks[i] for i in top_k_indices])

# Function to preprocess and save the uploaded file
def preprocess_and_save(file):
    try:
        # Read the uploaded file into a DataFrame
        if file.name.endswith('.csv'):
            df = pd.read_csv(file, encoding='utf-8', na_values=['NA', 'N/A', 'missing'])
        elif file.name.endswith('.xlsx'):
            df = pd.read_excel(file, na_values=['NA', 'N/A', 'missing'])
        else:
            st.error("Unsupported file format. Please upload a CSV or Excel file.")
            return None, None, None

        # Ensure string columns are properly quoted
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].astype(str).replace({r'"': '""'}, regex=True)

        # Parse dates and numeric columns
        for col in df.columns:
            if 'date' in col.lower():
                df[col] = pd.to_datetime(df[col], errors='coerce')
            elif df[col].dtype == 'object':
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    # Keep as is if conversion fails
                    pass

        # Create a temporary file to save the preprocessed data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as temp_file:
            temp_path = temp_file.name
            # Save the DataFrame to the temporary CSV file with quotes around string fields
            df.to_csv(temp_path, index=False, quoting=csv.QUOTE_ALL)

        return temp_path, df.columns.tolist(), df  # Return the DataFrame as well
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

# Streamlit app
st.title("📊 Data Analyst Agent")

# Sidebar for API keys
with st.sidebar:
    st.header("API Keys")
    openai_key = st.text_input("Enter your OpenAI API key:", type="password")
    if openai_key:
        st.session_state.openai_key = openai_key
        st.success("API key saved!")
    else:
        st.warning("Please enter your OpenAI API key to proceed.")

# File upload widget
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file is not None and "openai_key" in st.session_state:
    # Initialize similarity search if not in session state
    if "similarity_model" not in st.session_state:
        with st.spinner('Setting up documentation search...'):
            chunks = load_and_process_docs()
            model, embeddings, doc_chunks = create_similarity_index(chunks)
            st.session_state.similarity_model = model
            st.session_state.stored_embeddings = embeddings
            st.session_state.chunks = doc_chunks
    
    # Preprocess and save the uploaded file
    temp_path, columns, df = preprocess_and_save(uploaded_file)

    if temp_path and columns and df is not None:
        # Display the uploaded data as a table
        st.write("Uploaded Data:")
        st.dataframe(df)  # Use st.dataframe for an interactive table

        # Display the columns of the uploaded data
        st.write("Uploaded columns:", columns)

        # Configure the semantic model with the temporary file path
        semantic_model = {
            "tables": [
                {
                    "name": "uploaded_data",
                    "description": "Contains the uploaded dataset.",
                    "path": temp_path,
                    "format": "csv"
                }
            ]
        }

        # Main query input widget
        user_query = st.text_area("Ask a query about the data:")

        # Add info message about terminal output
        st.info("💡 Check your terminal for a clearer output of the agent's response")

        if st.button("Submit Query"):
            if user_query.strip() == "":
                st.warning("Please enter a query.")
            else:
                try:
                    # Show loading spinner while processing
                    with st.spinner('Processing your query...'):
                        # Get relevant documentation using similarity search
                        relevant_docs = get_relevant_docs(
                            user_query,
                            st.session_state.similarity_model,
                            st.session_state.stored_embeddings,
                            st.session_state.chunks
                        )
                        
                        # Create dynamic system prompt with relevant documentation
                        system_prompt = f"""You are an expert data analyst specializing in DuckDB SQL syntax.
                        Here is the relevant DuckDB documentation for this query:
                        {relevant_docs}
                        
                        Generate a SQL query that strictly follows DuckDB syntax to solve the user's query.
                        Return only the SQL query, enclosed in ```sql ``` and give the final answer."""
                        
                        # Initialize the DuckDbAgent with documentation-aware prompt
                        duckdb_agent = DuckDbAgent(
                            model=OpenAIChat(model="gpt-4", api_key=st.session_state.openai_key),
                            semantic_model=json.dumps(semantic_model),
                            tools=[PandasTools()],
                            markdown=True,
                            add_history_to_messages=False,  # Disable chat history
                            followups=False,  # Disable follow-up queries
                            read_tool_call_history=False,  # Disable reading tool call history
                            system_prompt=system_prompt,
                        )

                        # Get the response from DuckDbAgent
                        response1 = duckdb_agent.run(user_query)

                        # Extract the content from the RunResponse object
                        if hasattr(response1, 'content'):
                            response_content = response1.content
                        else:
                            response_content = str(response1)
                        response = duckdb_agent.print_response(
                            user_query,
                            stream=True,
                        )

                        # Display the response in Streamlit
                        st.markdown(response_content)

                except Exception as e:
                    st.error(f"Error generating response from the DuckDbAgent: {e}")
                    st.error("Please try rephrasing your query or check if the data format is correct.")