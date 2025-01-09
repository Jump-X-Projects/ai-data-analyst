import json
import tempfile
import csv
import os
import re
import streamlit as st
import pandas as pd
import duckdb
from phi.model.openai import OpenAIChat
from phi.assistant.duckdb import DuckDbAssistant as DuckDbAgent
from phi.tools.pandas import PandasTools

# Import visualization components
from components.visualization.data_viz import (
    analyze_data_for_visualization,
    prepare_data_for_visualization,
    format_data_for_visualization
)
from components.visualization.viz_component import create_visualization

# Initialize session state
if 'current_query_result' not in st.session_state:
    st.session_state.current_query_result = None
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'current_viz_suggestion' not in st.session_state:
    st.session_state.current_viz_suggestion = None
if 'current_sql' not in st.session_state:
    st.session_state.current_sql = None

def init_openai_client(api_key: str):
    """Initialize OpenAI client with API key"""
    os.environ["OPENAI_API_KEY"] = api_key
    return OpenAIChat(model="gpt-4", api_key=api_key)

def parse_gpt_response(response_content: str) -> dict:
    """
    Sanitize and parse GPT response, with fallback for malformed responses
    """
    try:
        # First try to parse as is
        parsed = json.loads(response_content)
        
        # Validate required fields
        if not isinstance(parsed, dict):
            raise ValueError("Response must be a dictionary")
        
        if 'answer' not in parsed:
            raise ValueError("Response missing 'answer' field")
            
        if not isinstance(parsed['answer'], dict):
            raise ValueError("'answer' must be a dictionary")
            
        if 'text' not in parsed['answer'] or 'sql' not in parsed['answer']:
            raise ValueError("Answer must contain 'text' and 'sql' fields")
        
        # Validate visualization suggestion if present
        if 'visualization_suggestion' in parsed:
            viz = parsed['visualization_suggestion']
            if not isinstance(viz, dict):
                raise ValueError("visualization_suggestion must be a dictionary")
            if 'type' not in viz or 'reason' not in viz:
                raise ValueError("visualization_suggestion must contain 'type' and 'reason'")
            if viz['type'] not in ['bar', 'line', 'scatter', 'pie']:
                viz['type'] = 'bar'  # Default to bar if invalid type
        
        return parsed
        
    except json.JSONDecodeError:
        # Try to extract SQL query and text from non-JSON response
        sql_match = re.search(r'```sql\s*(.*?)\s*```', response_content, re.DOTALL)
        sql_query = sql_match.group(1) if sql_match else response_content
        
        # Remove SQL block from text if found
        text_content = response_content
        if sql_match:
            text_content = response_content.replace(sql_match.group(0), '').strip()
        
        return {
            'answer': {
                'text': text_content,
                'sql': sql_query
            }
        }
    except ValueError as e:
        # If validation fails, create a minimal valid response
        st.warning(f"Warning: {str(e)}. Using default response format.")
        return {
            'answer': {
                'text': response_content,
                'sql': response_content
            }
        }

def get_system_prompt():
    """Generate system prompt"""
    return """You are an expert data analyst specializing in DuckDB SQL syntax.
    Generate a SQL query that strictly follows DuckDB syntax to solve the user's query.
    Return your response in this format:
    {
        "answer": {
            "text": "Your explanation here",
            "sql": "Your SQL query here"
        },
        "visualization_suggestion": {
            "type": "bar|line|scatter|pie",
            "reason": "Brief explanation of why this visualization would be helpful"
        }
    }"""

def preprocess_and_save(file):
    """Preprocess and save uploaded file"""
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

        return temp_path, df.columns.tolist(), df
    except Exception as e:
        st.error(f"Error processing file: {e}")
        return None, None, None

def execute_query(sql_query: str, file_path: str) -> pd.DataFrame:
    """Execute SQL query using DuckDB"""
    try:
        # Create a DuckDB connection
        conn = duckdb.connect()
        
        # Register the CSV file
        conn.execute(f"CREATE TABLE uploaded_data AS SELECT * FROM read_csv_auto('{file_path}')")
        
        # Execute the query and fetch results as DataFrame
        result = conn.execute(sql_query).fetchdf()
        
        # Close the connection
        conn.close()
        
        return result
    except Exception as e:
        raise Exception(f"Error executing query: {str(e)}")

def process_query_and_visualize(query_result, viz_suggestion=None):
    """Process query results and show visualization"""
    if query_result is not None:
        try:
            # Show regular query results
            st.write("Query Results:")
            st.dataframe(query_result)
            
            # Store the results in session state
            st.session_state.current_query_result = query_result
            st.session_state.current_viz_suggestion = viz_suggestion
            
            # Analyze data for visualization
            viz_info = analyze_data_for_visualization(query_result)
            
            # If there's a visualization suggestion from GPT, incorporate it
            if viz_suggestion and viz_suggestion.get('type') in viz_info['possible_viz']:
                viz_info['recommended_viz'] = viz_suggestion['type']
                viz_info['reason'] = viz_suggestion['reason']
            
            # Show visualization section
            st.write("---")
            st.write("📊 Data Visualization")
            
            # Create visualization
            create_visualization(query_result, viz_info)

        except Exception as e:
            st.error(f"Error creating visualization: {str(e)}")

def main():
    st.title("📊 Data Analyst Agent")

    # Sidebar for API keys
    with st.sidebar:
        st.header("API Keys")
        openai_key = st.text_input("Enter your OpenAI API key:", type="password")
        if openai_key:
            st.session_state.openai_key = openai_key
            # Initialize OpenAI client
            st.session_state.openai_client = init_openai_client(openai_key)
            st.success("API key saved!")
        else:
            st.warning("Please enter your OpenAI API key to proceed.")

    # File upload widget
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file is not None and "openai_key" in st.session_state:
        # Preprocess and save the uploaded file
        temp_path, columns, df = preprocess_and_save(uploaded_file)

        if temp_path and columns and df is not None:
            # Display the uploaded data
            st.write("Uploaded Data:")
            st.dataframe(df)
            st.write("Available columns:", columns)

            # Configure the semantic model
            semantic_model = {
                "tables": [{
                    "name": "uploaded_data",
                    "description": "Contains the uploaded dataset.",
                    "path": temp_path,
                    "format": "csv"
                }]
            }

            # Main query input
            user_query = st.text_area("Ask a query about the data:")

            if st.button("Submit Query"):
                if user_query.strip() == "":
                    st.warning("Please enter a query.")
                else:
                    try:
                        with st.spinner('Processing your query...'):
                            # Initialize DuckDB agent with the OpenAI client
                            duckdb_agent = DuckDbAgent(
                                model=st.session_state.openai_client,
                                semantic_model=json.dumps(semantic_model),
                                tools=[PandasTools()],
                                markdown=True,
                                system_prompt=get_system_prompt()
                            )

                            try:
                                # Get response from DuckDB agent
                                response = duckdb_agent.run(user_query)
                                
                                # Convert generator to string if needed
                                if hasattr(response, '__iter__') and not isinstance(response, (str, dict)):
                                    response_content = ''.join(list(response))
                                else:
                                    response_content = str(response)
                                
                                # Parse and validate response
                                parsed_response = parse_gpt_response(response_content)
                                
                                # Store analysis in session state
                                st.session_state.current_analysis = parsed_response['answer']['text']
                                st.session_state.current_sql = parsed_response['answer']['sql']
                                
                                try:
                                    # Execute query using DuckDB directly
                                    query_result = execute_query(parsed_response['answer']['sql'], temp_path)
                                    
                                    if query_result is not None and not query_result.empty:
                                        process_query_and_visualize(
                                            query_result,
                                            parsed_response.get('visualization_suggestion')
                                        )
                                    else:
                                        st.warning("Query returned no results.")
                                        
                                except Exception as e:
                                    st.error(f"Error executing SQL query: {str(e)}")
                                    st.error("Please check the SQL query syntax.")
                                    
                            except Exception as e:
                                st.error(f"Error processing GPT response: {str(e)}")
                                st.error("Please try rephrasing your query.")
                                
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        st.error("Please check your input and try again.")

            # Display stored results if they exist
            if st.session_state.current_analysis:
                st.write("### Analysis")
                st.write(st.session_state.current_analysis)
                
                st.write("### SQL Query")
                st.code(st.session_state.current_sql, language='sql')
                
                if st.session_state.current_query_result is not None:
                    process_query_and_visualize(
                        st.session_state.current_query_result,
                        st.session_state.current_viz_suggestion
                    )

if __name__ == "__main__":
    main()