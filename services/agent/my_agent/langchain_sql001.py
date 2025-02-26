# Do not remove this link.
# https://python.langchain.com/docs/tutorials/sql_qa/
# Build a Question/Answering system over SQL data

import os
import re
import json
import pprint
from tabulate import tabulate
from typing import Dict, TypedDict, Optional, Union, List, Any
from langchain_community.utilities import SQLDatabase
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

# Import configuration (assuming config.py exists with openai_api_key and database_uri)
from config import config


# Define the state type
class State(TypedDict):
    question: str
    query: str
    result: Union[str, List[tuple]]  # Raw list of tuples
    answer: str
    error: Optional[str]

# Purchase Receipts Tool
def list_all_purchase_receipts(db: SQLDatabase, llm: ChatOpenAI) -> List[Dict[str, Any]]:
    """
    Retrieve a comprehensive list of purchase receipts with detailed item information.
    """
    query = """
    SELECT 
        pt.TxnId_i, 
        s.IntId_v, 
        s.SuppAbbrev_v, 
        pt.DocRef_v, 
        pt.SuppRef_v, 
        pt.TxnDate_dd, 
        pi.ItemId_i, 
        pi.StkId_i, 
        pi.Batch_v, 
        pc.StkCode_v, 
        pc.ProdName_v, 
        c.CategoryCode_v, 
        ac.AcctcategoryCode_v, 
        pi.ActuaLPrice_d, 
        pi.LineTotal_d, 
        pi.TxnQty_d, 
        u.UomCode_v,
        md.MetafieldVal_v
    FROM 
        nex_valiant.tbl_preceipt_item pi
        INNER JOIN nex_valiant.tbl_preceipt_txn pt ON pt.TxnId_i = pi.TxnId_i 
        INNER JOIN nex_valiant.tbl_product_code pc ON pi.ItemId_i = pc.ItemId_i AND pi.StkId_i = pc.StkId_i
        INNER JOIN nex_valiant.tbl_product_master pm ON pi.ItemId_i = pm.ItemId_i
        INNER JOIN nex_valiant.tbl_metadata md ON pi.ItemId_i = md.ItemId_i AND md.MetafieldId_i = 5
        INNER JOIN nex_valiant.tbl_uom u ON pi.TxnuomId_i = u.UomId_i
        INNER JOIN nex_valiant.tbl_supplier s ON pt.SuppId_i = s.SuppId_i
        INNER JOIN nex_valiant.tbl_category c ON pm.CategoryId_i = c.CategoryId_i
        INNER JOIN nex_valiant.tbl_acctcategory ac ON pm.AcctcategoryId_i = ac.AcctcategoryId_i
    ORDER BY pt.TxnId_i DESC
    """
    
    try:
        results = db.run(query, fetch="all")
        return results
    except Exception as e:
        print(f"Error retrieving purchase receipts: {e}")
        return []

def summarize_purchase_receipts(results: List[tuple], llm: ChatOpenAI) -> str:
    """
    Generate a summary of purchase receipts using the language model with enhanced tabular display.
    """
    if not results:
        return "No purchase receipts found."
    
    print(f"DEBUG: Number of receipt tuples: {len(results)}")
    
    # Define column headers for purchase receipts
    receipt_headers = ["TxnId", "Supplier", "Date", "Product", "Quantity", "Total"]
    print(f"DEBUG: First tuple: {pretty_print_value(results[0], headers=receipt_headers)}")
    
    # Map tuple indices to meaningful names
    try:
        receipts = [
            {
                "TxnId_i": receipt[0],
                "SuppAbbrev_v": receipt[1],
                "TxnDate_dd": receipt[2],
                "ProdName_v": receipt[3],
                "TxnQty_d": receipt[4],
                "LineTotal_d": receipt[5]
            }
            for receipt in results
        ]
    except IndexError as e:
        return f"Error: Tuple structure mismatch - {str(e)}"
    
    print(f"DEBUG: First receipt dict: {pretty_print_value(receipts[0])}")
    
    # Prepare context for summarization
    try:
        total_value = sum(float(receipt['LineTotal_d']) for receipt in receipts)
        unique_suppliers = len(set(receipt['SuppAbbrev_v'] for receipt in receipts))
        
        # Create a tabular view of the sample transactions
        sample_data = []
        for receipt in receipts[:5]:
            sample_data.append([
                receipt['TxnId_i'],
                receipt['SuppAbbrev_v'],
                receipt['TxnDate_dd'],
                receipt['ProdName_v'],
                receipt['TxnQty_d'],
                f"${float(receipt['LineTotal_d']):.2f}"
            ])
        
        sample_table = tabulate(sample_data, headers=receipt_headers, tablefmt="grid")
        
        context = f"""
        Total Purchase Receipts: {len(receipts)}
        
        Key Insights:
        - Total Transactions: {len(receipts)}
        - Unique Suppliers: {unique_suppliers}
        - Total Value: ${total_value:.2f}
        
        Sample Transactions:
        {sample_table}
        """
        print(f"DEBUG: Context prepared: {context[:200]}...")  # Truncate for brevity
    except Exception as e:
        return f"Error preparing summary context: {str(e)}"
    
    # Generate summary using LLM
    try:
        summary_prompt = f"""
        Provide a concise, professional summary of the following purchase receipt data:
        
        {context}
        
        Focus on key business insights, trends, and notable observations.
        """
        summary_response = llm.invoke(summary_prompt)
        return summary_response.content
    except Exception as e:
        return f"Error generating summary: {str(e)}"

# Initialize the database
def initialize_database(database_uri: str) -> SQLDatabase:
    """Initialize and validate the database connection."""
    if not database_uri:
        raise ValueError("DATABASE_URI is not set.")
    
    try:
        db = SQLDatabase.from_uri(database_uri)
        db.get_usable_table_names()
        return db
    except Exception as e:
        raise ConnectionError(f"Failed to connect to database: {str(e)}")

# Initialize the language model
def initialize_llm(api_key: str) -> ChatOpenAI:
    """Initialize and validate the language model."""
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set.")
    
    model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    temperature = float(os.getenv("OPENAI_TEMPERATURE", "0.0"))
    
    try:
        return ChatOpenAI(api_key=api_key, model=model_name, temperature=temperature)
    except Exception as e:
        raise ConnectionError(f"Failed to initialize language model: {str(e)}")

# Define the prompt templates
query_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Given an input question, create a syntactically correct {dialect} query to run to help find the answer. 
            Unless the user specifies in their question a specific number of examples they wish to obtain, always limit 
            your query to at most {top_k} results. You can order the results by a relevant column to return the most 
            interesting examples in the database.
            
            Never query for all the columns from a specific table, only ask for a few relevant columns given the 
            question.
            
            Pay attention to use only the column names that you can see in the schema description. Be careful to not 
            query for columns that do not exist. Also, pay attention to which column is in which table.
            
            Only use the following tables:
            {table_info}
            
            If the question cannot be answered with an SQL query, return an empty query.
            
            Question: {input}
            """,
        ),
        ("human", "{input}"),
    ]
)

answer_prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            Given the question, the SQL query used, and the SQL query result, provide a natural language answer.
            
            If the query is empty or cannot be executed, provide a helpful response explaining that 
            the question cannot be answered using the current database.
            
            Format the answer in a clear, concise manner. If the result contains numerical data, provide context and interpretation.
            If the result is empty, explain what that likely means in the context of the question.
            
            Question: {question}
            SQL Query: {query}
            SQL Result: {result}
            """,
        ),
    ]
)

def extract_sql_from_response(response_text: str) -> str:
    """Extract SQL query from a markdown-formatted response."""
    sql_pattern = r"```(?:sql)?\s*(.*?)```"
    matches = re.findall(sql_pattern, response_text, re.DOTALL)
    
    if matches:
        return matches[0].strip()
    return response_text.strip()

def write_query(state: State, db: SQLDatabase, llm: ChatOpenAI) -> State:
    """Generate SQL query to fetch information."""
    try:
        if "purchase receipts" in state["question"].lower():
            state["query"] = """
            SELECT 
                pt.TxnId_i, 
                s.SuppAbbrev_v, 
                pt.TxnDate_dd, 
                pc.ProdName_v, 
                pi.TxnQty_d, 
                pi.LineTotal_d
            FROM 
                nex_valiant.tbl_preceipt_item pi
                INNER JOIN nex_valiant.tbl_preceipt_txn pt ON pt.TxnId_i = pi.TxnId_i 
                INNER JOIN nex_valiant.tbl_product_code pc ON pi.ItemId_i = pc.ItemId_i AND pi.StkId_i = pc.StkId_i
                INNER JOIN nex_valiant.tbl_supplier s ON pt.SuppId_i = s.SuppId_i
            ORDER BY pt.TxnDate_dd DESC
            LIMIT 20
            """
            return state

        prompt = query_prompt_template.format_messages(
            dialect=db.dialect,
            top_k=10,
            table_info=db.get_table_info(),
            input=state["question"],
        )
        
        result = llm.invoke(prompt)
        raw_query = extract_sql_from_response(result.content)
        
        state["query"] = raw_query
        state["error"] = None
        return state
    except Exception as e:
        state["error"] = f"Failed to generate query: {str(e)}"
        return state

def execute_query(state: State, db: SQLDatabase) -> State:
    """Execute the generated SQL query and return the results."""
    if state.get("error"):
        return state
    
    if not state["query"].strip():
        state["result"] = []
        return state
    
    try:
        result = db.run(state["query"], fetch="all")
        state["result"] = result
        state["error"] = None
        return state
    except Exception as e:
        state["result"] = []
        state["error"] = f"Query execution failed: {str(e)}"
        return state

def generate_answer(state: State, db: SQLDatabase, llm: ChatOpenAI) -> State:
    """Generate a natural language answer based on SQL results."""
    if state.get("error"):
        return state
    
    try:
        if "purchase receipts" in state["question"].lower():
            receipts = state["result"] if state["result"] else []
            print(f"DEBUG: Passing {len(receipts)} receipts to summarize")
            summary = summarize_purchase_receipts(receipts, llm)
            state["answer"] = summary
            return state

        if not state["query"].strip() or not state["result"]:
            state["answer"] = (
                "I apologize, but I cannot find a direct answer to your question in the current database. "
                f"Available tables are: {', '.join(db.get_usable_table_names())}. "
                "Could you rephrase your question or ask about the available tables?"
            )
            return state
        
        prompt = answer_prompt_template.format_messages(
            question=state["question"],
            query=state["query"],
            result=pretty_print_value(state["result"])
        )
        
        result = llm.invoke(prompt)
        state["answer"] = result.content
        return state
    except Exception as e:
        state["error"] = f"Failed to generate answer: {str(e)}"
        return state

def process_question(question: str, db: SQLDatabase, llm: ChatOpenAI) -> State:
    """Process a natural language question through the entire pipeline."""
    state: State = {
        "question": question,
        "query": "",
        "result": [],
        "answer": "",
        "error": None
    }
    
    state = write_query(state, db, llm)
    state = execute_query(state, db)
    state = generate_answer(state, db, llm)
    return state

def pretty_print_value(value, headers=None):
    """
    Format different data structures for pretty printing with emphasis on tabular display:
    - For tuples/lists of tuples: Convert to tabular format with proper headers
    - For dictionaries/lists of dictionaries: Convert to tabular format
    - For strings and other primitive values: Return as is
    
    Args:
        value: The data to format
        headers: Optional column headers for tabular data
    """
    try:
        # Case 1: List of tuples (SQL results)
        if isinstance(value, list) and value and isinstance(value[0], tuple):
            if len(value) > 0:
                # If headers are not provided, use generic column names
                if not headers:
                    headers = [f"Column {i+1}" for i in range(len(value[0]))]
                return "\n" + tabulate(value, headers=headers, tablefmt="grid")
            return "[]"
        
        # Case 2: Dictionary or list of dictionaries - convert to tabular format
        elif isinstance(value, dict):
            # Single dictionary - display as a key-value table
            rows = [[k, v] for k, v in value.items()]
            return "\n" + tabulate(rows, headers=["Field", "Value"], tablefmt="grid")
            
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            # Extract all possible keys as headers
            all_keys = set()
            for item in value:
                all_keys.update(item.keys())
            headers = sorted(list(all_keys))
            
            # Build rows with values aligned to headers
            rows = []
            for item in value:
                row = [item.get(key, "") for key in headers]
                rows.append(row)
            
            return "\n" + tabulate(rows, headers=headers, tablefmt="grid")
        
        # Case 3: Simple string
        elif isinstance(value, str):
            return value
        
        # Case 4: Other types (default handling)
        else:
            return str(value)
    except Exception as e:
        return f"Error formatting value: {str(e)}"

def display_results(state: State) -> None:
    """Display the results of the query processing pipeline with enhanced tabular formatting."""
    print("\n" + "="*80)
    print(f"Question: {state['question']}")
    print("="*80)
    
    print("\nAnswer:")
    print("-"*80)
    print(state["answer"] or "No answer generated.")
    
    if state["query"].strip():
        print("\nGenerated SQL Query:")
        print("-"*80)
        print(state["query"])
        
        if state["result"]:
            print("\nQuery Result:")
            print("-"*80)
            
            # Try to extract column names from query
            headers = extract_column_names_from_query(state["query"])
            print(pretty_print_value(state["result"], headers=headers))
    
    if state.get("error"):
        print("\nError:")
        print("-"*80)
        print(state["error"])
    
    print("="*80)

def extract_column_names_from_query(query: str) -> List[str]:
    """
    Extract column names from a SELECT SQL query.
    
    Args:
        query: SQL query string
        
    Returns:
        List of column names or empty list if extraction fails
    """
    try:
        # Normalize query - remove newlines and extra spaces
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Extract the part between SELECT and FROM
        select_pattern = re.compile(r'SELECT\s+(.*?)\s+FROM', re.IGNORECASE)
        match = select_pattern.search(query)
        
        if not match:
            return []
            
        columns_part = match.group(1)
        
        # Split by commas and clean each column name
        columns = []
        for col in columns_part.split(','):
            col = col.strip()
            
            # Handle "table.column" format
            if '.' in col:
                table, col_name = col.split('.')
                col = col_name
            
            # Handle "column AS alias" format
            if ' AS ' in col.upper():
                col = col.split(' AS ')[1]
            elif ' as ' in col:
                col = col.split(' as ')[1]
                
            # Remove any backticks, brackets or quotes
            col = re.sub(r'[`\[\]\'"]', '', col)
            
            columns.append(col)
            
        return columns
    except Exception:
        # If anything goes wrong, return empty list
        return []

def main(
        openai_api_key: str,
        database_uri: str,
        langsmith_project: str = None,
        langsmith_api_key: str = None,
) -> None:
    """Main function to run the SQL agent."""
    try:
        db = initialize_database(database_uri)
        llm = initialize_llm(openai_api_key)

        print(f"Database dialect: {db.dialect}")
        print(f"Available tables: {', '.join(db.get_usable_table_names())}")
        
        while True:
            question = input("> ")
            
            if question.lower() in ["exit", "quit"]:
                break
            
            if not question.strip():
                continue
            
            state = process_question(question, db, llm)
            display_results(state)
            
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"Unexpected error: {str(e)}")

if __name__ == "__main__":
    main(
        openai_api_key=config.openai_api_key,
        database_uri=config.database_uri,
    )