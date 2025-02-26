from langchain.tools import tool
from langchain_community.utilities import SQLDatabase
from dotenv import load_dotenv
import os

# Load environment variables once
load_dotenv()

# First define your database connection - Get the actual value from environment variable
database_uri = os.getenv("DATABASE_URI")
if not database_uri:
    raise ValueError("DATABASE_URI environment variable is not set")

db = SQLDatabase.from_uri(database_uri)

@tool
def execute_sql_query(query):
    """Execute a SQL query against the database.
    
    Args:
        query: SQL query string to execute
        
    Returns:
        Query results or error message
    """
    try:
        result = db.run(query)
        return result
    except Exception as e:
        return f"Error executing query: {str(e)}"


@tool
def generate_last_month_invoice():
    """Generates an invoice report for transactions from the previous month."""
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
        pc.StkCode_v
    FROM 
        nex_valiant.tbl_preceipt_item pi
        INNER JOIN nex_valiant.tbl_preceipt_txn pt ON pt.TxnId_i = pi.TxnId_i
        INNER JOIN nex_valiant.suppliers s ON s.SuppId_i = pt.SuppId_i
        INNER JOIN nex_valiant.product_codes pc ON pc.StkId_i = pi.StkId_i
    WHERE 
        pt.TxnDate_dd BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH) AND CURRENT_DATE()
    ORDER BY 
        pt.TxnDate_dd DESC
    """
    return execute_sql_query(query)


# If you want to test the file directly
if __name__ == "__main__":
    print("Tools defined successfully")
    # You could also add test code here if you want