# tools.py
import json
import os
from loguru import logger
import re

def load_sql_examples(file_path):
    """
    Load SQL examples from a JSON or JSONL file.
    
    Args:
        file_path (str): Path to the file containing SQL examples
        
    Returns:
        list: List of SQL examples
    """
    examples = []
    try:
        if os.path.exists(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                if file_path.endswith('.jsonl'):
                    examples = [json.loads(line) for line in f if line.strip()]
                else:
                    examples = json.load(f)
            logger.success(f"Loaded {len(examples)} SQL examples from {file_path}")
        else:
            logger.warning(f"Examples file not found: {file_path}")
    except Exception as e:
        logger.error(f"Error loading SQL examples: {str(e)}")
    
    return examples

def is_database_question(question):
    """
    Determine if a question is likely asking for database information.
    
    Args:
        question (str): The user's question
        
    Returns:
        bool: True if it's a database question, False otherwise
    """
    question_lower = question.lower().strip()
    
    db_keywords = [
        'query', 'sql', 'database', 'table', 
        'column', 'data', 'find', 'show me', 
        'list', 'how many', 'count', 'total',
        'average', 'sum', 'minimum', 'maximum',
        'top', 'rank', 'ordered by', 'group by',
        'report', 'statistics', 'analytics', 'metrics',
        'sales', 'customers', 'orders', 'products',
        'revenue', 'profit', 'loss', 'inventory',
        'user', 'client', 'purchase', 'transaction',
        'who is', 'what is the', 'when did', 'where are',
        'how much', 'which', 'why did', 'compare', 'between',
        'search for', 'look up', 'locate', 'identify',
        'summarize', 'aggregate', 'breakdown', 'analyze'
    ]
    
    db_patterns = [
        r'how many .+ (?:in|with|by|for)',
        r'what (?:is|are) the .+?(?:of|in|with|by|for)',
        r'who (?:is|are) .+?(?:in|with|by|for)',
        r'show (?:me|all|the) .+',
        r'list (?:all|the) .+',
        r'find .+',
        r'search for .+',
        r'get .+ (?:from|in|with|by|for)',
        r'display .+',
        r'(?:count|total|sum) .+',
        r'(?:average|avg) .+',
        r'(?:maximum|minimum|max|min) .+',
        r'top .+',
        r'(?:compare|comparing) .+',
        r'(?:analyze|analysis) .+',
        r'(?:statistics|stats) .+',
        r'(?:report|reports) .+',
        r'(?:trend|trends) .+',
        r'.+ (?:grouped by|ordered by|sorted by) .+',
        r'(?:overall|breakdown|summary) .+'
    ]
    
    time_indicators = [
        'today', 'yesterday', 'this week', 'last week', 
        'this month', 'last month', 'this year', 'last year',
        'quarter', 'recent', 'latest', 'current', 'past', 
        'previous', 'since', 'from', 'between', 'during', 'after', 'before'
    ]
    
    bi_terms = [
        'sales', 'revenue', 'profit', 'margin', 'cost', 'expense',
        'growth', 'decline', 'performance', 'kpi', 'metric',
        'customer', 'client', 'user', 'product', 'service',
        'conversion', 'retention', 'churn', 'acquisition',
        'segment', 'demographic', 'geography', 'location',
        'trend', 'forecast', 'prediction', 'projection',
        'distribution', 'allocation', 'portfolio', 'budget',
        'target', 'goal', 'benchmark', 'comparison'
    ]
    
    for keyword in db_keywords:
        if keyword in question_lower:
            logger.debug(f"Database keyword detected: {keyword}")
            return True
    
    for pattern in db_patterns:
        if re.search(pattern, question_lower):
            logger.debug(f"Database question pattern detected: {pattern}")
            return True
    
    has_time = any(indicator in question_lower for indicator in time_indicators)
    has_bi_term = any(term in question_lower for term in bi_terms)
    if has_time and has_bi_term:
        logger.debug(f"Temporal + business indicator detected")
        return True
    
    bi_term_count = sum(1 for term in bi_terms if term in question_lower)
    if bi_term_count >= 2:
        logger.debug(f"Multiple business terms detected: {bi_term_count}")
        return True
    
    starting_words = ['who', 'what', 'when', 'where', 'how', 'which', 'why', 'find', 'list', 'show']
    if any(question_lower.startswith(word) for word in starting_words):
        logger.debug(f"Query indicator starting word detected")
        return True
    
    logger.debug(f"Not classified as a database question: {question}")
    return False

def extract_question_type(question):
    """
    Extract the type of question to help guide SQL generation.
    
    Args:
        question (str): The user's question
        
    Returns:
        str: The question type (aggregation, comparison, ranking, etc.)
    """
    question_lower = question.lower()
    
    question_types = {
        'count': [r'how many', r'count', r'number of'],
        'sum': [r'total', r'sum', r'sum of', r'add up'],
        'average': [r'average', r'mean', r'avg'],
        'maximum': [r'maximum', r'max', r'highest', r'top', r'best'],
        'minimum': [r'minimum', r'min', r'lowest', r'worst', r'bottom'],
        'ranking': [r'rank', r'top \d+', r'bottom \d+', r'best \d+', r'worst \d+'],
        'comparison': [r'compare', r'versus', r'vs', r'difference', r'trend', r'growth'],
        'filtering': [r'where', r'which', r'find', r'search', r'filter', r'containing'],
        'grouping': [r'group', r'categorize', r'by each', r'per'],
        'time_series': [r'over time', r'trend', r'history', r'development'],
        'lookup': [r'details of', r'information on', r'data for', r'specifics about']
    }
    
    detected_types = []
    for q_type, patterns in question_types.items():
        for pattern in patterns:
            if re.search(pattern, question_lower):
                detected_types.append(q_type)
                break
    
    if len(detected_types) > 1:
        if 'ranking' in detected_types:
            return 'ranking'
        elif 'comparison' in detected_types:
            return 'comparison'
        elif any(t in detected_types for t in ['count', 'sum', 'average', 'maximum', 'minimum']):
            for agg_type in ['count', 'sum', 'average', 'maximum', 'minimum']:
                if agg_type in detected_types:
                    return agg_type
    
    return detected_types[0] if detected_types else 'general'

def save_query_results(query, sql, result, success=True):
    """
    Save query results for continuous improvement.
    
    Args:
        query (str): The user's question
        sql (str): The SQL query
        result (str): The query result
        success (bool): Whether the query was successful
    """
    try:
        log_dir = os.getenv("QUERY_LOG_DIR", "logs/queries")
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        status = "success" if success else "failed"
        filename = f"{log_dir}/query_{status}_{timestamp}.json"
        
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "query": query,
            "sql": sql,
            "result": result,
            "success": success
        }
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(log_entry, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Query log saved to {filename}")
    
    except Exception as e:
        logger.error(f"Error saving query log: {str(e)}")

def get_table_columns(db, table_name):
    """
    Get column information for a specific table.
    
    Args:
        db (SQLDatabase): Database connection
        table_name (str): Name of the table
        
    Returns:
        list: List of column information dictionaries
    """
    try:
        columns = db._get_table_columns(table_name)  # Use internal method if available
        if not columns:
            query = f"SHOW COLUMNS FROM {table_name};"  # For MariaDB
            result = db.run(query)
            columns = [{"name": row.split()[0], "type": row.split()[1]} for row in result.split('\n') if row]
        return columns
    except Exception as e:
        logger.error(f"Error getting column info for {table_name}: {str(e)}")
        return []

def get_sample_data(db, table_name, limit=5):
    """
    Get sample data from a table for better SQL generation.
    
    Args:
        db (SQLDatabase): Database connection
        table_name (str): Name of the table
        limit (int): Number of rows to retrieve
        
    Returns:
        str: Sample data as a string
    """
    try:
        query = f"SELECT * FROM {table_name} LIMIT {limit};"
        return db.run(query)
    except Exception as e:
        logger.error(f"Error getting sample data for {table_name}: {str(e)}")
        return ""