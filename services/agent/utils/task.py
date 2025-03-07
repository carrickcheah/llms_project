########################################################################################
# task.py
# Contents:
# 1. extract_relevant_tables: Identifies tables related to user question.
# 2. find_similar_examples: Uses embeddings to find similar examples.
# 3. generate_dynamic_sql: Creates SQL based on tables and examples.
# 4. execute_sql_query: Executes SQL query and returns result.
# 5. format_response: Formats SQL results into user-friendly responses.
# 6. evaluate_sql_quality: Scores SQL quality for ensemble selection.
# 7. extract_tables_from_sql: Helper to extract tables from SQL queries.
# 8. clean_sql_query: Helper to clean SQL queries.
# 9. validate_sql_with_llm: Validates SQL query using LLM.
# 10. search_examples_for_sql: Searches for exact match examples in file.
# 11. execute_sql_with_no_data_handling: Executes SQL with no-data fallback
########################################################################################

# task.py
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
import re
import json
from difflib import SequenceMatcher
from langchain_core.output_parsers import StrOutputParser
from langgraph.func import task


########################################################################################
##     1. extract_relevant_tables: Identifies tables related to user question         ##
########################################################################################

def extract_relevant_tables(question, db):
    """
    Extract tables that might be relevant to the user's question using multiple strategies.

    Args:
        question (str): The user's question
        db (SQLDatabase): Database connection

    Returns:
        list: List of table names that might be relevant
    """
    all_tables = db.get_usable_table_names()
    question_lower = question.lower()
    relevance_scores = {}  # To track relevance scores for tables

    # Strategy 1: Direct table name matching
    for table in all_tables:
        table_lower = table.lower()
        if table_lower in question_lower:
            relevance_scores[table] = relevance_scores.get(table, 0) + 15  # High weight for direct matches

    # Strategy 2: Word-level matching with semantic expansion
    keywords = {
        "customer": ["customer", "client", "user", "buyer", "purchaser", "account", "member", "shopper", "patron", "consumer"],
        "order": ["order", "purchase", "transaction", "buy", "sale", "invoice", "receipt", "acquisition"],
        "product": ["product", "item", "goods", "merchandise", "inventory", "stock", "commodity"],
        "shipping": ["ship", "shipping", "delivery", "transport", "carrier", "freight", "logistics", "mail", "post"],
        "payment": ["payment", "pay", "price", "cost", "fee", "charge", "billing", "invoice", "transaction", "money"],
        "location": ["country", "city", "region", "area", "location", "address", "place", "territory", "zone"],
        "time": ["date", "time", "period", "month", "year", "quarter", "week", "day", "season", "duration"],
        "category": ["category", "type", "class", "group", "classification", "genre", "kind", "style"],
        "user": ["user", "account", "profile", "member", "login", "access", "permission", "identity"],
        "supplier": ["supplier", "vendor", "provider", "manufacturer", "distributor", "source", "producer"],
        "sport": ["sport", "game", "activity", "exercise", "competition", "athletic", "fitness", "event"]
    }

    for keyword_category, synonyms in keywords.items():
        if any(synonym in question_lower for synonym in synonyms):
            for table in all_tables:
                table_lower = table.lower()
                if keyword_category in table_lower or any(synonym in table_lower for synonym in synonyms):
                    relevance_scores[table] = relevance_scores.get(table, 0) + 8

    # Strategy 3: n-gram matching
    table_ngrams = {}
    for table in all_tables:
        words = re.findall(r'\w+', table.lower())
        ngrams = words + [' '.join(words[i:i+2]) for i in range(len(words)-1)]
        table_ngrams[table] = ngrams

    question_words = set(re.findall(r'\w+', question_lower))
    for table, ngrams in table_ngrams.items():
        for ngram in ngrams:
            if ngram in question_lower or ngram in question_words:
                relevance_scores[table] = relevance_scores.get(table, 0) + 5

    # Strategy 4: Handle specific query patterns
    patterns = {
        r'most (\w+)': ["count", "summary", "aggregate", "max", "highest"],
        r'average (\w+)': ["average", "mean", "stat", "avg"],
        r'total (\w+)': ["sum", "total", "amount", "overall"],
        r'shipping method': ["ship", "carrier", "transport", "delivery", "method"],
        r'customer (?:from|who) (\w+)': ["customer", "address", "location", "client"],
        r'product in (\w+)': ["product", "inventory", "category", "item"],
        r'orders? in (\w+)': ["order", "transaction", "date", "purchase"],
        r'revenue from (\w+)': ["order", "invoice", "payment", "sales", "revenue"],
        r'top (\w+)': ["rank", "best", "highest", "max", "leader"],
        r'(compare|comparison)': ["statistic", "analysis", "report"],
        r'recent (\w+)': ["date", "time", "latest", "recent"]
    }

    for pattern, related_concepts in patterns.items():
        if re.search(pattern, question_lower):
            for table in all_tables:
                table_lower = table.lower()
                if any(concept in table_lower for concept in related_concepts):
                    relevance_scores[table] = relevance_scores.get(table, 0) + 6

    if "carricksport" in all_tables and any(word in question_lower for word in ["ship", "method", "country", "sport", "product"]):
        relevance_scores["carricksport"] = relevance_scores.get("carricksport", 0) + 10

    # Strategy 5: Entity type recognition
    entity_types = {
        "time": ["year", "month", "quarter", "week", "date", "period", "day", "hour", "minute", "season"],
        "product": ["product", "item", "good", "merchandise", "sku", "commodity", "stock"],
        "customer": ["customer", "client", "buyer", "user", "consumer", "purchaser", "shopper", "patron"],
        "location": ["country", "city", "state", "region", "province", "area", "territory", "address", "location"],
        "measurement": ["price", "cost", "amount", "quantity", "number", "total", "sum", "count", "rate", "percentage", "value"],
        "action": ["buy", "purchase", "order", "ship", "deliver", "return", "cancel", "refund", "exchange", "pay"]
    }

    question_tokens = re.findall(r'\b\w+\b', question_lower)
    detected_entities = {}
    for entity_type, keywords in entity_types.items():
        for token in question_tokens:
            if token in keywords:
                detected_entities[entity_type] = detected_entities.get(entity_type, 0) + 1

    entity_table_mapping = {
        "time": ["date", "time", "calendar", "period", "schedule"],
        "product": ["product", "item", "inventory", "catalog", "goods", "carricksport"],
        "customer": ["customer", "client", "user", "account", "member", "profile"],
        "location": ["location", "address", "geography", "region", "place", "country"],
        "measurement": ["order", "sale", "transaction", "invoice", "payment", "price"],
        "action": ["order", "transaction", "activity", "event", "log", "history"]
    }

    for entity_type, count in detected_entities.items():
        if count > 0:
            related_table_patterns = entity_table_mapping.get(entity_type, [])
            for table in all_tables:
                table_lower = table.lower()
                if any(pattern in table_lower for pattern in related_table_patterns):
                    relevance_scores[table] = relevance_scores.get(table, 0) + 4 * count

    # Sort tables by relevance score
    sorted_tables = sorted(relevance_scores.items(), key=lambda x: x[1], reverse=True)
    relevant_tables = [table for table, score in sorted_tables if score > 0]

    if not relevant_tables:
        logger.warning(f"No tables found relevant for question: {question}")
        default_tables = [t for t in all_tables if any(term in t.lower() for term in
                          ["customer", "order", "product", "user", "transaction", "main", "sport", "carrick"])]
        if default_tables:
            logger.info(f"Using default central tables as fallback: {', '.join(default_tables[:3])}")
            return default_tables[:3]
        else:
            logger.info("Using first 3 tables as last resort")
            return all_tables[:3]

    MAX_TABLES = 5
    if len(relevant_tables) > MAX_TABLES:
        logger.info(f"Limiting from {len(relevant_tables)} to top {MAX_TABLES} most relevant tables")
        relevant_tables = relevant_tables[:MAX_TABLES]

    logger.info(f"Extracted {len(relevant_tables)} relevant tables: {', '.join(relevant_tables)}")
    return relevant_tables

########################################################################################
##      2. find_similar_examples: Uses embeddings to find similar examples            ##
########################################################################################

@task
def find_similar_examples(question, vector_store, embeddings):
    """
    Find the most similar examples to the user's question using vector embeddings.

    Args:
        question (str): The user's question
        vector_store: Vector database store
        embeddings: Embedding model

    Returns:
        list: List of similar examples with their SQL queries
    """
    logger.info(f"Finding similar examples for: {question}")

    try:
        retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        retrieved_docs = retriever.invoke(question)

        similar_examples = []
        if retrieved_docs:
            logger.success(f"Retrieved {len(retrieved_docs)} similar examples")
            for i, doc in enumerate(retrieved_docs):
                example_question = doc.page_content
                example_sql = doc.metadata.get("sql_query", "") if isinstance(doc.metadata, dict) else str(doc.metadata)
                score = doc.metadata.get('score', 0) if isinstance(doc.metadata, dict) else 0
                example = {
                    "question": example_question,
                    "sql": example_sql,
                    "score": score,
                    "rank": i + 1
                }
                similar_examples.append(example)
                logger.info(f"Example #{i+1}: '{example_question[:80]}...' (score: {score:.4f})")
        else:
            logger.warning("No similar examples found")

        return similar_examples

    except Exception as e:
        logger.error(f"Error finding similar examples: {str(e)}")
        return []

########################################################################################
##      3. generate_dynamic_sql: Creates SQL based on tables and examples             ##
########################################################################################

@task
def generate_dynamic_sql(question, relevant_tables, similar_examples, db, llm):
    """
    Generate an SQL query using relevant tables and similar examples.

    Args:
        question (str): The user's question
        relevant_tables (list): List of relevant table names
        similar_examples (list): List of similar examples with SQL queries
        db (SQLDatabase): Database connection
        llm: Language model

    Returns:
        str: Generated SQL query
    """
    try:
        table_info = db.get_table_info(relevant_tables)
        examples_text = "\n".join([f"- Q: {ex['question']}\n  SQL: {ex['sql']}" for ex in similar_examples[:3]])
        prompt = ChatPromptTemplate.from_template("""
        System: You are an expert SQL data analyst.
        Given an input question, create a syntactically correct {dialect} query.
        Return only the SQL query.
        Question: "{question}"
        Available tables: {table_info}
        Similar examples:
        {examples_text}
        """)
        chain = prompt | llm | StrOutputParser()
        sql_query = chain.invoke({"dialect": db.dialect, "question": question, "table_info": table_info, "examples_text": examples_text})
        logger.info(f"Generated SQL: {sql_query}")
        return sql_query
    except Exception as e:
        logger.error(f"Error generating SQL: {str(e)}")
        return f"Error: {str(e)}"

########################################################################################
##      4. execute_sql_query: Executes SQL query and returns result                   ##
########################################################################################

@task
def execute_sql_query(sql_query: str, db: SQLDatabase) -> str:
    """Executes the SQL query and returns the result."""
    logger.info(f"Executing SQL: {sql_query}")
    try:
        result = db.run(sql_query)
        return result
    except Exception as e:
        logger.error(f"SQL execution failed: {str(e)}")
        return f"Error executing query: {str(e)}"

########################################################################################
##      5. format_response: Formats SQL results into user-friendly responses          ##
########################################################################################

@task
def format_response(question, sql_query, result, llm):
    """
    Format the SQL query result into a user-friendly response.

    Args:
        question (str): The user's question
        sql_query (str): The executed SQL query
        result (str): The raw SQL result
        llm: Language model

    Returns:
        str: Formatted response
    """
    try:
        prompt = ChatPromptTemplate.from_template("""
        Given an input question and its executed SQL result, return the answer with column names explored from the query.
        Use the following format:

        Question: "{question}"
        SQLQuery: "{sql_query}"
        SQLResult: "{result}"
        Answer: "Final answer incorporating column names"
        Insight: "Optimize the Answer into a simple report, approximately 20 words"
        """)
        chain = prompt | llm | StrOutputParser()
        response = chain.invoke({"question": question, "sql_query": sql_query, "result": result})
        return response
    except Exception as e:
        logger.error(f"Error formatting response: {str(e)}")
        return f"Error: {str(e)}"

########################################################################################
##      6. evaluate_sql_quality: Scores SQL quality for ensemble selection            ##
########################################################################################

def evaluate_sql_quality(sql_query, question, db):
    """
    Evaluate the quality of an SQL query based on relevance and syntax.

    Args:
        sql_query (str): The SQL query to evaluate
        question (str): The user's question
        db (SQLDatabase): Database connection

    Returns:
        float: Quality score (0-1)
    """
    score = 0.0
    try:
        # Check syntax by attempting a dry run
        try:
            db.run(f"EXPLAIN {sql_query}")
            score += 0.4  # Valid syntax
        except:
            return score  # Invalid syntax

        # Relevance to question (simple heuristic)
        question_words = set(re.findall(r'\w+', question.lower()))
        sql_words = set(re.findall(r'\w+', sql_query.lower()))
        common_words = question_words.intersection(sql_words)
        score += 0.3 * (len(common_words) / max(len(question_words), 1))

        # Check if it uses relevant tables
        tables = extract_tables_from_sql(sql_query)
        all_tables = db.get_usable_table_names()
        if all(t in all_tables for t in tables):
            score += 0.3
        else:
            score -= 0.2

        return min(1.0, max(0.0, score))
    except Exception as e:
        logger.error(f"Error evaluating SQL quality: {str(e)}")
        return 0.0

########################################################################################
##      7. extract_tables_from_sql: Helper to extract tables from SQL queries         ##
########################################################################################

def extract_tables_from_sql(sql_query: str) -> list:
    """Extract table names from an SQL query."""
    table_pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s|;|$|\n)'
    tables = re.findall(table_pattern, sql_query, re.IGNORECASE)
    tables = list(set(t.strip() for t in tables if t.strip()))
    logger.info(f"Extracted tables from SQL: {tables}")
    return tables

########################################################################################
##      8. clean_sql_query: Helper to clean SQL queries                               ##
########################################################################################

def clean_sql_query(sql_query: str) -> str:
    """Remove markdown tags or unnecessary formatting from SQL query."""
    sql_query = sql_query.strip()
    if sql_query.startswith("```sql") and sql_query.endswith("```"):
        sql_query = sql_query[6:-3].strip()
    elif sql_query.startswith("```") and sql_query.endswith("```"):
        sql_query = sql_query[3:-3].strip()
    return sql_query


########################################################################################
##                              9. validate_sql_with_llm                               ##
########################################################################################

@task
def validate_sql_with_llm(question: str, sql_query: str, db: SQLDatabase, llm) -> bool:
    prompt = ChatPromptTemplate.from_template("""
    You are a SQL validation expert. Check if this SQL query correctly answers the question and matches the database schema.

    Question: "{question}"
    SQL Query: "{sql_query}"
    Database Schema: {schema_info}

    Check for:
    1. Correct tables/columns
    2. Proper filters (e.g., year)
    3. Appropriate aggregation

    Respond with "VALID" or "INVALID" followed by a reason.
    """)
    schema_info = db.get_table_info(extract_tables_from_sql(sql_query))
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question, "sql_query": sql_query, "schema_info": schema_info})
    is_valid = response.strip().startswith("VALID")
    if is_valid:
        logger.info("SQL validation passed")
    else:
        logger.warning(f"SQL validation failed: {response}")
    return is_valid

########################################################################################
##      10. search_examples_for_sql: Searches for exact match examples in file.         ##
########################################################################################

def search_examples_for_sql(question: str, sql_examples) -> tuple[str, list]:
    """Search sql_examples for an exact match and return its SQL and tables."""
    for example in sql_examples:
        if example["page_content"].strip().lower() == question.strip().lower():
            sql_query = example["sql_query"]
            logger.info(f"Exact match found in examples: '{sql_query}'")
            tables = extract_tables_from_sql(sql_query)
            return sql_query, tables
    logger.warning(f"No exact match found in examples for '{question}'")
    return None, []



########################################################################################
##      11. execute_sql_with_no_data_handling: Executes SQL with no-data fallback     ##
########################################################################################

def execute_sql_with_no_data_handling(question: str, sql_query: str, db: SQLDatabase, llm) -> str:
    """
    Execute an SQL query and handle cases where no data is found or execution fails.
    Returns a formatted response, either with results or a no-data message.

    Args:
        question (str): The user's question
        sql_query (str): The SQL query to execute
        db (SQLDatabase): Database connection
        llm: Language model for response formatting

    Returns:
        str: Formatted response (either query results or a no-data message)
    """
    # Clean the SQL query to avoid syntax errors from formatting
    sql_query = clean_sql_query(sql_query)
    logger.info(f"Executing SQL with no-data handling: {sql_query}")

    # Step 1: Execute the query safely
    try:
        result = db.run(sql_query)
        logger.info(f"SQL result: {result}")
    except Exception as e:
        logger.error(f"SQL execution failed: {str(e)}")
        result = None  # Treat as no data if execution fails

    # Step 2: Check if the result is empty or invalid
    if not result or result.strip() == "[]" or result.strip() == "" or result is None:
        # No data found case
        no_data_prompt = ChatPromptTemplate.from_template("""
        You are a helpful SQL agent. The query returned no data or failed to execute.
        Politely inform the user that no data was found for their question.

        Question: "{question}"
        SQL Query: "{sql_query}"
        SQL Result: "{result}"
        Response:
        """)
        no_data_chain = no_data_prompt | llm | StrOutputParser()
        response = no_data_chain.invoke({
            "question": question,
            "sql_query": sql_query,
            "result": result if result is not None else "Execution failed"
        })
        # Fallback response if LLM fails
        if not response:
            response = f"No data found for '{question}' in the database."
    else:
        # Data found, format the response
        response_prompt = ChatPromptTemplate.from_template("""
        Given an input question and its executed SQL result, return a user-friendly answer.
        Use column names from the result where applicable.

        Question: "{question}"
        SQL Query: "{sql_query}"
        SQL Result: "{result}"
        Answer: "Final answer incorporating column names if available"
        """)
        response_chain = response_prompt | llm | StrOutputParser()
        response = response_chain.invoke({
            "question": question,
            "sql_query": sql_query,
            "result": result
        })

    return response




@task
def collect_user_feedback(question: str, response: str, feedback_store: dict) -> dict:
    """
    Collect user feedback on the agent's response and store it for analysis.

    Args:
        question (str): The user's original question
        response (str): The agent's response
        feedback_store (dict): In-memory store for feedback (thread-specific)

    Returns:
        dict: Feedback result with 'is_helpful' (bool) and 'comment' (str or None)
    """
    try:
        # In an interactive context, this would prompt the user.
        # For now, we'll simulate it in run_interactive and log it here.
        logger.info(f"Collecting feedback for question: '{question}' | Response: '{response[:50]}...'")
        # Placeholder for feedback input (handled in run_interactive)
        feedback = {"is_helpful": None, "comment": None}
        feedback_store.setdefault("responses", []).append({
            "question": question,
            "response": response,
            "is_helpful": feedback["is_helpful"],
            "comment": feedback["comment"]
        })
        return feedback
    except Exception as e:
        logger.error(f"Error collecting feedback: {str(e)}")
        return {"is_helpful": None, "comment": f"Error: {str(e)}"}