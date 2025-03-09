###################################################
# Never edit this comments, notify me if you keen to edit.
# task_01.py:
# Purpose: Handles the core SQL-related tasks (generation, validation, execution).
# Functions: Includes all functions needed to produce a valid SQL query and execute it, plus helpers.
# Dependencies: Self-contained except for external imports (db, llm, etc.), which are provided by main.py.

# Content of tables
# 1. extract_relevant_tables
# 2. find_sql_examples
# 3. generate_dynamic_sql
# 4. deleted...
# 5. extract_tables_from_sql
# 6. clean_sql_query
# 7. validate_sql_with_llm
# 8. deleted...
# 9. check_tables_exist
###################################################


# Core SQL Processing Functions
from loguru import logger
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities.sql_database import SQLDatabase
import re
from langgraph.func import task


#########################################################################################
##      1. extract_relevant_tables: Identifies tables related to user question         ##
#########################################################################################
# 1. extract_relevant_tables: Identifies tables related to user question
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
    relevance_scores = {}

    # Strategy 1: Direct table name matching
    for table in all_tables:
        table_lower = table.lower()
        if table_lower in question_lower:
            relevance_scores[table] = relevance_scores.get(table, 0) + 15

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

#########################################################################################
##      2. find_sql_examples: Uses embeddings to find similar examples            ##
#########################################################################################

@task
def find_sql_examples(question, vector_store, embeddings, sql_examples, method="both"):
    """
    Find SQL examples for a question using vector search, exact match, or both.

    Args:
        question (str): The user's question
        vector_store: Vector database store (for semantic search)
        embeddings: Embedding model (for semantic search)
        sql_examples: List of SQL examples from file (for exact match)
        method (str): Search method - "vector", "exact", or "both" (default)

    Returns:
        list: List of example dicts (vector) or single dict (exact) with keys "question", "sql", "tables", "score" (optional)
              Returns None if no examples found
    """
    logger.info(f"Finding SQL examples for: '{question}' with method: {method}")
    examples = []

    # Vector search (semantic similarity)
    if method in ["vector", "both"]:
        try:
            retriever = vector_store.as_retriever(search_kwargs={"k": 5})
            retrieved_docs = retriever.invoke(question)
            if retrieved_docs:
                logger.success(f"Retrieved {len(retrieved_docs)} similar examples via vector search")
                for i, doc in enumerate(retrieved_docs):
                    example_question = doc.page_content
                    example_sql = doc.metadata.get("sql_query", "") if isinstance(doc.metadata, dict) else str(doc.metadata)
                    score = doc.metadata.get('score', 0) if isinstance(doc.metadata, dict) else 0
                    tables = extract_tables_from_sql(example_sql)
                    example = {
                        "question": example_question,
                        "sql": example_sql,
                        "tables": tables,
                        "score": score,
                        "rank": i + 1
                    }
                    examples.append(example)
                    logger.info(f"Vector Example #{i+1}: '{example_question[:80]}...' (score: {score:.4f})")
            else:
                logger.warning("No similar examples found via vector search")
        except Exception as e:
            logger.error(f"Error in vector search: {str(e)}")

    # Exact match search
    if method in ["exact", "both"] and (method == "exact" or not examples):
        for example in sql_examples:
            if example["page_content"].strip().lower() == question.strip().lower():
                sql_query = example["sql_query"]
                tables = extract_tables_from_sql(sql_query)
                examples = [{
                    "question": example["page_content"],
                    "sql": sql_query,
                    "tables": tables,
                    "score": 1.0  # Exact match gets max score
                }]
                logger.info(f"Exact match found: '{sql_query}'")
                break
        else:
            logger.warning(f"No exact match found in examples for '{question}'")

    return examples if examples else None


#########################################################################################
##      3. generate_dynamic_sql: Creates SQL based on tables and examples             ##
#########################################################################################

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

#########################################################################################
##      4. deleted                ##
#########################################################################################



#########################################################################################
##      5. extract_tables_from_sql: Helper to extract tables from SQL queries         ##
#########################################################################################

def extract_tables_from_sql(sql_query: str) -> list:
    """Extract table names from an SQL query."""
    table_pattern = r'(?:FROM|JOIN)\s+([a-zA-Z_][a-zA-Z0-9_]*)(?:\s|;|$|\n)'
    tables = re.findall(table_pattern, sql_query, re.IGNORECASE)
    tables = list(set(t.strip() for t in tables if t.strip()))
    logger.info(f"Extracted tables from SQL: {tables}")
    return tables

#########################################################################################
##      6. clean_sql_query: Helper to clean SQL queries                               ##
#########################################################################################

def clean_sql_query(sql_query: str) -> str:
    """Remove markdown tags or unnecessary formatting from SQL query."""
    sql_query = sql_query.strip()
    if sql_query.startswith("```sql") and sql_query.endswith("```"):
        sql_query = sql_query[6:-3].strip()
    elif sql_query.startswith("```") and sql_query.endswith("```"):
        sql_query = sql_query[3:-3].strip()
    return sql_query


#########################################################################################
##      7. validate_sql_with_llm: Validates SQL query using LLM                       ##
#########################################################################################

@task
def validate_sql_with_llm(question: str, sql_query: str, db: SQLDatabase, llm) -> bool:
    """
    Validate an SQL query using an LLM to ensure it matches the question and schema.

    Args:
        question (str): The user's question
        sql_query (str): The SQL query to validate
        db (SQLDatabase): Database connection
        llm: Language model

    Returns:
        bool: True if valid, False otherwise
    """
    prompt = ChatPromptTemplate.from_template("""
    You are a MariaDB SQL validation expert. Verify if this SQL query correctly answers the question, matches the database schema, and ensures compatibility with MariaDB

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

#########################################################################################
##      8. search_examples_for_sql: deleted        ##
#########################################################################################

# def search_examples_for_sql(question: str, sql_examples) -> tuple[str, list]:
#     """Search sql_examples for an exact match and return its SQL and tables."""
#     for example in sql_examples:
#         if example["page_content"].strip().lower() == question.strip().lower():
#             sql_query = example["sql_query"]
#             logger.info(f"Exact match found in examples: '{sql_query}'")
#             tables = extract_tables_from_sql(sql_query)
#             return sql_query, tables
#     logger.warning(f"No exact match found in examples for '{question}'")
#     return None, []


#########################################################################################
##      9. check_tables_exist: Check if all required tables exist in the database     ##
#########################################################################################
def check_tables_exist(tables, database):
    """
    Check if all required tables exist in the database.

    Args:
        tables (list): List of table names to check
        database (SQLDatabase): Database connection

    Returns:
        list: List of missing table names
    """
    existing_tables = set(database.get_usable_table_names())
    missing_tables = [t for t in tables if t not in existing_tables]
    return missing_tables