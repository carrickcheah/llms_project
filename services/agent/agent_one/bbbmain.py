# # merged_main.py - SQL Agent with Gradio Interface

# import os
# import gradio as gr
# from loguru import logger
# from langgraph.func import entrypoint, task
# from langgraph.checkpoint.memory import MemorySaver

# # Custom utility imports
# from model import initialize_openai, initialize_embeddings, initialize_open_deep
# from maria import initialize_database
# from vectordb import qdrant_on_prem
# from task_03 import load_sql_examples, is_database_question
# from task_01 import (
#     find_sql_examples,
#     extract_relevant_tables,
#     validate_sql_with_llm,
#     extract_tables_from_sql,
#     generate_dynamic_sql,
#     check_tables_exist,
#     clean_sql_query
# )
# from task_02 import (
#     collect_user_feedback,
#     generate_response,
#     generate_reasoned_response,
# )

# # Global initialization 
# os.environ["TOKENIZERS_PARALLELISM"] = "false"
# llm = initialize_open_deep(os.getenv("DEEPSEEK_API_KEY"))
# embeddings = initialize_embeddings(os.getenv("HUGGINGFACE_MODEL"))
# db = initialize_database(os.getenv("DATABASE_URI"))
# qdrant_store = qdrant_on_prem(embeddings, os.getenv("COLLECTION_NAME"))
# sql_examples = load_sql_examples(os.getenv("EXAMPLES_FILE_PATH"))

# # Global feedback store
# feedback_store = {}

# @entrypoint(checkpointer=MemorySaver())
# def sql_agent_workflow(inputs: dict, config: dict = None) -> dict:
#     """
#     Workflow to process a user's question, generate and execute an SQL query if applicable,
#     and return a formatted response with feedback collection for database questions only.
#     """
#     question = inputs.get("question", "").strip()
#     if not question:
#         return {"response": "Please provide a question.", "feedback": None}

#     thread_id = config.get("configurable", {}).get("thread_id", "default_thread")
#     feedback_store = globals().setdefault("feedback_store", {}).setdefault(thread_id, {})
    
#     # Default response in case of errors
#     response = "I couldn't process your request. Please try again."
#     feedback = None

#     try:
#         if is_database_question(question):
#             logger.info(f"Database question detected: {question}")

#             # Step 1: Search for examples (vector + exact match)
#             examples_future = find_sql_examples(question, qdrant_store, embeddings, sql_examples, method="both")
#             examples = examples_future.result()
#             sql_query = None
#             tables = []
#             exact_match_found = False
            
#             if examples:
#                 for example in examples:
#                     if example["question"].strip().lower() == question.strip().lower():
#                         sql_query = example["sql"]
#                         tables = example["tables"]
#                         logger.info(f"Exact match found: '{sql_query}'")
#                         exact_match_found = True
#                         break

#             # Step 2: Execute directly if exact match found (skip validation), otherwise generate new SQL
#             if sql_query and exact_match_found:
#                 missing_tables = check_tables_exist(tables, db)
#                 if missing_tables:
#                     logger.error(f"Table(s) {missing_tables} not found in database.")
#                     response = f"Sorry, I can't execute the query because the following table(s) were not found: {', '.join(missing_tables)}."
#                 else:
#                     # Skip validation for exact matches and execute directly
#                     logger.info("Exact match found; executing without validation")
#                     sql_result = None
#                     try:
#                         sql_result = db.run(sql_query)
#                         logger.info(f"SQL result: {sql_result}")
#                     except Exception as e:
#                         logger.error(f"SQL execution failed: {str(e)}")
                    
#                     response_future = generate_reasoned_response(
#                         question=question,
#                         llm=llm, 
#                         sql_query=sql_query, 
#                         result=sql_result, 
#                         tables=tables
#                     )
#                     response = response_future.result()
            
#             # Step 3: Generate SQL if no exact match
#             else:
#                 logger.info("No exact match found; generating SQL.")
#                 tables = extract_relevant_tables(question, db)
#                 missing_tables = check_tables_exist(tables, db)
#                 if missing_tables:
#                     logger.error(f"Table(s) {missing_tables} not found in database.")
#                     response = f"Sorry, I can't process your request because the following table(s) were not found: {', '.join(missing_tables)}."
#                 else:
#                     sql_query_future = generate_dynamic_sql(question, tables, examples or [], db, llm)
#                     sql_query = sql_query_future.result()
#                     if sql_query.startswith("Error:"):
#                         logger.error(f"SQL generation failed: {sql_query}")
#                         response_future = generate_reasoned_response(
#                             question=question,
#                             llm=llm,
#                             sql_query="Generation failed",
#                             result=None,
#                             tables=tables
#                         )
#                         response = response_future.result()
#                     else:
#                         tables = extract_tables_from_sql(sql_query)
#                         missing_tables = check_tables_exist(tables, db)
#                         if missing_tables:
#                             logger.error(f"Table(s) {missing_tables} not found in database.")
#                             response = f"Sorry, I can't execute the query because the following table(s) were not found: {', '.join(missing_tables)}."
#                         else:
#                             # Only validate dynamically generated SQL
#                             is_valid_future = validate_sql_with_llm(question, sql_query, db, llm)
#                             is_valid = is_valid_future.result()
#                             if not is_valid:
#                                 # Use contextual validation failure response
#                                 products_list = [ex["question"] for ex in examples[:5]] if examples else []
#                                 error_context = {
#                                     "question": question,
#                                     "available_examples": products_list,
#                                     "database_tables": tables
#                                 }
#                                 response_future = generate_response(
#                                     question=question,
#                                     llm=llm,
#                                     response_type="validation_failure",
#                                     context=error_context
#                                 )
#                                 response = response_future.result()
#                             else:
#                                 # Execute SQL and get response
#                                 sql_query = clean_sql_query(sql_query)
#                                 logger.info(f"Executing SQL: {sql_query}")
#                                 sql_result = None
#                                 try:
#                                     sql_result = db.run(sql_query)
#                                     logger.info(f"SQL result: {sql_result}")
#                                 except Exception as e:
#                                     logger.error(f"SQL execution failed: {str(e)}")
                                
#                                 response_future = generate_reasoned_response(
#                                     question=question,
#                                     llm=llm,
#                                     sql_query=sql_query,
#                                     result=sql_result,
#                                     tables=tables
#                                 )
#                                 response = response_future.result()

#             if not sql_query:
#                 response = "Sorry, I could not generate a suitable SQL query for your question. Please try rephrasing it."

#             # Step 4: Collect feedback only for database questions
#             try:
#                 feedback_future = collect_user_feedback(question, response, feedback_store)
#                 feedback = feedback_future.result()
#             except Exception as e:
#                 logger.error(f"Error collecting feedback: {str(e)}")
#                 feedback = {"is_helpful": None, "comment": f"Error collecting feedback: {str(e)}"}

#         else:
#             logger.info(f"Non-database question detected: {question}")
#             response_future = generate_response(question, llm, response_type="polite")
#             response = response_future.result()
#             feedback = None  # No feedback for non-database questions
    
#     except Exception as e:
#         logger.error(f"Error in sql_agent_workflow: {str(e)}")
#         response = f"I encountered an error while processing your question. Please try again or rephrase your question."
#         feedback = None

#     # Create a completely new dict to ensure no Future objects are returned
#     return {
#         "response": response, 
#         "feedback": feedback
#     }

# def sql_agent_chat(message, history):
#     """
#     Wrapper function for Gradio ChatInterface that formats the response with
#     simplified formatting that works across all Gradio versions.
#     """
#     config = {"configurable": {"thread_id": "sql_agent_thread"}}
#     user_input = message["content"] if isinstance(message, dict) else message
    
#     try:
#         result = sql_agent_workflow.invoke({"question": user_input}, config=config)
#         response = result["response"]
        
#         # Format the response for better display with simple formatting
#         if "**Reasoning Process:**" in response and "**Answer:**" in response:
#             # Split the response into reasoning and answer parts
#             parts = response.split("**Answer:**")
            
#             if len(parts) == 2:
#                 reasoning = parts[0].strip()
#                 answer = parts[1].strip()
                
#                 # Extract SQL query if present
#                 sql_query = ""
#                 if "SQL Query:" in reasoning:
#                     sql_lines = [line for line in reasoning.split('\n') if "SQL Query:" in line]
#                     if sql_lines:
#                         sql_query_line = sql_lines[0]
#                         sql_query = sql_query_line.replace("SQL Query:", "").strip()
                
#                 # Format with simple text separators
#                 formatted_response = (
#                     f"▼ REASONING PROCESS (click to expand) ▼\n\n"
#                     f"{reasoning}\n\n"
#                     f"▲ END OF REASONING PROCESS ▲\n\n"
#                 )
                
#                 if sql_query:
#                     formatted_response += f"SQL QUERY:\n{sql_query}\n\n"
                
#                 formatted_response += f"ANSWER:\n{answer}"
#                 return formatted_response
        
#         # Handle cases with "Reasoning Process" in other formats
#         if "### Reasoning Process" in response and "### Answer" in response:
#             parts = response.split("### Answer")
            
#             if len(parts) == 2:
#                 reasoning = parts[0].replace("### Reasoning Process", "").strip()
#                 answer = parts[1].strip()
                
#                 # Same formatting as above
#                 formatted_response = (
#                     f"▼ REASONING PROCESS (click to expand) ▼\n\n"
#                     f"{reasoning}\n\n"
#                     f"▲ END OF REASONING PROCESS ▲\n\n"
#                 )
                
#                 if "SQL Query:" in reasoning:
#                     sql_lines = [line for line in reasoning.split('\n') if "SQL Query:" in line]
#                     if sql_lines:
#                         sql_query_line = sql_lines[0]
#                         sql_query = sql_query_line.replace("SQL Query:", "").strip()
#                         formatted_response += f"SQL QUERY:\n{sql_query}\n\n"
                
#                 formatted_response += f"ANSWER:\n{answer}"
#                 return formatted_response
                
#         # If we can't extract reasoning and answer, return the original response
#         return response
        
#     except Exception as e:
#         logger.error(f"Error in sql_agent_chat: {str(e)}")
#         return f"I encountered an error processing your request. Please try again or rephrase your question."

# def main():
#     """Initialize and launch the Gradio interface."""
#     logger.info("Starting Gradio chat interface")

#     # Define simple CSS that should work in most Gradio versions
#     css = """
#     .chatbot {border-radius: 5px; box-shadow: 0 4px 8px rgba(0,0,0,0.1);}
#     .message {font-family: 'Arial', sans-serif;}
#     #component-0 {max-width: 2500px; margin: 0 auto;}
#     """
    
#     # Create Gradio interface with minimal options for compatibility
#     demo = gr.ChatInterface(
#         fn=sql_agent_chat,
#         title="Nex SQL Agent",
#         description="Your friendly SQL data analyst. Ask me anything about the database!",
#         examples=[
#             "Show me the top 5 customers by sales",
#             "Who is our top customer for Cycling Gloves in 2023?",
#             "How many orders were placed this year?"
#         ],
#         theme="soft",
#         chatbot=gr.Chatbot(
#             height=750,
#             placeholder="Ask me a database question..."
#         ),
#         textbox=gr.Textbox(placeholder="Type your question here...", container=False, scale=7),
#         css=css
#     )
    
#     logger.info("Launching Gradio interface")
#     demo.launch(share=False)
#     logger.info("Gradio interface closed")

# if __name__ == "__main__":
#     # Configure logging
#     logger.add("logs/sql_agent_{time}.log", rotation="500 MB", level="INFO")
#     logger.info("Starting SQL Agent with Gradio interface")
#     main()