import inspect
import importlib.util
import os
import re

# Load your tools.py file as a module
def load_tools(file_path):
    # Get just the filename for better printing
    filename = os.path.basename(file_path)
    
    # Load the module
    spec = importlib.util.spec_from_file_location("tools_module", file_path)
    tools_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tools_module)
    return tools_module, filename

# Try to detect tools using various methods
def extract_tools(module):
    tools_info = []
    
    for name, obj in inspect.getmembers(module):
        # Try different ways to detect if it's a tool
        is_tool = False
        if inspect.isfunction(obj):
            # Method 1: Check for _tool attribute
            if hasattr(obj, '_tool'):
                is_tool = True
            # Method 2: Check for tool-related attributes
            elif hasattr(obj, 'tool_name') or hasattr(obj, 'return_direct'):
                is_tool = True
            # Method 3: Check for schema_cls attribute (used by some LangChain tools)
            elif hasattr(obj, 'schema_cls'):
                is_tool = True
            # Method 4: Check function source for @tool decorator
            else:
                try:
                    source = inspect.getsource(obj)
                    if "@tool" in source:
                        is_tool = True
                except Exception:
                    pass
            
            # Let's also look at function decorators explicitly
            if not is_tool and hasattr(obj, '__wrapped__'):
                # The function has been decorated
                is_tool = True
        
        if is_tool:
            tool_info = {
                "name": name,
                "description": inspect.getdoc(obj) or "",
                "code": inspect.getsource(obj)
            }
            tools_info.append(tool_info)
            print(f"Found tool: {name}")
    
    return tools_info

# Extract tools directly from source code using regex
def extract_tools_from_source(file_path):
    # Read the file directly
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Split by function definitions after @tool decorator
    tools_info = []
    
    # Try to find code blocks that start with @tool
    tool_pattern = r'@tool\s+def\s+(\w+).*?"""(.*?)""".*?(?=@tool|\Z)'
    matches = re.findall(tool_pattern, content, re.DOTALL)
    
    if matches:
        for name, docstring in matches:
            print(f"Found tool via regex: {name}")
            # Find the whole function source
            func_pattern = rf'@tool\s+def\s+{name}.*?(?=@tool|\Z)'
            func_match = re.search(func_pattern, content, re.DOTALL)
            if func_match:
                code = func_match.group(0)
                tool_info = {
                    "name": name,
                    "description": docstring.strip(),
                    "code": code
                }
                tools_info.append(tool_info)
    
    return tools_info

def prepare_for_qdrant(tools_info):
    # Create texts and metadata for Qdrant
    texts = []
    metadatas = []
    
    for tool in tools_info:
        # Create a text representation of the tool
        text = f"Tool Name: {tool['name']}\nDescription: {tool['description']}\nCode: {tool['code']}"
        texts.append(text)
        
        # Create metadata
        metadata = {
            "name": tool['name'],
            "type": "tool"
        }
        metadatas.append(metadata)
    
    return texts, metadatas

from initialize_group import init_qdrant_from_texts
from langchain_openai import OpenAIEmbeddings

def store_tools_in_qdrant(texts, metadatas):
    # Initialize embedding model
    embedding = OpenAIEmbeddings()
    
    # Store in Qdrant
    qdrant = init_qdrant_from_texts(
        texts=texts,
        metadatas=metadatas,
        embedding=embedding,
        collection_name="tools_collection",
        url="http://localhost:6333",  # Update with your Qdrant URL
        overwrite=True
    )
    
    return qdrant

def ingest_tools_to_qdrant(tools_path):
    # Try first method - module inspection
    tools_module, filename = load_tools(tools_path)
    tools_info = extract_tools(tools_module)
    
    # If first method found nothing, try the regex approach
    if len(tools_info) == 0:
        print("No tools found using module inspection, trying source code parsing...")
        tools_info = extract_tools_from_source(tools_path)
    
    print(f"Found {len(tools_info)} tools in {filename}")
    
    if len(tools_info) == 0:
        print("WARNING: No tools were found with either method. Check if the tools are properly decorated.")
        return None
    
    # Step 2: Prepare data for Qdrant
    texts, metadatas = prepare_for_qdrant(tools_info)
    
    # Step 3: Store in Qdrant
    qdrant = store_tools_in_qdrant(texts, metadatas)
    print("Successfully stored tools in Qdrant!")
    
    return qdrant

# Add search functionality
from initialize_group import dense_vector_search
from langchain_openai import OpenAIEmbeddings

def find_tool(query):
    # Initialize embedding model
    embedding = OpenAIEmbeddings()
    
    # Search Qdrant
    results = dense_vector_search(
        query=query,
        embedding=embedding,
        collection_name="tools_collection",
        url="http://localhost:6333",  # Update with your Qdrant URL
        k=3  # Return top 3 matching tools
    )
    
    return results

# Run the ingestion
if __name__ == "__main__":
    # Ingest tools
    ingest_tools_to_qdrant("/Users/carrickcheah/llms_project/services/agent/tools.py")
    
    # Uncomment to test search
    # print("\nTesting search functionality...")
    # results = find_tool("generate monthly invoice")
    # for doc in results:
    #     print(f"Tool: {doc.metadata['name']}")
    #     print(f"Content: {doc.page_content[:200]}...")
    #     print("---")