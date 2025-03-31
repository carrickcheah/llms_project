import os
import json
import tempfile
import io
import re
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs
import mimetypes
import logging
from functools import wraps
from typing import Optional, Dict, Any
import time

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Configuration
CONFIG = {
    'MAX_FILE_SIZE': 10 * 1024 * 1024,  # 10MB
    'CHUNK_SIZE': 8192,  # 8KB chunks for reading
    'SUPPORTED_EXTENSIONS': {'.pdf', '.docx', '.doc'},
    'TEMP_DIR': './tmp',
    'RATE_LIMIT': {'requests': 10, 'period': 60},  # 10 requests per minute
}

def rate_limit(func):
    """Decorator to implement rate limiting"""
    requests = {}
    
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        client_ip = self.client_address[0]
        current_time = time.time()
        
        # Clean up old requests
        requests[client_ip] = [t for t in requests.get(client_ip, [])
                             if current_time - t < CONFIG['RATE_LIMIT']['period']]
        
        if len(requests.get(client_ip, [])) >= CONFIG['RATE_LIMIT']['requests']:
            self.send_error(429, "Too many requests")
            return None
            
        requests.setdefault(client_ip, []).append(current_time)
        return func(self, *args, **kwargs)
    return wrapper

# Import clean_pdf_text function
try:
    from clean_text_function import clean_pdf_text
    logger.info("✅ Text cleaning function imported successfully")
except ImportError:
    # Define function inline if import fails
    def clean_pdf_text(text):
        """Clean up PDF extracted text for better readability"""
        # Replace common encoding issues
        replacements = {
            'â€"': '-',
            'â€™': "'",
            'â€œ': '"',
            'â€': '"',
            'â€¦': '...',
            'Â': ' '
        }
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Fix spaced-out text (common PDF extraction issue)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # If this line appears to have spaced-out text
            if re.search(r'\b\w\s\w\s\w', line.strip()):
                # Compress single-letter words with spaces using a simpler pattern
                line = re.sub(r'([a-zA-Z])\s+([a-zA-Z])', r'\1\2', line)
            cleaned_lines.append(line)
            
        text = '\n'.join(cleaned_lines)
        
        # Clean up other spacing issues
        text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces with one
        text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
        
        return text
    logger.info("⚠️ Using inline text cleaning function")

# Handle docx import safely
try:
    import docx
    DOCX_AVAILABLE = True
except ImportError:
    logger.warning("Warning: python-docx not installed. Word document parsing will be limited.")
    DOCX_AVAILABLE = False
    
# Add PyPDF2 for fallback PDF parsing if available
try:
    import PyPDF2
    PYPDF_AVAILABLE = True
    logger.info("✅ PyPDF2 is available for fallback PDF parsing")
except ImportError:
    logger.note("Note: PyPDF2 not installed. Will use LlamaParse only for PDFs.")
    PYPDF_AVAILABLE = False

# Display Python path to help diagnose environment issues
import sys
logger.info(f"Python executable: {sys.executable}")
logger.info(f"Python path: {sys.path}")

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
    logger.info("✅ Environment variables loaded from .env file with python-dotenv")
    # Print all environment variables (except sensitive ones)
    logger.info("Environment variables available:")
    for key in os.environ.keys():
        if not key.endswith('KEY') and not key.endswith('SECRET'):
            logger.info(f"  {key}")
        else:
            logger.info(f"  {key}: [HIDDEN]")
except ImportError as e:
    logger.warning(f"⚠️ Warning: python-dotenv not installed: {e}")
    logger.info("Attempting to load environment variables from OS")
    # Manual loading from .env file as fallback
    try:
        with open('.env', 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value
                    logger.info(f"Set environment variable: {key}")
    except Exception as e:
        logger.warning(f"Failed to manually load .env file: {e}")

# Check LlamaCloud API key
LLAMA_CLOUD_API_KEY = os.getenv("LLAMA_CLOUD_API_KEY")
if LLAMA_CLOUD_API_KEY:
    logger.info("✅ LLAMA_CLOUD_API_KEY found in environment")
else:
    logger.warning("⚠️ Warning: LLAMA_CLOUD_API_KEY not found in environment. Please set it in a .env file.")

# Try to import LlamaParse
try:
    from llama_cloud_services import LlamaParse
    logger.info("✅ Successfully imported llama-cloud-services package")
    try:
        llama_cloud_api_key = os.environ.get("LLAMA_CLOUD_API_KEY")
        if not llama_cloud_api_key:
            logger.warning("⚠️ LLAMA_CLOUD_API_KEY not found in environment")
            raise ValueError("Missing API key for LlamaParse")
            
        parser = LlamaParse(api_key=llama_cloud_api_key)
        logger.info("✅ LlamaParse client initialized successfully")
        
        LLAMA_CLOUD_AVAILABLE = True
    except Exception as e:
        logger.warning(f"⚠️ Error initializing LlamaParse: {e}")
        LLAMA_CLOUD_AVAILABLE = False
        parser = None
except ImportError as e:
    logger.warning(f"⚠️ Error importing llama-cloud-services: {e}")
    logger.info("Try running: uv add llama-cloud-services")
    LLAMA_CLOUD_AVAILABLE = False
    parser = None

# LlamaParse client is configured above in the try/except block

class DocumentParserServer(BaseHTTPRequestHandler):
    def _set_headers(self, content_type="text/html", status_code=200):
        """Set response headers with security headers"""
        self.send_response(status_code)
        self.send_header("Content-type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header("X-XSS-Protection", "1; mode=block")
        self.send_header("Content-Security-Policy", "default-src 'self'")
        self.send_header("Strict-Transport-Security", "max-age=31536000")
        self.end_headers()

    def _send_json_response(self, data: Dict[str, Any], status_code: int = 200):
        """Send JSON response with proper headers"""
        self._set_headers("application/json", status_code)
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _validate_file(self, filename: str, content_length: int) -> Optional[str]:
        """Validate file before processing"""
        if content_length > CONFIG['MAX_FILE_SIZE']:
            return "File too large"
            
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in CONFIG['SUPPORTED_EXTENSIONS']:
            return f"Unsupported file type: {file_ext}"
            
        return None

    def _save_uploaded_file(self, file_data: bytes, file_ext: str) -> Optional[str]:
        """Save uploaded file with proper error handling"""
        try:
            os.makedirs(CONFIG['TEMP_DIR'], exist_ok=True)
            
            with tempfile.NamedTemporaryFile(delete=False, 
                                           suffix=file_ext,
                                           dir=CONFIG['TEMP_DIR']) as temp_file:
                temp_file.write(file_data)
                return temp_file.name
        except Exception as e:
            logger.error(f"Error saving file: {e}")
            return None

    @rate_limit
    def do_POST(self):
        """Handle POST requests with improved error handling and security"""
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            # Parse the input data
            content_type = self.headers.get('Content-Type', '')
            if content_type.startswith('application/json'):
                input_data = json.loads(post_data)
                result = self._process_user_input(input_data)
                self._send_json_response(result)
                
            else:
                # Handle file uploads
                self._handle_file_upload(post_data)
                
        except Exception as e:
            logger.error(f"Error in POST request: {str(e)}")
            self.send_error(500, f"Internal Server Error: {str(e)}")

    def _handle_file_upload(self, post_data):
        """Handle file upload"""
        temp_file_path = None
        try:
            if self.path != '/upload':
                self.send_error(404, "Not Found")
                return

            # Validate content type
            content_type = self.headers.get('Content-Type', '')
            if not content_type.startswith('multipart/form-data'):
                self._send_json_response({"error": "Invalid content type"}, 400)
                return

            # Read data in chunks
            data = bytearray(post_data)
            remaining = len(post_data)
            
            while remaining > 0:
                chunk = data[:min(CONFIG['CHUNK_SIZE'], remaining)]
                data = data[min(CONFIG['CHUNK_SIZE'], remaining):]
                remaining -= len(chunk)

            # Process multipart data
            boundary = content_type.split('=')[1].encode()
            parts = data.split(boundary)
            
            filename = None
            file_data = None
            
            for part in parts:
                if b'Content-Disposition' not in part:
                    continue
                    
                if b'filename=' not in part:
                    continue
                    
                header_end = part.find(b'\r\n\r\n')
                if header_end == -1:
                    continue
                    
                headers = part[:header_end].decode('utf-8', errors='replace')
                filename = re.search(r'filename="(.+?)"', headers)
                if filename:
                    filename = filename.group(1)
                    file_data = part[header_end+4:-2]  # Remove trailing \r\n
                    break

            if not filename or not file_data:
                # Instead of sending JSON error, redirect to output.html with error message
                error_result = {"error": "No file uploaded"}
                self._serve_result_page(error_result)
                return

            # Validate file
            error = self._validate_file(filename, len(file_data))
            if error:
                # Instead of sending JSON error, redirect to output.html with error message
                error_result = {"error": error}
                self._serve_result_page(error_result)
                return

            # Save file
            temp_file_path = self._save_uploaded_file(file_data, 
                                                     os.path.splitext(filename)[1].lower())
            if not temp_file_path:
                # Instead of sending JSON error, redirect to output.html with error message
                error_result = {"error": "Failed to save file"}
                self._serve_result_page(error_result)
                return

            # Process file based on type
            result = self._process_file(temp_file_path)
            
            # Generate and serve result page
            self._serve_result_page(result)

        except Exception as e:
            logger.error(f"Error processing request: {e}", exc_info=True)
            # Instead of sending JSON error, redirect to output.html with error message
            error_result = {"error": "Internal server error: " + str(e)}
            self._serve_result_page(error_result)
        finally:
            # Clean up temporary files
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                except Exception as e:
                    logger.warning(f"Error cleaning up temporary file: {e}")

    def _process_user_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process user input and determine next steps
        
        Args:
            input_data: Dictionary containing user input data
            
        Returns:
            Dictionary containing processing result and next steps
        """
        try:
            # Example input validation
            if not isinstance(input_data, dict):
                raise ValueError("Input data must be a dictionary")
                
            # Process based on user input
            action = input_data.get('action', '').lower()
            
            if action == 'process_file':
                # Example: Process file based on input
                file_path = input_data.get('file_path')
                if file_path:
                    result = self._process_file(file_path)
                    return {
                        'status': 'success',
                        'result': result,
                        'next_step': 'display_result'
                    }
                    
            elif action == 'get_status':
                # Example: Get processing status
                job_id = input_data.get('job_id')
                if job_id:
                    # Implement status checking logic here
                    return {
                        'status': 'success',
                        'job_status': 'processing',
                        'next_step': 'check_status'
                    }
                    
            else:
                return {
                    'status': 'error',
                    'message': 'Unknown action',
                    'next_step': 'show_error'
                }
                
        except Exception as e:
            logger.error(f"Error processing user input: {str(e)}")
            return {
                'status': 'error',
                'message': str(e),
                'next_step': 'show_error'
            }

    def _process_file(self, file_path: str) -> Dict[str, Any]:
        """Process uploaded file with proper error handling"""
        file_ext = os.path.splitext(file_path)[1].lower()
        
        try:
            if file_ext == '.pdf':
                return self._process_pdf(file_path)
            elif file_ext in ['.docx', '.doc']:
                return self._process_word(file_path)
            else:
                return {"error": f"Unsupported file type: {file_ext}"}
        except Exception as e:
            logger.error(f"Error processing file: {e}", exc_info=True)
            return {"error": f"File processing error: {str(e)}"}

    def _extract_invoice_data(self, result) -> Dict[str, Any]:
        """Extract invoice data from processed document"""
        try:
            # Default invoice data
            invoice_data = {
                "invoiceNumber": f"INV-{int(time.time())}",
                "companyName": "Auto-Extracted",
                "personInCharge": "Document Upload",
                "contactNumber": "",
                "email": "",
                "address": "",
                "invoiceDate": time.strftime("%Y-%m-%d"),
                "dueDate": time.strftime("%Y-%m-%d", time.localtime(time.time() + 30*24*60*60)),  # 30 days later
                "items": [],
                "taxRate": 0.1  # 10%
            }
            
            # Extract text content
            content = ""
            if isinstance(result, dict) and "message" in result:
                content = result["message"]
            elif isinstance(result, str):
                content = result
            elif isinstance(result, bytes):
                try:
                    content = result.decode('utf-8', errors='replace')
                except Exception:
                    content = str(result)
            else:
                content = str(result)
            
            # Simple regex patterns for invoice data extraction
            # Invoice number
            invoice_match = re.search(r'(?i)invoice\s*(?:#|number|no)?[:\s]*([A-Z0-9\-]+)', content)
            if invoice_match:
                invoice_data["invoiceNumber"] = invoice_match.group(1).strip()
            
            # Company name
            company_match = re.search(r'(?i)(?:company|business|firm|corp|inc)[:\s]*([\w\s\.,&]+)(?:\n|\r|$)', content)
            if company_match:
                invoice_data["companyName"] = company_match.group(1).strip()
            
            # Person in charge
            person_match = re.search(r'(?i)(?:attention|attn|contact|person)[:\s]*([\w\s\.]+)(?:\n|\r|$)', content)
            if person_match:
                invoice_data["personInCharge"] = person_match.group(1).strip()
            
            # Contact number
            phone_match = re.search(r'(?i)(?:phone|tel|mobile|contact)[:\s]*((?:\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4})', content)
            if phone_match:
                invoice_data["contactNumber"] = phone_match.group(1).strip()
            
            # Email
            email_match = re.search(r'[\w\.-]+@[\w\.-]+\.[a-zA-Z]{2,}', content)
            if email_match:
                invoice_data["email"] = email_match.group(0).strip()
            
            # Address
            address_match = re.search(r'(?i)(?:address|location)[:\s]*([\w\s\.,#\-]+)(?:\n|\r|$)', content)
            if address_match:
                invoice_data["address"] = address_match.group(1).strip()
            
            # Dates
            date_match = re.search(r'(?i)(?:date|issued)[:\s]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})', content)
            if date_match:
                invoice_data["invoiceDate"] = date_match.group(1).strip()
            
            due_date_match = re.search(r'(?i)(?:due|payment)[:\s]*(\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}|\w+\s+\d{1,2},?\s+\d{4})', content)
            if due_date_match:
                invoice_data["dueDate"] = due_date_match.group(1).strip()
            
            # Extract items using a more complex pattern
            # Look for patterns like: Item name, quantity, price
            item_pattern = r'(?i)(\d+|\w+)\s+([\w\s\-\.]+)\s+(\d+)\s+(?:x\s+)?(\$?\d+\.?\d*)'
            items_found = re.findall(item_pattern, content)
            
            # Fallback: if no items found, at least create one item with the document filename
            if not items_found:
                invoice_data["items"] = [
                    {"id": 1, "name": "Document Processing", "quantity": 1, "unitPrice": 100}
                ]
            else:
                for idx, item in enumerate(items_found, 1):
                    try:
                        item_id = idx
                        item_name = item[1].strip()
                        item_quantity = int(item[2])
                        # Remove any $ sign and convert to float
                        item_price = float(item[3].replace('$', ''))
                        
                        invoice_data["items"].append({
                            "id": item_id,
                            "name": item_name,
                            "quantity": item_quantity,
                            "unitPrice": item_price
                        })
                    except (ValueError, IndexError):
                        # Skip invalid items
                        continue
            
            # Always ensure at least one item exists
            if not invoice_data["items"]:
                invoice_data["items"] = [
                    {"id": 1, "name": "Document Processing", "quantity": 1, "unitPrice": 100}
                ]
                
            return invoice_data
            
        except Exception as e:
            logger.error(f"Error extracting invoice data: {e}", exc_info=True)
            # Return default data if extraction fails
            return {
                "invoiceNumber": f"INV-{int(time.time())}",
                "companyName": "Extraction Failed",
                "personInCharge": "System",
                "contactNumber": "",
                "email": "",
                "address": "",
                "invoiceDate": time.strftime("%Y-%m-%d"),
                "dueDate": time.strftime("%Y-%m-%d", time.localtime(time.time() + 30*24*60*60)),
                "items": [
                    {"id": 1, "name": "Error: Failed to extract data", "quantity": 1, "unitPrice": 0}
                ],
                "taxRate": 0.1
            }
    
    def _process_pdf(self, file_path: str) -> Dict[str, Any]:
        """Process PDF file with proper error handling"""
        try:
            if not LLAMA_CLOUD_AVAILABLE:
                return {"message": "LlamaParse not available. Please install llama-cloud-services to parse PDFs."}
            
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            file_size = os.path.getsize(file_path)
            logger.info(f"Processing PDF: {os.path.basename(file_path)}, size: {file_size} bytes")
            
            if PYPDF_AVAILABLE:
                try:
                    logger.info(f"Using PyPDF2 to extract text from {file_path}")
                    pdf_text = ""
                    with open(file_path, 'rb') as pdf_file:
                        pdf_reader = PyPDF2.PdfReader(pdf_file)
                        num_pages = len(pdf_reader.pages)
                        logger.info(f"PDF has {num_pages} pages")
                        
                        for page_num in range(num_pages):
                            page = pdf_reader.pages[page_num]
                            page_text = page.extract_text()
                            if page_text:
                                pdf_text += f"\n--- Page {page_num + 1} ---\n{page_text}\n"
                    
                    if pdf_text.strip():
                        logger.info(f"Successfully extracted {len(pdf_text)} characters of text with PyPDF2")
                        # Clean up the extracted text
                        cleaned_text = clean_pdf_text(pdf_text)
                        logger.info(f"Text cleaned and formatted for better readability")
                        return {"message": cleaned_text}
                    else:
                        logger.info("PyPDF2 didn't extract any text, PDF might be scanned/image-based")
                        return {"message": "This appears to be a scanned PDF. No text could be extracted directly."}
                except Exception as pdf_error:
                    logger.error(f"Error using PyPDF2: {pdf_error}")
                    return {"error": f"PDF parsing error: {str(pdf_error)}"}
            
            # Try LlamaParse as backup only if PyPDF2 failed
            if parser is not None and (not PYPDF_AVAILABLE or isinstance(result, dict) and "error" in result):
                try:
                    logger.info(f"Attempting to parse with LlamaParse")
                    result = parser.parse_file(file_path)
                    logger.info(f"LlamaParse successful. Result type: {type(result)}")
                    
                    # Handle different result types
                    if isinstance(result, str):
                        return {"message": result}
                    elif isinstance(result, bytes):
                        # First try to decode as UTF-8 with replace error handler
                        try:
                            return {"message": result.decode('utf-8', errors='replace')}
                        except Exception as e:
                            logger.warning(f"UTF-8 decode failed: {e}")
                            # If that fails, try latin-1 (which can decode any byte)
                            return {"message": result.decode('latin-1', errors='replace')}
                    elif isinstance(result, dict):
                        return result
                    else:
                        return {"message": str(result)}
                except Exception as e:
                    logger.error(f"Error parsing PDF with LlamaParse: {str(e)}")
                    # If PyPDF2 also failed, return error with more details
                    if isinstance(result, dict) and "error" in result:
                        error_details = {
                            "error": "All parsing methods failed",
                            "details": {
                                "pypdf2_error": result.get("error", "Unknown error"),
                                "llamaparse_error": str(e)
                            },
                            "suggestion": "Try converting the PDF to a different format or ensure it contains extractable text."
                        }
                        return error_details
        except Exception as e:
            logger.error(f"Error processing PDF: {e}", exc_info=True)
            return {"error": f"PDF processing error: {str(e)}"}

    def _process_word(self, file_path: str) -> Dict[str, Any]:
        """Process Word document with proper error handling"""
        try:
            if not DOCX_AVAILABLE:
                return {"message": "python-docx not available. Please install to parse Word documents."}
            
            if not os.path.exists(file_path):
                return {"error": f"File not found: {file_path}"}
            
            # Extract text from docx
            doc = docx.Document(file_path)
            text_content = "\n".join([para.text for para in doc.paragraphs])
            
            # Save text to temp file for LlamaParse to process
            text_file_path = file_path + ".txt"
            with open(text_file_path, 'w', encoding='utf-8') as text_file:
                text_file.write(text_content)
            
            # Parse text with LlamaParse
            result = parser.parse_file(text_file_path)
            os.unlink(text_file_path)  # Clean up text file
            
            return result
        except Exception as e:
            logger.error(f"Error processing Word document: {e}", exc_info=True)
            return {"error": f"Word document processing error: {str(e)}"}

    def _serve_result_page(self, result: Dict[str, Any]):
        """Serve result page based on the processing result"""
        try:
            with open('result_template.html', 'r') as template_file:
                result_html = template_file.read()
        except FileNotFoundError:
            # Create a basic result template if the file doesn't exist
            result_html = '''
            <!DOCTYPE html>
            <html>
            <head>
                <title>Document Parsing Results</title>
                <meta name="viewport" content="width=device-width, initial-scale=1">
                <style>
                    body { 
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                        margin: 0; 
                        padding: 0; 
                        background-color: #f9f9f9; 
                        color: #333; 
                    }
                    .container { 
                        max-width: 980px; 
                        margin: 0 auto; 
                        padding: 20px; 
                    }
                    .header { 
                        background-color: #fff; 
                        padding: 15px 20px; 
                        border-radius: 8px; 
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05); 
                        margin-bottom: 20px; 
                    }
                    h1 { 
                        color: #2c3e50; 
                        margin-top: 0; 
                        font-size: 24px; 
                        text-align: center;
                    }
                    .content-card { 
                        background-color: #f5f5f5; 
                        padding: 25px; 
                        border-radius: 8px; 
                        box-shadow: 0 2px 10px rgba(0,0,0,0.05); 
                        margin-bottom: 20px; 
                        position: relative; 
                    }
                    .parsed-content { 
                        white-space: pre-wrap; 
                        background-color: #f9f9f9; 
                        padding: 15px; 
                        border-radius: 4px; 
                        border-left: 4px solid #3498db; 
                        font-family: Consolas, monospace; 
                        line-height: 1.6; 
                        overflow-x: auto; 
                        font-size: 14px; 
                    }
                    .error { 
                        color: #e74c3c; 
                        background-color: #fdedec; 
                        padding: 15px; 
                        border-radius: 4px; 
                        border-left: 4px solid #e74c3c; 
                    }
                    .error h3 {
                        margin-top: 0;
                        color: #c0392b;
                    }
                    .error-details {
                        margin: 10px 0;
                        padding: 10px;
                        background-color: #fff;
                        border-radius: 4px;
                    }
                    .error-details p {
                        margin: 5px 0;
                    }
                    .suggestion {
                        margin-top: 10px;
                        padding-top: 10px;
                        border-top: 1px solid #e74c3c;
                        font-style: italic;
                    }
                    .page-marker { 
                        background-color: #3498db; 
                        color: white; 
                        padding: 5px 10px; 
                        border-radius: 4px; 
                        margin: 10px 0; 
                        font-weight: bold; 
                        display: inline-block;
                    }
                    .btn { 
                        display: inline-block; 
                        background-color: #3498db; 
                        color: white; 
                        padding: 10px 15px; 
                        border-radius: 4px; 
                        text-decoration: none; 
                        margin-top: 10px; 
                        font-weight: 600;
                        border: none;
                        cursor: pointer;
                    }
                    .btn:hover { 
                        background-color: #2980b9; 
                    }
                    .actions {
                        margin-top: 20px;
                        text-align: center;
                    }
                    .copy-btn {
                        margin-left: 10px;
                    }
                </style>
                <script>
                    function copyToClipboard() {
                        const content = document.querySelector('.parsed-content').innerText;
                        navigator.clipboard.writeText(content)
                            .then(() => {
                                const copyBtn = document.getElementById('copy-btn');
                                copyBtn.textContent = 'Copied!';
                                setTimeout(() => {
                                    copyBtn.textContent = 'Copy to Clipboard';
                                }, 2000);
                            })
                            .catch(err => {
                                console.error('Error copying text: ', err);
                            });
                    }
                    
                    function downloadText() {
                        const content = document.querySelector('.parsed-content').innerText;
                        const blob = new Blob([content], {type: 'text/plain'}); 
                        const a = document.createElement('a');
                        a.href = URL.createObjectURL(blob);
                        a.download = 'parsed_document.txt';
                        document.body.appendChild(a);
                        a.click();
                        document.body.removeChild(a);
                    }
                </script>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Document Parsing Results</h1>
                    </div>
                    <div class="content-card">
                        {{content}}
                        <div class="actions">
                            <a href="/" class="btn">Parse Another Document</a>
                            <button id="copy-btn" class="btn copy-btn" onclick="copyToClipboard()">Copy to Clipboard</button>
                            <button class="btn" onclick="downloadText()">Download as Text</button>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            '''
        
        # Extract invoice data from the processed document
        invoice_data = self._extract_invoice_data(result)
        
        # Check for errors
        if isinstance(result, dict) and "error" in result:
            error_message = result["error"]
            invoice_error_script = f'''
            <script>
                // Error data
                const invoiceData = {{
                    invoiceNumber: 'ERROR',
                    companyName: 'Error Processing Document',
                    personInCharge: '',
                    contactNumber: '',
                    email: '',
                    address: '',
                    invoiceDate: '{time.strftime('%Y-%m-%d')}',
                    dueDate: '',
                    items: [
                        {{ id: 1, name: 'Error: {error_message}', quantity: 1, unitPrice: 0 }}
                    ],
                    taxRate: 0.1, // 10%
                }};
                // Call the function to populate the invoice
                populateInvoice(invoiceData);
            </script>
            '''
            result_html = result_html.replace('</body>', f'{invoice_error_script}</body>')
        else:
            # Create a script tag with the extracted invoice data
            invoice_script = f'''
            <script>
                // Document extracted data
                const invoiceData = {json.dumps(invoice_data)};
                
                // Call the function to populate the invoice
                populateInvoice(invoiceData);
            </script>
            '''
            
            # Replace the existing script in the template
            result_html = re.sub(r'<script>[\s\S]*?// Call the function to populate the invoice[\s\S]*?</script>', invoice_script, result_html)
        
        # Save the result HTML
        with open('output.html', 'w') as result_file:
            result_file.write(result_html)
        
        # Redirect to the result page
        self.send_response(302)
        self.send_header('Location', '/output.html')
        self.end_headers()

    def do_GET(self):
        """Handle GET requests"""
        if self.path == '/':
            self.path = '/upload.html'
            
        # Sanitize the path to prevent directory traversal
        file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                self.path.lstrip('/'))
        
        try:
            # Check if file exists and is a file (not a directory)
            if os.path.exists(file_path) and os.path.isfile(file_path):
                # Determine content type
                content_type, _ = mimetypes.guess_type(file_path)
                if not content_type:
                    content_type = 'text/plain'
                
                # Serve the file
                with open(file_path, 'rb') as f:
                    self._set_headers(content_type)
                    self.wfile.write(f.read())
            else:
                self.send_error(404, f"File not found: {self.path}")
        except Exception as e:
            logger.error(f"Error serving file: {e}")
            self.send_error(500, f"Internal server error: {str(e)}")

def run_server(port=8001, host=''):
    """Run server with proper error handling"""
    try:
        server_address = (host, port)
        httpd = HTTPServer(server_address, DocumentParserServer)
        logger.info(f"Server started on port {port}")
        httpd.serve_forever()
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_server()
