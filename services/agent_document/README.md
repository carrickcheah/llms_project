# Document Parser Application

This application parses PDF and Word documents using LlamaCloud's parsing capabilities. It provides a simple web interface for uploading documents and displays the parsed content.

## Features

- Upload PDF and Word documents through a web interface
- Parse documents using LlamaCloud's LlamaParse
- Display parsed content in a structured format

## Setup

1. Create a `.env` file with your LlamaCloud API key:

```env
LLAMA_CLOUD_API_KEY=llx-xxxxxx
```

1. Install dependencies using `uv`:

```bash
cd services/agent_document
uv add llama-cloud-services llama-index-core llama-index-readers-file llama-index-llms-openai llama-index-embeddings-openai python-dotenv python-docx
```

## Usage

1. Run the server:

```bash
python server.py
```

1. Open a browser and navigate to `http://localhost:8000`

1. Upload a PDF or Word document and click 'Parse Document'

1. View the parsed document content

## Files

- `server.py`: Web server that handles file uploads and document parsing
- `upload.html`: User interface for document upload
- `result_template.html`: Template for displaying parsing results
- `requirements.txt`: List of required Python packages
