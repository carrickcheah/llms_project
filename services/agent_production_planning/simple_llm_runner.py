#!/usr/bin/env python
# simple_llm_runner.py - A simple application to use LLMs to run programs

import os
import sys
import json
import subprocess
import argparse
import logging
import requests
from typing import Dict, List, Optional, Union

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleLLMRunner:
    """A simpler class that uses LLMs to execute programs based on instructions."""
    
    def __init__(self, 
                 llm_api_url: Optional[str] = None, 
                 api_key: Optional[str] = None,
                 use_ollama: bool = False,
                 ollama_model: str = "llama3"):
        """Initialize the SimpleLLMRunner.
        
        Args:
            llm_api_url: URL of the LLM API endpoint
            api_key: API key for authentication (if needed)
            use_ollama: Whether to use Ollama (local LLM)
            ollama_model: The Ollama model to use if use_ollama is True
        """
        self.use_ollama = use_ollama
        
        if use_ollama:
            self.llm_api_url = "http://localhost:11434/api/generate"
            self.ollama_model = ollama_model
            logger.info(f"Using Ollama with model: {ollama_model}")
        else:
            if not llm_api_url:
                raise ValueError("LLM API URL is required when not using Ollama")
            self.llm_api_url = llm_api_url
        
        self.api_key = api_key
        
        # Define available programs and their arguments
        self.available_programs = {
            "main.py": {
                "description": "Production Planning Scheduler",
                "arguments": [
                    "--file", "--max-jobs", "--force-greedy", "--output", 
                    "--enforce-sequence", "--verbose", "--max-operators",
                    "--urgent50", "--urgent100"
                ]
            }
            # Add more programs as needed
        }
    
    def query_llm(self, prompt: str) -> str:
        """Query the LLM API with a prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response as text
        """
        if self.use_ollama:
            return self._query_ollama(prompt)
        else:
            return self._query_generic_llm(prompt)
    
    def _query_ollama(self, prompt: str) -> str:
        """Query Ollama's API with a prompt.
        
        Args:
            prompt: The prompt to send to Ollama
            
        Returns:
            Ollama's response as text
        """
        data = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,  # Lower temperature for more deterministic responses
                "num_predict": 500,   # Roughly equivalent to max_tokens
            }
        }
        
        try:
            logger.info(f"Querying Ollama with model {self.ollama_model}...")
            response = requests.post(
                self.llm_api_url,
                json=data
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except requests.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            if hasattr(e, 'response') and e.response:
                logger.error(f"Response: {e.response.text}")
            return ""
    
    def _query_generic_llm(self, prompt: str) -> str:
        """Query a generic LLM API with a prompt.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Returns:
            The LLM's response as text
        """
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        data = {
            "prompt": prompt,
            "max_tokens": 500
        }
        
        try:
            response = requests.post(
                self.llm_api_url,
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json().get("response", "")
        except requests.RequestException as e:
            logger.error(f"LLM API request failed: {e}")
            return ""
    
    def parse_llm_response(self, response: str) -> Dict:
        """Parse the LLM's response to extract program and arguments.
        
        Args:
            response: The LLM's text response
            
        Returns:
            Dictionary with program and arguments
        """
        # Try to find a JSON block in the response
        try:
            # Look for JSON between triple backticks or just try to parse the whole response
            if "```json" in response and "```" in response.split("```json", 1)[1]:
                json_str = response.split("```json", 1)[1].split("```", 1)[0].strip()
                return json.loads(json_str)
            elif "```" in response and "```" in response.split("```", 1)[1]:
                json_str = response.split("```", 1)[1].split("```", 1)[0].strip()
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    pass
            
            # Try to parse the whole response
            return json.loads(response)
        except (json.JSONDecodeError, IndexError):
            # If JSON parsing fails, try to extract program and arguments manually
            logger.warning("Failed to parse JSON from LLM response, attempting manual extraction")
            
            result = {
                "program": None,
                "arguments": [],
                "reasoning": "Extracted manually from text response"
            }
            
            lines = response.split("\n")
            for line in lines:
                if "program:" in line.lower() or "program " in line.lower():
                    parts = line.split(":", 1) if ":" in line else line.split("program", 1)
                    if len(parts) > 1 and "main.py" in parts[1]:
                        result["program"] = "main.py"
                
                if "argument" in line.lower() and "--" in line:
                    arg = line.split("--", 1)[1].strip().split()[0]
                    result["arguments"].append(f"--{arg}")
                    
                    # Try to extract the value after the argument
                    parts = line.split(f"--{arg}", 1)
                    if len(parts) > 1:
                        value = parts[1].strip().split()[0]
                        if value and not value.startswith("--"):
                            result["arguments"].append(value)
            
            return result
    
    def process_user_input(self, user_input: str) -> str:
        """Process user input to run the appropriate program.
        
        Args:
            user_input: The user's input text
            
        Returns:
            Output from the executed program or error message
        """
        try:
            # Create prompt for LLM
            prompt = f"""You are an assistant that helps run computer programs based on user instructions.
Available programs:
{json.dumps(self.available_programs, indent=2)}

User input: "{user_input}"

Based on the user's input, determine:
1. Which program to run
2. What arguments to pass to that program
3. Your reasoning

Respond with JSON in this format:
{{
  "program": "program_name.py",
  "arguments": ["--arg1", "value1", "--arg2", "value2"],
  "reasoning": "Explanation of why you chose this program and these arguments"
}}
"""
            
            # Query LLM
            logger.info("Querying LLM for program instructions...")
            llm_response = self.query_llm(prompt)
            
            if not llm_response:
                return "Error: No response from LLM API"
            
            # Parse LLM response
            instructions = self.parse_llm_response(llm_response)
            logger.info(f"Parsed instructions: {instructions}")
            
            # Validate program
            program = instructions.get("program")
            if not program:
                return "Error: No program specified in LLM response"
            
            if program not in self.available_programs:
                return f"Error: Program '{program}' is not available"
            
            # Prepare command
            command = ["uv", "run", "python", program]
            arguments = instructions.get("arguments", [])
            command.extend(arguments)
            
            # Run command
            logger.info(f"Running command: {' '.join(command)}")
            result = subprocess.run(
                command,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return f"Command failed with exit code {result.returncode}:\n{result.stderr}"
            
            return f"Command output:\n{result.stdout}"
            
        except Exception as e:
            error_message = f"Error processing request: {str(e)}"
            logger.error(error_message)
            return error_message


def main():
    """Main function to run the simple LLM application."""
    parser = argparse.ArgumentParser(description="Run programs using LLM interpretation")
    parser.add_argument("--api-url", help="LLM API endpoint URL (not needed if using Ollama)")
    parser.add_argument("--api-key", help="API key for LLM service (if needed)")
    parser.add_argument("--use-ollama", action="store_true", help="Use Ollama for local LLM inference")
    parser.add_argument("--ollama-model", default="llama3", help="Model to use with Ollama (default: llama3)")
    parser.add_argument("instruction", nargs="*", help="Instruction for the LLM")
    args = parser.parse_args()
    
    # Create the runner
    try:
        # Validate arguments
        if not args.use_ollama and not args.api_url:
            print("Error: Either --use-ollama or --api-url must be specified")
            return 1
        
        runner = SimpleLLMRunner(
            llm_api_url=args.api_url,
            api_key=args.api_key,
            use_ollama=args.use_ollama,
            ollama_model=args.ollama_model
        )
        
        # If instruction provided as argument, use it
        if args.instruction:
            instruction = " ".join(args.instruction)
            print(f"Processing instruction: {instruction}")
            output = runner.process_user_input(instruction)
            print(output)
        else:
            # Interactive mode
            print("Simple LLM Program Runner - Interactive Mode")
            if args.use_ollama:
                print(f"Using Ollama with model: {args.ollama_model}")
            else:
                print(f"Using external LLM API: {args.api_url}")
            print("Enter 'exit' or 'quit' to exit")
            
            while True:
                instruction = input("\nEnter your instruction: ")
                if instruction.lower() in ["exit", "quit"]:
                    break
                    
                output = runner.process_user_input(instruction)
                print("\nResult:")
                print(output)
                
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
        
    return 0


if __name__ == "__main__":
    sys.exit(main()) 