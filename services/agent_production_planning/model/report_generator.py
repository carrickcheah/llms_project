from openai import OpenAI
import os
import re
import sys
from datetime import datetime
from time import sleep


client = OpenAI(
    base_url = 'http://localhost:11434/v1',
    api_key='ollama',
)

def generate_report_from_logs(log_file_path="../production_scheduler.log", max_tokens=5000):
    """
    Generate a structured report from production scheduler logs using LLM.
    This is an internal function used by generate_html_report.
    Only processes the latest run data from the log file.
    
    Args:
        log_file_path: Path to the log file
        max_tokens: Maximum number of tokens to process in a single API call
    
    Returns:
        The generated report as a string
    """
    # Check if the log file exists
    if not os.path.exists(log_file_path):
        return f"Error: Log file '{log_file_path}' not found."
    
    # Read log content
    try:
        with open(log_file_path, 'r') as file:
            log_lines = file.readlines()
        
        if not log_lines:
            return f"Error: Log file '{log_file_path}' is empty."
        
        # Find the start of the most recent run (latest occurrence of "Reference time initialized")
        latest_run_start = -1
        for i, line in enumerate(log_lines):
            if "Reference time initialized" in line:
                latest_run_start = i
        
        # If we found a marker for the latest run, extract only those lines
        if latest_run_start >= 0:
            log_content = ''.join(log_lines[latest_run_start:])
            print(f"Processing only the latest run data from line {latest_run_start+1} of the log file")
        else:
            # Fallback to recent lines if no run marker found
            # Take the last 500 lines or all if fewer
            num_lines = min(500, len(log_lines))
            log_content = ''.join(log_lines[-num_lines:])
            print(f"No run marker found, processing last {num_lines} lines of the log file")
    except Exception as e:
        return f"Error reading log file: {str(e)}"
    
    # Truncate log if still needed to fit token limits
    if len(log_content) > max_tokens * 4:  # Rough estimate of characters to tokens
        # Get the last part of the log file
        log_content = log_content[-max_tokens * 4:]
        # Find the first complete log entry to avoid partial entries
        first_entry = re.search(r'\d{4}-\d{2}-\d{2}', log_content)
        if first_entry:
            log_content = log_content[first_entry.start():]
    
    # Get current date for the report
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Prepare the prompt for the LLM - more concise for faster response
    system_prompt = (
        "Create a concise production scheduling report with: 1) Executive summary, 2) Key metrics, "
        "3) Issues/warnings, 4) Performance analysis, 5) Brief recommendations. Use bullet points where possible."
    )
    
    user_prompt = f"Generate a production scheduling report based on these logs:\n\n{log_content}"
    
    # Specify OpenAI model directly
    model = "gemma3"  # You can also use "gpt-4" for more advanced analysis
    
    # Generate the report using the LLM
    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model=model,
            temperature=0.3,  # Lower temperature for more factual output
            max_tokens=5000,   # Reduced token limit for faster response
            timeout=65  # Reduced timeout for faster total processing
        )
        
        report = response.choices[0].message.content
        return report
    
    except Exception as e:
        return f"Error generating report: {str(e)}"

def generate_html_report(log_file_path="../production_scheduler.log", max_tokens=5000):
    """
    Generate an HTML report from production scheduler logs.
    
    Args:
        log_file_path: Path to the log file
        max_tokens: Maximum number of tokens to process in a single API call
    
    Returns:
        HTML report as a string
    """
    # First get the text report
    text_report = generate_report_from_logs(log_file_path, max_tokens)
    
    if text_report.startswith("Error"):
        # Use triple quotes for CSS
        error_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Error Report</title>
    <style>
        :root {{
            --primary-color: #e74c3c;
            --text-color: #333;
            --background-color: #f9f9f9;
        }}
        body {{
            font-family: 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: var(--text-color);
            background-color: var(--background-color);
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 800px;
            margin: 40px auto;
            padding: 30px;
            background-color: #fff;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
        }}
        h1 {{
            color: var(--primary-color);
            margin-top: 0;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .error-content {{
            padding: 20px;
            background-color: #fff8f8;
            border-radius: 4px;
            border-left: 4px solid var(--primary-color);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Error Report</h1>
        <div class="error-content">
            <p>{text_report}</p>
        </div>
    </div>
</body>
</html>"""
        return error_html
    
    # Get current date for the report
    current_date = datetime.now().strftime("%Y-%m-%d")
    report_time = datetime.now().strftime("%H:%M:%S")
    
    # Skip the second API call and use a simplified template to wrap the report
    # This saves significant time by avoiding a second LLM call
    formatted_report = text_report.replace("\n", "<br>")
    
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Production Scheduler Report - {current_date}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            background-color: #f5f5f5;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1000px;
            margin: 0 auto;
            background-color: #fff;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        header {{
            margin-bottom: 30px;
            border-bottom: 2px solid #1a73e8;
            padding-bottom: 20px;
        }}
        h1 {{
            color: #1a73e8;
            margin: 0 0 10px 0;
        }}
        h2 {{
            color: #1a73e8;
            margin-top: 30px;
            border-bottom: 1px solid #eee;
            padding-bottom: 10px;
        }}
        .meta-info {{
            display: flex;
            justify-content: space-between;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 30px;
        }}
        .meta-item {{
            text-align: center;
        }}
        .meta-label {{
            font-size: 0.9rem;
            color: #666;
            margin-bottom: 5px;
        }}
        .meta-value {{
            font-size: 1.1rem;
            font-weight: 600;
        }}
        pre {{
            white-space: pre-wrap;
            word-wrap: break-word;
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
        }}
        ul, ol {{
            padding-left: 25px;
        }}
        li {{
            margin-bottom: 8px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px 15px;
            border: 1px solid #ddd;
            text-align: left;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: 600;
        }}
        tr:nth-child(even) {{
            background-color: #f8f9fa;
        }}
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>Production Scheduler Report</h1>
            <p>Analysis of production scheduling operations</p>
        </header>
        
        <div class="meta-info">
            <div class="meta-item">
                <div class="meta-label">Report Date</div>
                <div class="meta-value">{current_date}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Report Time</div>
                <div class="meta-value">{report_time}</div>
            </div>
            <div class="meta-item">
                <div class="meta-label">Log File</div>
                <div class="meta-value">production_scheduler.log</div>
            </div>
        </div>
        
        <div class="report-content">
            {formatted_report}
        </div>
    </div>
</body>
</html>"""
    
    return html_template

def save_html_report(html_report, output_file=None):
    """
    Save the generated HTML report to a file.
    
    Args:
        html_report: The HTML report content
        output_file: Destination file. If None, uses a fixed filename
    
    Returns:
        Path to the saved report file
    """
    if output_file is None:
        # Use a fixed filename instead of a timestamp
        output_file = f"../production_report.html"
    
    try:
        with open(output_file, 'w') as file:
            file.write(html_report)
        return output_file
    except Exception as e:
        return f"Error saving report: {str(e)}"

# Example usage
if __name__ == "__main__":
    try:
        # Print which model we're using
        model = "gemma3"
        print(f"Using model: {model}")
        
        # Generate HTML report
        print("Generating HTML report... This may take up to 30 seconds...")
        # Removed sleep to save time
        html_report = generate_html_report()
        
        # Save the HTML report with fixed filename
        output_file = save_html_report(html_report)
        
        if not output_file.startswith("Error"):
            print(f"HTML Report generated and saved to {output_file}")
            print(f"You can open this HTML file in a web browser to view the formatted report.")
        else:
            print(f"Error: {output_file}")
            
        # Explicitly exit the program
        print("Report generation complete. Exiting program.")
        sys.exit(0)
        
    except Exception as e:
        print(f"An unexpected error occurred: {str(e)}")
        sys.exit(1)