import os
import re
import sys
from datetime import datetime
import ollama
from ollama import ChatResponse

# Global model name setting
OLLAMA_MODEL = "xingyaow/codeact-agent-mistral"
MAX_TOKENS = 2048  

def generate_report_from_logs(log_file_path="../production_scheduler.log", max_tokens=MAX_TOKENS):
    """
    Generate a structured report from production scheduler logs using Ollama.
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
            # Don't use all lines after marker, just take the last 100 lines after the marker
            # or all lines after marker if fewer than 50
            lines_after_marker = log_lines[latest_run_start:]
            num_lines = min(50, len(lines_after_marker))
            
            # Strip timestamps from log lines to reduce token usage
            processed_lines = []
            for line in lines_after_marker[-num_lines:]:
                # Remove timestamp pattern (e.g., 2025-04-07T14:47:14.374181+0800)
                line = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+\d{4}\s+\|\s+', '', line)
                processed_lines.append(line)
            
            log_content = ''.join(processed_lines)
            print(f"Processing only the last {num_lines} lines from the latest run (starting at line {latest_run_start+1})")
        else:
            # Fallback to recent lines if no run marker found
            # Take the last 50 lines or all if fewer 
            num_lines = min(50, len(log_lines))
            
            # Strip timestamps from log lines to reduce token usage
            processed_lines = []
            for line in log_lines[-num_lines:]:
                # Remove timestamp pattern (e.g., 2025-04-07T14:47:14.374181+0800)
                line = re.sub(r'\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+\+\d{4}\s+\|\s+', '', line)
                processed_lines.append(line)
            
            log_content = ''.join(processed_lines)
            print(f"No run marker found, processing last {num_lines} lines of the log file")
    except Exception as e:
        return f"Error reading log file: {str(e)}"
    
    # Truncate log if still needed to fit token limits
    if len(log_content) > max_tokens * 3:  # Conservative estimate of characters to tokens
        # Get the last part of the log file
        log_content = log_content[-max_tokens * 3:]
        # Find the first complete log entry to avoid partial entries
        first_entry = re.search(r'\d{4}-\d{2}-\d{2}', log_content)
        if first_entry:
            log_content = log_content[first_entry.start():]
        else:
            # If no timestamp found, just take the last n characters as a fallback
            log_content = log_content[-max_tokens * 2:]
    
    print(f"Log content length: {len(log_content)} characters, approximately {len(log_content)/4} tokens")
    
    # Prepare system prompt for structured analysis
    system_prompt = (
        "You are an expert in production scheduling analysis. Analyze the log content and create a concise report with exactly 5 paragraphs of about 30 words each:\n"
        "1. Overall schedule status and key metrics\n\n"
        "2. Most critical late jobs with specific details\n\n"
        "3. Machine utilization and capacity analysis\n\n"
        "4. Highest priority concerns\n\n"
        "5. Immediate recommendations\n\n"
        "Format each paragraph as plain text with no markdown, no HTML, no bold, no italics, no formatting. Use clear, concise language. Do not include any interactive elements or questions at the end."
    )
    
    # Prepare the chat messages
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"Analyze this production scheduler log and create a concise 5-paragraph report:\n\n{log_content}"}
    ]
    
    print(f"Using Ollama model: {OLLAMA_MODEL}")
    
    # Generate the report using Ollama
    try:
        # Use Ollama Python library to call the model
        response: ChatResponse = ollama.chat(
            model=OLLAMA_MODEL,
            messages=messages,
            options={"temperature": 0.7}
        )
        return response.message.content
        
    except Exception as e:
        print(f"Error using Ollama API: {str(e)}")
        print("Falling back to basic report generation")
        return generate_basic_report(log_content)

def generate_basic_report(log_content):
    """Generate a basic report without using LLM when API calls fail"""
    # Extract some basic info from logs
    late_jobs = []
    for line in log_content.split('\n'):
        if "late" in line.lower() and "job" in line.lower():
            if len(line) > 20:  # Only include meaningful lines
                late_jobs.append(line)
    
    late_jobs = late_jobs[:5]  # Limit to 5 examples
    
    report = """
## Production Schedule Analysis

### Executive Summary
The production schedule appears to have some jobs that will complete after their due dates.

### Key Metrics
Several jobs are showing as late in the schedule.

### Issues/Warnings
"""
    
    if late_jobs:
        report += "Late jobs detected:\n"
        for job in late_jobs:
            report += f"- {job.strip()}\n"
    else:
        report += "No specific issues could be extracted from the logs.\n"
    
    report += """
### Performance Analysis
The system is generating schedules but may need optimization to reduce late jobs.

### Recommendations
Consider resource reallocation or adjusting due dates where possible.
"""
    
    return report

def generate_html_report(log_file_path="../production_scheduler.log", max_tokens=MAX_TOKENS):
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
    
    # Format the report in HTML
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
        .report-content {{
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 8px;
            line-height: 1.7;
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
        # Generate HTML report
        print("Generating HTML report... This may take up to 30 seconds...")
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