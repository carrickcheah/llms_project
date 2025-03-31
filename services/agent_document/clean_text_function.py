def clean_pdf_text(text):
    """Clean up PDF extracted text for better readability"""
    import re
    
    # Replace common encoding issues
    replacements = {
        'â€"': '-',
        'â€™': "'",
        'â€œ': '"',
        'â€': '"',
        'â€¦': '...',
        'Â': ' ',
        'Ā·': ' ',
        'ā€"': '-',
        'ā€"': '"',
        'ā€"': '"',
        'ā€¦': '...'
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Fix spaced-out text (common PDF extraction issue)
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # If this line appears to have spaced-out text (characters with spaces between them)
        if re.search(r'\b\w\s\w\s\w', line.strip()):
            # First, compress single characters with spaces between them
            line = re.sub(r'([a-zA-Z0-9])\s+([a-zA-Z0-9])', r'\1\2', line)
            
            # Then fix common patterns like dates and numbers
            # Fix dates (e.g., "Fe br ua ry" -> "February")
            line = re.sub(r'([A-Za-z])\s+([a-z])', r'\1\2', line)
            
            # Fix numbers with spaces (e.g., "2 0 2 5" -> "2025")
            line = re.sub(r'(\d)\s+(\d)', r'\1\2', line)
            
            # Fix email addresses
            line = re.sub(r'([a-zA-Z0-9])\s+@\s+([a-zA-Z0-9])', r'\1@\2', line)
            line = re.sub(r'([a-zA-Z0-9])\s+\.\s+([a-zA-Z0-9])', r'\1.\2', line)
            
            # Fix currency amounts
            line = re.sub(r'\$\s+(\d)', r'$\1', line)
            
        cleaned_lines.append(line)
        
    text = '\n'.join(cleaned_lines)
    
    # Clean up other spacing issues
    text = re.sub(r'\s{2,}', ' ', text)  # Replace multiple spaces with one
    text = re.sub(r'\n{3,}', '\n\n', text)  # Limit consecutive newlines
    
    # Fix common patterns in the entire text
    text = re.sub(r'(\d)\s+(\d)\s+(\d)\s+(\d)', r'\1\2\3\4', text)  # Fix 4-digit numbers
    text = re.sub(r'([A-Za-z])\s+([a-z])\s+([a-z])', r'\1\2\3', text)  # Fix 3-letter words
    
    # Format receipt-like documents
    lines = text.split('\n')
    formatted_lines = []
    
    for line in lines:
        # Format header lines
        if re.search(r'Receipt|Invoice', line, re.IGNORECASE):
            formatted_lines.append('\n' + line.strip().upper() + '\n')
            continue
            
        # Format date lines
        if re.search(r'\d{1,2}\s+[A-Za-z]+\s+\d{4}', line):
            formatted_lines.append(line.strip())
            continue
            
        # Format address lines
        if re.search(r'@|Street|Avenue|Road|Lane|Drive|Boulevard', line, re.IGNORECASE):
            formatted_lines.append(line.strip())
            continue
            
        # Format amount lines
        if re.search(r'\$\s*\d+\.\d{2}', line):
            # Align amounts to the right
            parts = line.split('$')
            if len(parts) > 1:
                amount = parts[1].strip()
                description = parts[0].strip()
                formatted_line = f"{description:40} ${amount:>10}"
            else:
                formatted_line = line.strip()
            formatted_lines.append(formatted_line)
            continue
            
        # Format other lines
        formatted_lines.append(line.strip())
    
    # Join lines with proper spacing
    text = '\n'.join(formatted_lines)
    
    # Add section separators
    text = re.sub(r'\n\n+', '\n\n', text)  # Remove extra blank lines
    text = re.sub(r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\n', r'\n\1\n', text)  # Add spacing around section headers
    
    return text
