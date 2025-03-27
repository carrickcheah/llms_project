FROM python:3.9-slim

WORKDIR /app

# Copy requirements first to leverage Docker cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application
COPY . .

# Make port 8000 available to the world outside this container
EXPOSE 8000

# Create a non-root user to run our application
RUN useradd -m appuser
USER appuser

# Command to run when the container starts
CMD ["sh", "-c", "python main.py && python -m http.server 8000"]
