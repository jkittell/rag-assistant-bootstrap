FROM python:3.11-slim

WORKDIR /app

# Install development dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application code
COPY . .

# Development port
EXPOSE 8000

# Command for production (DevSpace will override this in development)
CMD ["python", "app.py"]
