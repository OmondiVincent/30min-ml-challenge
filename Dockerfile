# Use official lightweight Python image
FROM python:3.9-slim

# Set working directory in container
WORKDIR /app

# Copy requirements first and install dependencies (better cache)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy all code into the container
COPY . .

# Expose port 5000 for Flask app
EXPOSE 5000

# Command to run the app
CMD ["python", "app.py"]
