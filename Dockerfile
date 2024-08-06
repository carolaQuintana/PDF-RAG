FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the application code to the working directory
COPY . .

RUN pip install uvicorn fastapi

# Expose the port on which the application will run
EXPOSE 8080

# Run the FastAPI application using uvicorn server
CMD ["uvicorn", "fastapi:app", "--host", "0.0.0.0", "--port", "8080"]