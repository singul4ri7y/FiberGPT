## Installation

### 1. Set up Virtual Environment

Ensure you are in the `web` directory:

```bash
# Create venv inside web/
python3 -m venv venv

# Activate venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

## Usage

### Starting the Server

Ensure your virtual environment is active, then run:

```bash
python3 server.py
# OR using the venv python directly:
./venv/bin/python server.py
```

By default, this starts the server at `http://0.0.0.0:8000` with 1 worker.


### Accessing the Chat

Open your browser and navigate to:
[http://localhost:8000](http://localhost:8000)

## File Structure

- **`server.py`**: The main FastAPI application. Handles static files, API endpoints (`/chat/completions`), and the worker pool.
- **`index.html`**: The main HTML structure for the chat interface.
- **`style.css`**: CSS styles for the dark/light themes, animations ("pop"), and layout.
- **`script.js`**: Frontend logic for message handling, streaming (SSE parsing), and state management.
- **`requirements.txt`**: Python dependencies (`fastapi`, `uvicorn`, `pydantic`).
- **`logs/`**: (Optional) Server logs are output to console by default.

## API Endpoints

- **`GET /`**: Serves the Chat UI.
- **`POST /chat/completions`**: Streaming Chat API.
    - **Body**: `{"messages": [...], "temperature": 0.7, ...}`
    - **Returns**: `text/event-stream`
- **`GET /health`**: Health check returning worker pool status.
- **`GET /stats`**: Worker utilization statistics.
