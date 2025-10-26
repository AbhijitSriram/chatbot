<<<<<<< HEAD
# chatbot
=======
# PDF Chatbot

A Streamlit-based chatbot that can process PDF documents and answer questions based on their content.

## Features

- PDF document upload and processing
- Interactive chat interface
- Document-based question answering
- Chat history tracking
- Secure API key management

## Requirements

- Python 3.8+
- Dependencies listed in requirements.txt

## Local Development

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the app:
   ```bash
   ```markdown
# PDF Chatbot

A Streamlit-based chatbot that can process PDF documents and answer questions based on their content using OpenAI's GPT-4 and HuggingFace embeddings.

## Features

- PDF document upload and processing
- Interactive chat interface with GPT-4
- Document-based question answering
- Chat history tracking
- Secure API key management
- HuggingFace embeddings for document processing
- Automatic API key validation
- Error handling and recovery

## Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`
- OpenAI API key
- HuggingFace API token

## Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/AbhijitSriram/chatbot.git
   cd chatbot
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your API keys to `.env`:
     ```
     OPENAI_API_KEY=your-openai-api-key
     HUGGINGFACEHUB_API_TOKEN=your-huggingface-token
     ```

## Running Locally

Start the Streamlit app:
```bash
streamlit run app.py
```

The app will be available at http://localhost:8501

## Deployment

### Deploy on Streamlit Cloud

1. Fork this repository
2. Visit https://share.streamlit.io
3. Deploy from your forked repository
4. Add your API keys in Streamlit Cloud:
   - Go to App Settings > Secrets
   - Add the following secrets:
     ```toml
     OPENAI_API_KEY = "your-openai-api-key"
     HUGGINGFACEHUB_API_TOKEN = "your-huggingface-token"
     ```

### Security Notes

- Never commit API keys or sensitive information to the repository
- Use environment variables or Streamlit secrets for API keys
- The `.env` file is ignored by git for security
- API keys are validated before use to prevent errors

## Usage

1. Enter your OpenAI API key (if not set in environment/secrets)
2. Upload one or more PDF documents
3. Ask questions about the content of the documents
4. View chat history and responses
5. Use the "Reset API Key" button in the sidebar if needed

## Error Handling

The app includes robust error handling for:
- Invalid API keys
- PDF processing errors
- OpenAI API errors
- Missing environment variables

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to your branch
5. Create a Pull Request   ```
