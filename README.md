# Context-Aware Chatbot Using LangChain and RAG

A production-ready conversational chatbot that remembers conversation context and retrieves external information from a knowledge base using Retrieval-Augmented Generation (RAG).

## Project Overview

This project implements a complete conversational AI system with the following capabilities:

- Conversation memory (short-term context retention)
- Knowledge base retrieval using vector search (RAG)
- Multiple LLM support (Groq, Ollama, OpenAI)
- Source attribution for responses
- Streamlit web interface

## Architecture

The chatbot consists of four main components:

1. Document Processing
   - Loads documents from text, PDF, or web sources
   - Splits documents into chunks (1000 chars with 200 overlap)
   - Creates embeddings using sentence-transformers

2. Vector Storage
   - Uses Chroma DB for storing embeddings
   - Supports semantic similarity search
   - Persists to disk for reuse

3. Memory Management
   - Short-term: ConversationBufferMemory for session context
   - Long-term: VectorStoreRetrieverMemory for cross-session knowledge

4. LLM Integration
   - Primary: Groq (llama-3.1-8b-instant) for fast inference
   - Fallback: Ollama (llama3.2) for local execution
   - Configurable for OpenAI models

## Dataset / Knowledge Base

The project includes a sample knowledge base covering:
- Artificial Intelligence and Machine Learning
- Natural Language Processing
- LangChain Framework
- Retrieval-Augmented Generation (RAG)
- Vector Databases
- Python Programming
- Cloud Computing

You can extend the knowledge base by adding:
- Text files (.txt)
- PDF documents (.pdf)
- Web URLs
- Custom document sources

## Requirements

Python >= 3.10
langchain >= 0.1.0
langchain-community >= 0.0.10
langchain-groq >= 0.1.0
chromadb >= 0.4.0
sentence-transformers >= 2.2.0
streamlit >= 1.28.0
pypdf >= 3.0.0
tiktoken >= 0.5.0
python-dotenv >= 1.0.0
pandas >= 1.5.0
numpy >= 1.24.0

## Installation

1. Clone the repository:
git clone https://github.com/kanchankhatri12/DeveloperHub-phase-2-task4.git
cd context-aware-chatbot

2. Install dependencies:
pip install -r requirements.txt

3. Set up API keys (for Groq/OpenAI):
Create a .env file with:
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here

4. (Optional) Install Ollama for local LLM:
curl -fsSL https://ollama.com/install.sh | sh
ollama pull llama3.2

## Usage

### Running the Jupyter Notebook

Open and run the complete implementation:
jupyter notebook context_aware_chatbot.ipynb

Run all cells sequentially to:
- Build the knowledge base and vector store
- Initialize the chatbot with memory
- Test the chatbot interactively
- Export the chatbot for production

### Running the Streamlit Web App

After running the notebook to build the vector store:
streamlit run chatbot_app.py

The app will be available at http://localhost:8501

### Interactive Command Line

In the notebook, run:
interactive_chat()

Commands:
- Type your question to chat
- Type 'clear' to clear conversation memory
- Type 'history' to see conversation history
- Type 'quit' to exit

### Using the Chatbot Programmatically

from chatbot_app import ContextAwareChatbot, KnowledgeBaseBuilder

Load or create knowledge base
kb_builder = KnowledgeBaseBuilder("./knowledge_chroma_db")
if os.path.exists("./knowledge_chroma_db"):
    vectorstore = kb_builder.load_existing_vectorstore()
else:
    documents = kb_builder.create_sample_documents()
    vectorstore = kb_builder.build_vectorstore(documents)

Initialize chatbot
chatbot = ContextAwareChatbot(
    vectorstore=vectorstore,
    model_name="llama-3.1-8b-instant",
    use_groq=True
)

Chat
response = chatbot.chat("What is RAG?")
print(response["answer"])
print(response["source_documents"])

## Configuration Options

### LLM Configuration

| Parameter | Options | Default |
|-----------|---------|---------|
| use_groq | True/False | True |
| model_name | llama-3.1-8b-instant, gemma-7b-it, mixtral-8x7b | llama-3.1-8b-instant |
| temperature | 0.0 to 1.0 | 0.7 |

### Memory Configuration

| Parameter | Options | Default |
|-----------|---------|---------|
| memory_type | BufferMemory, WindowMemory, VectorMemory | BufferMemory |
| window_size | Integer | 5 |
| return_messages | True/False | True |

### Retrieval Configuration

| Parameter | Options | Default |
|-----------|---------|---------|
| chunk_size | Integer | 1000 |
| chunk_overlap | Integer | 200 |
| k (retrieval count) | Integer | 4 |
| embedding_model | all-MiniLM-L6-v2, all-mpnet-base-v2 | all-MiniLM-L6-v2 |

## Project Structure

context-aware-chatbot/
│
├── context_aware_chatbot.ipynb     Main implementation notebook
├── chatbot_app.py                   Streamlit web application
├── requirements.txt                 Python dependencies
├── Dockerfile                       Container configuration
├── deploy.sh                        Deployment script
├── README.md                        Project documentation
│
├── knowledge_chroma_db/             Vector store directory
│   ├── chroma.sqlite3
│   └── (embedding files)
│
├── exported_chatbot/                Exported configuration
│   ├── config.json
│   └── conversation_starters.json
│
└── .env                             API keys (create this file)

## Deployment Options

### Local Deployment

streamlit run chatbot_app.py

### Docker Deployment

docker build -t context-chatbot .
docker run -p 8501:8501 --env-file .env context-chatbot

### Streamlit Cloud

1. Push code to GitHub
2. Connect repository to Streamlit Cloud
3. Add environment variables (GROQ_API_KEY)
4. Deploy

### Hugging Face Spaces

1. Create new Space with Streamlit SDK
2. Upload code files
3. Add secrets for API keys
4. Deploy

## Evaluation Metrics

The chatbot includes evaluation capabilities:

- Retrieval Quality: Number of relevant chunks retrieved per query
- Response Quality: Success rate, answer length, source attribution
- Memory Performance: Context retention across conversation turns

To evaluate:
python
test_queries = ["What is RAG?", "Explain LangChain"]
results = evaluate_retrieval(chatbot, test_queries)

## Example Conversations

User: What is RAG?
Assistant: RAG (Retrieval-Augmented Generation) is a technique that combines information retrieval with text generation. It enhances LLM responses by retrieving relevant documents from a knowledge base and using them as context.

User: How does it work?
Assistant: RAG works by first converting your question into a vector embedding, searching a vector database for similar documents, and then providing those documents as context to the LLM for generating an informed response.

User: Can you give me an example?
Assistant: Yes. If you ask "What are the benefits of RAG?", the system would:
1. Convert your question to an embedding
2. Search for documents about RAG benefits
3. Use those documents to generate a specific answer mentioning reduced hallucinations, up-to-date information, and source attribution.

## Troubleshooting

Common Issues and Solutions:

1. Groq API Key Error
   - Ensure GROQ_API_KEY is set in .env file
   - Check API key is valid
   - Fallback to Ollama by setting use_groq=False

2. Vector Store Not Found
   - Run the notebook fully to build the vector store first
   - Check that knowledge_chroma_db directory exists

3. Memory Issues
   - Reduce chunk_size to 500
   - Use smaller embedding model
   - Reduce number of retrieved documents (k parameter)

4. Slow Responses
   - Use Groq instead of Ollama for faster inference
   - Reduce retrieved document count
   - Use smaller model (llama3.2 instead of llama3.1)

## Future Improvements

- Add support for more document formats (DOCX, Excel, HTML)
- Implement conversation summarization for long sessions
- Add user authentication and multiple sessions
- Implement feedback mechanism for response quality
- Add support for streaming responses
- Integrate with Slack/Teams for production use


- Chroma for vector database
- Sentence-Transformers for embedding models
