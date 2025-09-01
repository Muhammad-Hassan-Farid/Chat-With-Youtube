# 🎬 Chat-With-YouTube

An AI-powered RAG (Retrieval-Augmented Generation) system that enables interactive conversations with any YouTube video content. Transform any YouTube video into an intelligent chatbot that can answer questions, provide summaries, and discuss the video's content in depth.

## 🚀 Features

- **Smart Video Analysis**: Automatically extracts and processes YouTube video transcripts
- **AI-Powered Chat**: Engage in natural conversations about video content
- **Contextual Responses**: Get accurate answers based on the actual video content
- **Multi-Topic Support**: Works with videos on any topic or subject
- **Real-time Processing**: Fast transcript extraction and embedding generation
- **Semantic Search**: Find relevant information using advanced vector search

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI/Flask
- **AI/ML**: OpenAI GPT, LangChain, Vector Embeddings
- **Database**: Vector Database (Chroma/Pinecone/FAISS)
- **YouTube Integration**: YouTube Transcript API
- **Frontend**: Streamlit/HTML/CSS/JavaScript
- **Deployment**: Docker, Cloud Platform Ready

## 📋 Prerequisites

Before running this project, make sure you have:

- Python 3.8 or higher
- OpenAI API key
- YouTube Data API key (if required)
- Git

## ⚡ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/Muhammad-Hassan-Farid/Chat-With-Youtube.git
cd Chat-With-Youtube
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables
Create a `.env` file in the root directory:
```bash
OPENAI_API_KEY=your_openai_api_key_here
YOUTUBE_API_KEY=your_youtube_api_key_here  # Optional
```

### 5. Run the Application
```bash
streamlit run app.py
# or
python main.py
```

## 📖 Usage

1. **Start the Application**: Launch the app using the command above
2. **Enter YouTube URL**: Paste any YouTube video URL into the input field
3. **Wait for Processing**: The system will extract and process the video transcript
4. **Start Chatting**: Ask questions about the video content and get intelligent responses

### Example Conversations

**User**: "What are the main points discussed in this video?"
**AI**: "Based on the video transcript, the main points covered are..."

**User**: "Can you summarize the section about machine learning?"
**AI**: "The machine learning section discusses..."

## 🏗️ Project Structure

```
Chat-With-Youtube/
├── app.py                 # Main Streamlit application
├── main.py               # Alternative entry point
├── requirements.txt      # Python dependencies
├── .env.example         # Environment variables template
├── README.md            # Project documentation
├── src/
│   ├── __init__.py
│   ├── youtube_processor.py    # YouTube transcript extraction
│   ├── embeddings.py          # Vector embedding generation
│   ├── chat_engine.py         # RAG chat implementation
│   └── utils.py              # Utility functions
├── static/
│   ├── css/
│   ├── js/
│   └── images/
├── templates/            # HTML templates (if using Flask)
├── data/                # Temporary data storage
└── tests/               # Unit tests
```

## 🔧 Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `OPENAI_API_KEY` | OpenAI API key for GPT models | Yes |
| `YOUTUBE_API_KEY` | YouTube Data API key | Optional |
| `VECTOR_DB_PATH` | Path to vector database | No |
| `MODEL_NAME` | OpenAI model to use | No |

### Customization Options

- **Model Selection**: Choose between different OpenAI models (GPT-3.5, GPT-4)
- **Chunk Size**: Adjust transcript chunking for better context
- **Vector Database**: Switch between different vector database providers
- **UI Theme**: Customize the interface appearance

## 🚀 Deployment

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t chat-with-youtube .
```

2. Run the container:
```bash
docker run -p 8501:8501 --env-file .env chat-with-youtube
```

### Cloud Deployment

The application is ready for deployment on:
- **Streamlit Cloud**
- **Heroku**
- **AWS EC2/ECS**
- **Google Cloud Platform**
- **Azure App Service**

## 🤝 Contributing

We welcome contributions! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Make your changes**: Implement your feature or bug fix
4. **Run tests**: `python -m pytest tests/`
5. **Commit your changes**: `git commit -m 'Add amazing feature'`
6. **Push to the branch**: `git push origin feature/amazing-feature`
7. **Open a Pull Request**

### Development Guidelines

- Follow PEP 8 style guidelines
- Write unit tests for new features
- Update documentation as needed
- Use meaningful commit messages

## 🐛 Troubleshooting

### Common Issues

**Issue**: "No transcript available for this video"
**Solution**: Some videos don't have auto-generated transcripts. Try with a different video or enable manual captions.

**Issue**: "OpenAI API rate limit exceeded"
**Solution**: Check your API usage and upgrade your plan if necessary.

**Issue**: "Vector database connection error"
**Solution**: Ensure the vector database service is running and accessible.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenAI for providing powerful language models
- LangChain for the RAG framework
- YouTube for transcript accessibility
- Streamlit for the amazing web app framework

## 📞 Contact

**Muhammad Hassan Farid**
- GitHub: [@Muhammad-Hassan-Farid](https://github.com/Muhammad-Hassan-Farid)
- LinkedIn: [@Muhammad-Hassan-Farid](https://www.linkedin.com/in/muhammad-hassan-farid/)
- Email: your.email@example.com

## ⭐ Show Your Support

If you found this project helpful, please give it a star! It helps others discover the project and motivates continued development.
