import streamlit as st
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from urllib.parse import urlparse, parse_qs
import os

load_dotenv()

# Configure Streamlit page
st.set_page_config(
    page_title="YouTube Chat Bot",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #FF0000;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #2196F3;
    }
    .bot-message {
        background-color: #F3E5F5;
        border-left: 4px solid #9C27B0;
    }
    .sidebar-content {
        background-color: #F5F5F5;
        padding: 1rem;
        border-radius: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# YouTube chatbot functions (from the original script)
def extract_youtube_id(url):
    """Extract video ID from YouTube URL"""
    parsed_url = urlparse(url)
    video_id = parse_qs(parsed_url.query).get("v")
    return video_id[0] if video_id else None

def extract_transcript(id):
    """Extract transcript from YouTube video"""
    if id is None:
        raise ValueError("Invalid youtube URL")
    try:
        ytt_api = YouTubeTranscriptApi()
        fetched_transcript = ytt_api.fetch(id, languages=['en'])
        transcript = " ".join([chunk.text for chunk in fetched_transcript]) 
        return transcript
    except TranscriptsDisabled:
        return None
    except Exception as e:
        st.error(f"Error extracting transcript: {e}")
        return None

def text_splitter(text):
    """Split text into chunks"""
    if text is None:
        raise ValueError("No text is available")
    try:
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_text(text)
        return chunks
    except Exception as e:
        st.error(f"Error splitting text: {e}")
        return None

def create_vector_store(text_chunks):
    """Create FAISS vector store from text chunks"""
    if text_chunks is None:
        raise ValueError("No text chunks are available")
    try:
        embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        vector_store = FAISS.from_texts(text_chunks, embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def search_documents(query, vector_store, k=3):
    """Search for relevant documents in vector store"""
    if query is None or vector_store is None:
        raise ValueError("No query or vector store is available")
    try:
        retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": k})
        results = retriever.invoke(query)  
        return results
    except Exception as e:
        st.error(f"Error searching documents: {e}")
        return None

def generate_response(query, vector_store):
    """Generate response using LLM based on relevant documents"""
    if query is None or vector_store is None:
        raise ValueError("No query or vector store is available")
    try:
        # Get relevant documents
        relevant_docs = search_documents(query, vector_store)
        if not relevant_docs:
            return "No relevant information found in the video transcript."
        
        # Combine the content from relevant documents
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Create prompt template
        prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template="""Based on the following context from a YouTube video transcript, please answer the question:

Context:
{context}

Question: {question}

Answer:"""
        )
        
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
        
        # Generate response
        prompt = prompt_template.format(context=context, question=query)
        response = llm.invoke(prompt)
        
        return response.content
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

def process_youtube_video(url):
    """Process YouTube video and create vector store"""
    with st.spinner("Processing YouTube video..."):
        # Extract video ID
        video_id = extract_youtube_id(url)
        if not video_id:
            st.error("Invalid YouTube URL. Please provide a valid YouTube video URL.")
            return None
        
        st.info(f"Video ID: {video_id}")
        
        # Extract transcript
        transcript = extract_transcript(video_id)
        if transcript is None:
            st.error("Failed to extract transcript. The video might not have captions available.")
            return None
        
        st.success(f"Transcript extracted: {len(transcript)} characters")
        
        # Split text into chunks
        text_chunks = text_splitter(transcript)
        if text_chunks is None:
            st.error("Failed to split text into chunks.")
            return None
        
        st.success(f"Text split into {len(text_chunks)} chunks")
        
        # Create vector store
        vector_store = create_vector_store(text_chunks)
        if vector_store is None:
            st.error("Failed to create vector store.")
            return None
        
        st.success("Vector store created successfully! You can now ask questions about the video.")
        return vector_store

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'video_processed' not in st.session_state:
    st.session_state.video_processed = False

# Main app layout
st.markdown('<h1 class="main-header">🎥 YouTube Chat Bot</h1>', unsafe_allow_html=True)
st.markdown("### Ask questions about any YouTube video!")

# Sidebar for video input
with st.sidebar:
    st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
    st.header("📹 Video Setup")
    
    # YouTube URL input
    youtube_url = st.text_input(
        "Enter YouTube URL:",
        placeholder="https://www.youtube.com/watch?v=...",
        help="Paste the URL of the YouTube video you want to chat about"
    )
    
    # Process video button
    if st.button("🔄 Process Video", type="primary"):
        if youtube_url:
            st.session_state.vector_store = process_youtube_video(youtube_url)
            if st.session_state.vector_store:
                st.session_state.video_processed = True
                st.session_state.chat_history = []  # Clear chat history for new video
        else:
            st.error("Please enter a YouTube URL first.")
    
    # Clear chat button
    if st.button("🗑️ Clear Chat"):
        st.session_state.chat_history = []
        st.rerun()
    
    # Video status
    if st.session_state.video_processed:
        st.success("✅ Video processed and ready for questions!")
    else:
        st.info("👆 Enter a YouTube URL and click 'Process Video' to start chatting.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main chat interface
col1, col2 = st.columns([3, 1])

with col1:
    # Chat history display
    st.subheader("💬 Chat History")
    
    # Display chat history
    for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
        st.markdown(f'<div class="chat-message user-message"><strong>You:</strong> {user_msg}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="chat-message bot-message"><strong>Bot:</strong> {bot_msg}</div>', unsafe_allow_html=True)
    
    # Chat input
    if st.session_state.video_processed:
        # Initialize clear_input flag
        if 'clear_input' not in st.session_state:
            st.session_state.clear_input = False
        
        # Use a unique key that changes when we want to clear
        input_key = f"user_input_{len(st.session_state.chat_history)}"
        
        user_question = st.text_input(
            "Ask a question about the video:",
            placeholder="What is this video about?",
            key=input_key
        )
        
        col_send, col_example = st.columns([1, 2])
        
        with col_send:
            send_button = st.button("📤 Send", type="primary")
        
        with col_example:
            st.caption("💡 Try asking: 'What are the main points?', 'Summarize the video', 'What is mentioned about...?'")
        
        # Process user question
        if send_button and user_question:
            with st.spinner("Thinking..."):
                response = generate_response(user_question, st.session_state.vector_store)
                if response:
                    st.session_state.chat_history.append((user_question, response))
                    st.rerun()
                else:
                    st.error("Failed to generate response. Please try again.")
    
    else:
        st.info("🎬 Please process a YouTube video first to start chatting!")

with col2:
    # Instructions and tips
    st.subheader("📋 How to Use")
    st.markdown("""
    1. **Enter YouTube URL** in the sidebar
    2. **Click 'Process Video'** to analyze the transcript
    3. **Ask questions** about the video content
    4. **Get AI-powered answers** based on the transcript
    """)
    
    st.subheader("💡 Tips")
    st.markdown("""
    - Make sure the video has captions/subtitles
    - Ask specific questions for better answers
    - Try different phrasings if needed
    - Use 'Clear Chat' to start fresh
    """)
    
    st.subheader("⚙️ Features")
    st.markdown("""
    - 🎯 Semantic search through video content
    - 🧠 AI-powered responses
    - 💾 Chat history preservation
    - 🔄 Easy video switching
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, LangChain, and OpenAI")
