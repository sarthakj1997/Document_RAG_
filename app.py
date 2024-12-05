import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
import tempfile
import os
from dotenv import load_dotenv
load_dotenv()

# Initialize the embeddings model and vector store directory
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
hf_embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
persist_directory = "chroma_gradio_db"  # Directory to persist vector store

# Global variable for Chroma DB (can be refreshed with new uploads)
chroma_db = None

# Function to handle PDF upload and process it
def process_pdf(uploaded_file):
    # Ensure file is handled as bytes
    if isinstance(uploaded_file, str):  # If the input is a file path
        temp_file_path = uploaded_file
    else:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file)  # Write content as binary
            temp_file_path = temp_file.name

    # Load and parse the PDF
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    global vector_store

    # Step 2: Split the PDF into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)

    vector_store = Chroma(
        collection_name="test_collection",
        embedding_function=hf_embeddings,
        persist_directory="./data/app_test",  # Where to save data locally, remove if not necessary
    )
    vector_store.add_documents(documents=chunks)

    return "PDF has been successfully processed and stored in the vector database!"

# Function to answer questions
def answer_question(question):
    if vector_store is None:
        return "Please upload and process a PDF document first!"
    
    # Retrieve context from Chroma DB
    results = vector_store.similarity_search(question, k=3)
    context = "\n".join([result.page_content for result in results])
    
    prompt_template = ChatPromptTemplate.from_template("""
                                                You are a helpful assistant. Answer the question {question}as truthfully as possible based on the give chunks of text:
                                                
                                                {context}
                                                """)
    groq_api_key = os.getenv("GROQ_API_KEY")                                           
    llm = ChatGroq(groq_api_key=groq_api_key, model="llama-3.2-3b-preview")
    retriever_chain = prompt_template | llm
    response = retriever_chain.invoke({"question": question,"context":context})
    # Here, replace with your LLM API query logic (e.g., Groq API)
    # For demonstration, we'll just return the retrieved context
    return response.content

# Gradio interface
with gr.Blocks() as app:
    gr.Markdown("# PDF QA App with Gradio")
    with gr.Row():
        with gr.Column():
            upload_btn = gr.File(label="Upload PDF", type="filepath")
            process_btn = gr.Button("Process PDF")
            process_status = gr.Textbox(label="Processing Status", interactive=False)
        with gr.Column():
            question_input = gr.Textbox(label="Ask a Question")
            ask_btn = gr.Button("Get Answer")
            answer_output = gr.Textbox(label="Answer", interactive=False)
    
    # Button actions
    process_btn.click(process_pdf, inputs=[upload_btn], outputs=[process_status])
    ask_btn.click(answer_question, inputs=[question_input], outputs=[answer_output])

# Launch the app
app.launch()
