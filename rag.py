import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Set up your Gemini API key (loaded securely from .env, never hardcoded)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError(
        "GEMINI_API_KEY not found. Please create a .env file in the project "
        "root with a line like: GEMINI_API_KEY=your_key_here"
    )
os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY

DB_FAISS_PATH = 'vectorstore/db_faiss'
KNOWLEDGE_BASE_DIR = 'knowledge_base'

def create_vector_db():
    """Loads local text files, chunks them, embeds them, and saves to FAISS."""
    if not os.path.exists(KNOWLEDGE_BASE_DIR):
        os.makedirs(KNOWLEDGE_BASE_DIR)
        print("Knowledge base directory created. Please add text files.")
        return None

    # Load all txt documents from the knowledge_base folder
    documents = []
    for file in os.listdir(KNOWLEDGE_BASE_DIR):
        if file.endswith('.txt'):
            filepath = os.path.join(KNOWLEDGE_BASE_DIR, file)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                documents.append({"text": text, "source": file})

    if not documents:
        print("No documents found in knowledge base.")
        return None

    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = []
    metadatas = []
    for doc in documents:
        split_texts = text_splitter.split_text(doc["text"])
        for chunk in split_texts:
            chunks.append(chunk)
            metadatas.append({"source": doc["source"]})

    # Use the active gemini-embedding-001 model
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    db = FAISS.from_texts(chunks, embeddings, metadatas=metadatas)

    os.makedirs(os.path.dirname(DB_FAISS_PATH), exist_ok=True)
    db.save_local(DB_FAISS_PATH)
    print("Vector database created and saved successfully! ✅")
    return db

def get_vector_db():
    """Loads or creates the FAISS vector database."""
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
    if os.path.exists(DB_FAISS_PATH):
        # Allow loading local files (required for FAISS)
        return FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        return create_vector_db()

def query_rag(question, predicted_class):
    """Retrieves relevant chunks and generates an answer using Gemini, filtered by the predicted tumor."""
    try:
        # Load or create the database inside the try-except to catch configuration errors
        db = get_vector_db()
        if not db:
            return "The knowledge base is empty. Please add medical texts to the 'knowledge_base' folder."

        # Filter the search to only retrieve from the predicted class's file (e.g., "pituitary.txt")
        source_filter = f"{predicted_class.lower()}.txt"

        retriever = db.as_retriever(
            search_kwargs={
                'k': 3,
                'filter': {'source': source_filter}
            }
        )
        docs = retriever.invoke(question)

        # Fallback in case the filter returns nothing (safety check)
        if not docs:
            retriever_fallback = db.as_retriever(search_kwargs={'k': 3})
            docs = retriever_fallback.invoke(question)

        context = "\n\n".join([doc.page_content for doc in docs])

        # Define a clean system prompt that includes the predicted class context
        prompt = ChatPromptTemplate.from_template("""
        You are a professional medical assistant specialized in neuro-oncology.
        The user's scan was predicted as: {predicted_class}

        Use the following retrieved context from clinical databases to answer the patient or clinician's question.
        Ensure your answer is accurate and highly relevant to the diagnosis of {predicted_class}.

        Retrieved Context:
        {context}

        Question:
        {question}

        Answer:
        """)

        # Use active gemini-2.5-flash model (higher free limits)
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        chain = prompt | llm | StrOutputParser()

        response = chain.invoke({
            "context": context,
            "question": question,
            "predicted_class": predicted_class.upper()
        })

        # Safety check: convert to string if the API returns a list of blocks
        if isinstance(response, list):
            text_blocks = []
            for block in response:
                if isinstance(block, dict) and 'text' in block:
                    text_blocks.append(block['text'])
                elif isinstance(block, str):
                    text_blocks.append(block)
            response = "\n".join(text_blocks)

        return response
    except Exception as e:
        return f"Error querying RAG system: {str(e)}"

def get_rag_report(predicted_class, confidence):
    """
    Retrieves the entire text file for the predicted class and
    generates the initial clinical report.
    """
    kb_path = f"knowledge_base/{predicted_class.lower()}.txt"
    if os.path.exists(kb_path):
        with open(kb_path, "r", encoding="utf-8") as f:
            retrieved_context = f.read()
    else:
        retrieved_context = "No specific guidelines found in the knowledge base."

    prompt = f"""
    You are a professional clinical AI assistant specializing in neuro-oncology.

    A machine learning model has analyzed a brain MRI scan and outputted the following prediction:
    - Predicted Class: {predicted_class.upper()}
    - Model Confidence: {confidence:.2f}%

    Here is the official clinical context retrieved from our medical database:
    ---
    {retrieved_context}
    ---

    Using the prediction and the retrieved medical context, generate a structured, patient-friendly report.
    Include the following sections in clean Markdown format:
    1. **Overview & Analysis**: Explain the model's finding in easy-to-understand terms.
    2. **Clinical Symptoms & Signs**: List what symptoms are commonly associated with this classification.
    3. **General Treatment Guidelines**: Outline the standard care steps according to the retrieved guidelines.
    4. **Recommended Next Steps**: Advise the patient/clinician on follow-up steps.

    *Important Disclaimer*: Always end the report with a clear medical disclaimer.
    """
    try:
        # Use active gemini-2.5-flash model
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)
        response = llm.invoke(prompt)
        content = response.content

        # Safety check: convert to string if the API returns a list of blocks
        if isinstance(content, list):
            text_blocks = []
            for block in content:
                if isinstance(block, dict) and 'text' in block:
                    text_blocks.append(block['text'])
                elif isinstance(block, str):
                    text_blocks.append(block)
            content = "\n".join(text_blocks)

        return content
    except Exception as e:
        return f"Error generating clinical report: {str(e)}"