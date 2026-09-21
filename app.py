import os
import uuid
import requests
import chromadb

from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


app = Flask(__name__)


# =========================
# FOLDERS
# =========================

UPLOAD_FOLDER = "documents"
CHROMA_FOLDER = "chroma_db"


os.makedirs(
    UPLOAD_FOLDER,
    exist_ok=True
)

os.makedirs(
    CHROMA_FOLDER,
    exist_ok=True
)


# =========================
# ENVIRONMENT VARIABLES
# =========================

OLLAMA_URL = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434"
)

OLLAMA_MODEL = os.getenv(
    "OLLAMA_MODEL",
    "llama3.2"
)

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "all-MiniLM-L6-v2"
)


# =========================
# EMBEDDING MODEL
# LAZY LOADING
# =========================

embedding_model = None


def get_embedding_model():

    global embedding_model

    if embedding_model is None:

        print("Loading embedding model...")

        embedding_model = SentenceTransformer(
            EMBEDDING_MODEL
        )

        print("Embedding model loaded.")

    return embedding_model


# =========================
# CHROMA DATABASE
# =========================

chroma_client = chromadb.PersistentClient(
    path=CHROMA_FOLDER
)


collection = chroma_client.get_or_create_collection(
    name="documind_documents"
)


# =========================
# TEXT SPLITTER
# =========================

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)


# =========================
# PDF EXTRACTION
# =========================

def extract_pdf(file_path):

    pages = []

    reader = PdfReader(
        file_path
    )

    for page_number, page in enumerate(
        reader.pages,
        start=1
    ):

        text = page.extract_text()

        if text:

            pages.append(
                {
                    "text": text,
                    "page": page_number
                }
            )

    return pages


# =========================
# DOCX EXTRACTION
# =========================

def extract_docx(file_path):

    document = Document(
        file_path
    )

    text_parts = []

    for paragraph in document.paragraphs:

        if paragraph.text.strip():

            text_parts.append(
                paragraph.text
            )

    full_text = "\n".join(
        text_parts
    )

    return [
        {
            "text": full_text,
            "page": 1
        }
    ]


# =========================
# TXT EXTRACTION
# =========================

def extract_txt(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        text = file.read()

    return [
        {
            "text": text,
            "page": 1
        }
    ]


# =========================
# DOCUMENT EXTRACTION
# =========================

def extract_document(file_path):

    extension = os.path.splitext(
        file_path
    )[1].lower()

    if extension == ".pdf":

        return extract_pdf(
            file_path
        )

    if extension == ".docx":

        return extract_docx(
            file_path
        )

    if extension == ".txt":

        return extract_txt(
            file_path
        )

    return []


# =========================
# CREATE CHUNKS
# =========================

def create_chunks(
    pages,
    filename
):

    chunks = []

    for page_data in pages:

        text = page_data["text"]

        page_number = page_data["page"]

        text_chunks = text_splitter.split_text(
            text
        )

        for chunk in text_chunks:

            chunks.append(
                {
                    "text": chunk,
                    "source": filename,
                    "page": page_number
                }
            )

    return chunks


# =========================
# STORE DOCUMENT
# =========================

def store_document(chunks):

    if not chunks:

        return 0

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    # Load embedding model only when needed
    model = get_embedding_model()

    embeddings = model.encode(
        texts,
        batch_size=8,
        show_progress_bar=False
    ).tolist()

    ids = [
        str(uuid.uuid4())
        for _ in chunks
    ]

    metadatas = [
        {
            "source": chunk["source"],
            "page": str(chunk["page"])
        }
        for chunk in chunks
    ]

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas
    )

    return len(chunks)


# =========================
# RETRIEVE DOCUMENTS
# =========================

def retrieve_documents(
    question,
    top_k=4
):

    # Check whether documents exist
    collection_count = collection.count()

    if collection_count == 0:

        return []

    # Avoid requesting more results than available
    actual_top_k = min(
        top_k,
        collection_count
    )

    # Load embedding model only when needed
    model = get_embedding_model()

    question_embedding = model.encode(
        question,
        show_progress_bar=False
    ).tolist()

    results = collection.query(
        query_embeddings=[
            question_embedding
        ],
        n_results=actual_top_k
    )

    documents = results.get(
        "documents",
        [[]]
    )[0]

    metadatas = results.get(
        "metadatas",
        [[]]
    )[0]

    retrieved = []

    for document, metadata in zip(
        documents,
        metadatas
    ):

        retrieved.append(
            {
                "text": document,
                "source": metadata.get(
                    "source",
                    "Unknown"
                ),
                "page": metadata.get(
                    "page",
                    "Unknown"
                )
            }
        )

    return retrieved


# =========================
# GENERATE ANSWER USING OLLAMA
# =========================

def generate_answer(
    question,
    context
):

    prompt = f"""
You are DocuMind AI, an intelligent document assistant.

Answer the user's question using only the provided document context.

If the answer is not available in the context, say:

"I could not find this information in the uploaded documents."

Do not invent information.

Keep the answer clear and simple.

DOCUMENT CONTEXT:

{context}

USER QUESTION:

{question}

ANSWER:
"""

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json=payload,
        timeout=180
    )

    response.raise_for_status()

    data = response.json()

    return data.get(
        "response",
        "No answer generated."
    )


# =========================
# HOME PAGE
# =========================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================
# UPLOAD DOCUMENT
# =========================

@app.route(
    "/upload",
    methods=["POST"]
)
def upload_document():

    if "file" not in request.files:

        return jsonify(
            {
                "success": False,
                "message": "No file selected."
            }
        )

    file = request.files["file"]

    if file.filename == "":

        return jsonify(
            {
                "success": False,
                "message": "Please select a file."
            }
        )

    filename = file.filename

    allowed_extensions = [
        ".pdf",
        ".docx",
        ".txt"
    ]

    extension = os.path.splitext(
        filename
    )[1].lower()

    if extension not in allowed_extensions:

        return jsonify(
            {
                "success": False,
                "message": "Only PDF, DOCX and TXT files are supported."
            }
        )

    safe_filename = os.path.basename(
        filename
    )

    file_path = os.path.join(
        UPLOAD_FOLDER,
        safe_filename
    )

    try:

        file.save(
            file_path
        )

        pages = extract_document(
            file_path
        )

        if not pages:

            return jsonify(
                {
                    "success": False,
                    "message": "No readable text found in the document."
                }
            )

        chunks = create_chunks(
            pages,
            safe_filename
        )

        chunk_count = store_document(
            chunks
        )

        return jsonify(
            {
                "success": True,
                "message": "Document uploaded successfully.",
                "filename": safe_filename,
                "chunks": chunk_count
            }
        )

    except Exception as error:

        print(
            "Upload error:",
            error
        )

        return jsonify(
            {
                "success": False,
                "message": str(error)
            }
        )


# =========================
# CHAT
# =========================

@app.route(
    "/chat",
    methods=["POST"]
)
def chat():

    data = request.get_json()

    if not data:

        return jsonify(
            {
                "success": False,
                "message": "Invalid request."
            }
        )

    question = data.get(
        "question",
        ""
    ).strip()

    if not question:

        return jsonify(
            {
                "success": False,
                "message": "Please enter a question."
            }
        )

    try:

        retrieved = retrieve_documents(
            question
        )

        if not retrieved:

            return jsonify(
                {
                    "success": True,
                    "answer": "No relevant information was found in the uploaded documents.",
                    "sources": []
                }
            )

        context_parts = []

        for item in retrieved:

            context_parts.append(
                f"Source: {item['source']}\n"
                f"Page: {item['page']}\n"
                f"Content: {item['text']}"
            )

        context = "\n\n".join(
            context_parts
        )

        answer = generate_answer(
            question,
            context
        )

        sources = []

        for item in retrieved:

            source_text = (
                f"{item['source']} - "
                f"Page {item['page']}"
            )

            if source_text not in sources:

                sources.append(
                    source_text
                )

        return jsonify(
            {
                "success": True,
                "answer": answer,
                "sources": sources
            }
        )

    except requests.exceptions.ConnectionError:

        return jsonify(
            {
                "success": False,
                "message": "Ollama is not reachable. Please check the OLLAMA_URL."
            }
        )

    except requests.exceptions.Timeout:

        return jsonify(
            {
                "success": False,
                "message": "Ollama request timed out."
            }
        )

    except Exception as error:

        print(
            "Chat error:",
            error
        )

        return jsonify(
            {
                "success": False,
                "message": str(error)
            }
        )


# =========================
# CLEAR DATABASE
# =========================

@app.route(
    "/clear",
    methods=["POST"]
)
def clear_database():

    global collection

    try:

        chroma_client.delete_collection(
            "documind_documents"
        )

        collection = chroma_client.get_or_create_collection(
            name="documind_documents"
        )

        return jsonify(
            {
                "success": True,
                "message": "Document database cleared."
            }
        )

    except Exception as error:

        return jsonify(
            {
                "success": False,
                "message": str(error)
            }
        )


# =========================
# RUN APPLICATION
# =========================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            10000
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
        debug=False
    )