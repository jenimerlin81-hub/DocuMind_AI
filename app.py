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

UPLOAD_FOLDER = "documents"
CHROMA_FOLDER = "chroma_db"

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


os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(CHROMA_FOLDER, exist_ok=True)


print("Loading embedding model...")

embedding_model = SentenceTransformer(
    EMBEDDING_MODEL
)

print("Embedding model loaded.")


chroma_client = chromadb.PersistentClient(
    path=CHROMA_FOLDER
)


collection = chroma_client.get_or_create_collection(
    name="documind_documents"
)


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150
)


def extract_pdf(file_path):
    pages = []

    reader = PdfReader(file_path)

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if text:
            pages.append(
                {
                    "text": text,
                    "page": page_number
                }
            )

    return pages


def extract_docx(file_path):
    document = Document(file_path)

    text_parts = []

    for paragraph in document.paragraphs:
        if paragraph.text.strip():
            text_parts.append(paragraph.text)

    full_text = "\n".join(text_parts)

    return [
        {
            "text": full_text,
            "page": 1
        }
    ]


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


def extract_document(file_path):
    extension = os.path.splitext(
        file_path
    )[1].lower()

    if extension == ".pdf":
        return extract_pdf(file_path)

    if extension == ".docx":
        return extract_docx(file_path)

    if extension == ".txt":
        return extract_txt(file_path)

    return []


def create_chunks(pages, filename):
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


def store_document(chunks):
    if not chunks:
        return 0

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts
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


def retrieve_documents(question, top_k=4):
    question_embedding = embedding_model.encode(
        question
    ).tolist()

    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k
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


def generate_answer(question, context):
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


@app.route("/")
def home():
    return render_template(
        "index.html"
    )


@app.route("/upload", methods=["POST"])
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

    file.save(file_path)

    try:
        pages = extract_document(
            file_path
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
        return jsonify(
            {
                "success": False,
                "message": str(error)
            }
        )


@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()

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
                "message": "Ollama is not running."
            }
        )

    except Exception as error:
        return jsonify(
            {
                "success": False,
                "message": str(error)
            }
        )


@app.route("/clear", methods=["POST"])
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


if __name__ == "__main__":
    app.run(
        debug=True
    )