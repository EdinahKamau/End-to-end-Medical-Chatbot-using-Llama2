import os
from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve Pinecone API key from environment variables
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

# Initialize Pinecone without API key
pc = Pinecone()

# Set API key
pc.set_api_key(api_key=PINECONE_API_KEY)

# Check if index exists, create if it doesn't
index_name = "medical-bot-index"  # Adjust index name as needed
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region='us-west-2'
        )
    )

# Initialize Flask application
app = Flask(__name__)

# Download Hugging Face embeddings if needed
embeddings = download_hugging_face_embeddings()

# Define PromptTemplate, CTransformers, and RetrievalQA instances
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

llm = CTransformers(
    model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
    model_type="llama",
    config={'max_new_tokens': 512, 'temperature': 0.8}
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",  # Adjust chain type as needed
    retriever=pc.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs
)

# Define routes
@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    result = qa({"query": msg})
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="127.0.0.1", port=5000, debug=True)
