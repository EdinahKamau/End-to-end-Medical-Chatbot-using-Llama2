from src.helper import load_pdf, text_split, download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from environment variables
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# Debugging: Print API key to ensure it is loaded correctly
# print(PINECONE_API_KEY)

# Load PDF data
extracted_data = load_pdf("data/")
text_chunks = text_split(extracted_data)

# Download Hugging Face embeddings
embeddings = download_hugging_face_embeddings()

# Initialize Pinecone with API key
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-bot"

# Ensure the index exists
if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=embeddings.embed_dim,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-west-2')
    )

# Connect to the index
index = pc.Index(index_name)

# Create embeddings for each of the text chunks & store in Pinecone
docsearch = index.upsert(
    vectors=[
        {"id": "vec1", "values": [0.1] * 384, "metadata": {"genre": "drama"}},
        {"id": "vec2", "values": [0.2] * 384, "metadata": {"genre": "action"}},
        {"id": "vec3", "values": [0.3] * 384, "metadata": {"genre": "drama"}},
        {"id": "vec4", "values": [0.4] * 384, "metadata": {"genre": "action"}}
    ],
    namespace="ns1"
)

# Query the index
response = index.query(
    namespace="ns1",
    vector=[0.3] * 384,
    top_k=2,
    include_values=True,
    include_metadata=True,
    filter={"genre": {"$eq": "action"}}
)

print(response)
