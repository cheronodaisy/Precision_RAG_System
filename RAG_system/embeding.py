from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from pinecone import Pinecone as PineconeClient, ServerlessSpec
import os

# Constants
PINECONE_INDEX_NAME = "rag-basics"
PINECONE_NAME_SPACE = "rag-namespace"
FILE_PATH = 'RAG_system/sample-data.txt'

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def run():
    try:
        # Load raw text data from the text file
        raw_text = load_text_file(FILE_PATH)
        
        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_text(raw_text)
        print('split docs:', docs)

        print('creating vector store...')
        embeddings = OpenAIEmbeddings()

        pinecone_api_key = os.getenv('PINECONE_API_KEY')

        # Initialize Pinecone client
        pc = PineconeClient(api_key=pinecone_api_key)
        
        if PINECONE_INDEX_NAME not in pc.list_indexes().names():
            pc.create_index(
                name=PINECONE_INDEX_NAME, 
                dimension=1536, 
                metric='cosine',
                spec=ServerlessSpec(
                    cloud='aws',
                    region='us-east-1'
                )
            )

        index = pc.Index(PINECONE_INDEX_NAME)

        # Embed the text documents and store them in Pinecone
        vector_store = Pinecone.from_texts(texts=docs, embedding=embeddings, index_name=PINECONE_INDEX_NAME, namespace=PINECONE_NAME_SPACE)
        
    except Exception as error:
        print('error:', error)
        raise Exception('Failed to ingest your data')

if __name__ == "__main__":
    run()
    print('ingestion complete')
