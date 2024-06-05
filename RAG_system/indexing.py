from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import PineconeStore
import pinecone
import os

# Constants
PINECONE_INDEX_NAME = "RAG Basics"
PINECONE_NAME_SPACE = "RAG Basics"
FILE_PATH = 'rag_data.txt'

def load_text_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def run():
    try:
        # Load raw text data from the text file
        raw_text = load_text_file(FILE_PATH)
        raw_docs = [{'text': raw_text}]

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        docs = text_splitter.split_documents(raw_docs)
        print('split docs', docs)

        print('creating vector store...')
        # Create and store the embeddings in the vector store
        embeddings = OpenAIEmbeddings()

        # Access Pinecone API key and environment from environment variables
        pinecone_api_key = os.getenv('PINECONE_API_KEY')

        # Initialize Pinecone
        pinecone.init(api_key=pinecone_api_key)
        index = pinecone.Index(PINECONE_INDEX_NAME)

        # Embed the text documents and store them in Pinecone
        PineconeStore.from_documents(docs, embeddings, {
            'pineconeIndex': index,
            'namespace': PINECONE_NAME_SPACE,
            'textKey': 'text',
        })
    except Exception as error:
        print('error', error)
        raise Exception('Failed to ingest your data')

if __name__ == "__main__":
    run()
    print('ingestion complete')
