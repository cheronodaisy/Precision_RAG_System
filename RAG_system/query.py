import os
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.openai import OpenAI

# Load environment variables from .env file
load_dotenv()

# Initialize Pinecone client
pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))

# Create the Pinecone index if it doesn't exist
if "rag-basics" not in pc.list_indexes().names():
    pc.create_index(
        name="rag-basics", 
        dimension=1536, 
        metric='cosine',  # You can change the metric if needed
        spec=ServerlessSpec(
            cloud='aws',
            region=os.getenv("PINECONE_ENV")
        )
    )

# Initialize Pinecone index
index = pc.Index("rag-basics")

# Initialize OpenAI embeddings
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone vector store
vector_store = Pinecone.from_existing_index(
    index_name="rag-basics",
    embedding=embeddings,
    namespace="rag-namespace"
)

# Initialize OpenAI LLM
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load QA chain
qa_chain = load_qa_chain(llm, chain_type="stuff")

def retrieve_and_answer(query):
    # Retrieve documents from the vector store
    docs = vector_store.similarity_search(query)

    # Generate an answer using the QA chain
    answer = qa_chain.run(input_documents=docs, question=query)
    
    return answer

# Example query
query = "What are the instructions for this week's challenge?"
answer = retrieve_and_answer(query)
print("Answer:", answer)