import os
from dotenv import load_dotenv
from pinecone import Pinecone as PineconeClient, ServerlessSpec
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pinecone import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain.llms.openai import OpenAI

load_dotenv()

pc = PineconeClient(api_key=os.getenv("PINECONE_API_KEY"))

if "rag-basics" not in pc.list_indexes().names():
    pc.create_index(
        name="rag-basics", 
        dimension=1536, 
        metric='cosine',
        spec=ServerlessSpec(
            cloud='aws',
            region=os.getenv("PINECONE_ENV")
        )
    )

index = pc.Index("rag-basics")

embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone vector store
vector_store = Pinecone.from_existing_index(
    index_name="rag-basics",
    embedding=embeddings,
    namespace="rag-namespace"
)

# Initialize OpenAI LLM
llm = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

qa_chain = load_qa_chain(llm, chain_type="stuff")

def retrieve_and_answer(query):
    docs = vector_store.similarity_search(query)

    answer = qa_chain.run(input_documents=docs, question=query)
    
    return answer

# Example query
query = "What are the instructions for this week's challenge?"
answer = retrieve_and_answer(query)
print("Answer:", answer)