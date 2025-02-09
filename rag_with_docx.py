from langchain_community.document_loaders import Docx2txtLoader
from langchain.document_loaders import DirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_core.callbacks import StdOutCallbackHandler


DB_NAME = "vector_db"   
DOCUMENTS_PATH = "documents"
TEXT_SPLITTER_CHUNK_SIZE = 100
TEXT_SPLITTER_CHUNK_OVERLAP = 30
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_NAME = 'llama3.2'
LLAMA_ENDPOINT = 'http://localhost:11434/v1'
API_KEY = 'llama'

documents = []

loader = DirectoryLoader(DOCUMENTS_PATH, glob="**/*.docx", loader_cls=Docx2txtLoader)
folder_docs = loader.load()
documents.extend(folder_docs)

text_splitter = CharacterTextSplitter(
    chunk_size=TEXT_SPLITTER_CHUNK_SIZE,
    chunk_overlap=TEXT_SPLITTER_CHUNK_OVERLAP
)

chunks = text_splitter.split_documents(documents)

print(f"Total number of chunks: {len(chunks)}")

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)

vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=DB_NAME)
print(f"Vectorstore created with {vectorstore._collection.count()} documents")

llm = ChatOpenAI(temperature=0.7, model_name=MODEL_NAME, base_url=LLAMA_ENDPOINT, api_key=API_KEY)


memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
retriever = vectorstore.as_retriever()

conversation_chain = ConversationalRetrievalChain.from_llm(llm=llm, retriever=retriever, memory=memory, callbacks=[StdOutCallbackHandler()])

query = "Your question here"
result = conversation_chain.invoke({"question": query})
print(result["answer"])