from openai import OpenAI

from langchain_community.document_loaders import PyPDFLoader
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import CohereEmbeddings
from pinecone import Pinecone, PodSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)
import os
import time



def get_api_keys():

    """
    Collects Api keys from envirionment variables
    """

    try:
        llama_api_token = os.environ.get("LLAMA_API_KEY")
        cohere_api_token = os.environ.get("COHERE_API_KEY")
        pinecone_api_token = os.environ.get("PINECONE_API_KEY")
    except:
        print("Please add API keys to environment variables.")

    return llama_api_token, cohere_api_token, pinecone_api_token


def read_pdf(pdf_path):
    print("READING PDF")
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
    except Exception:
        print(Exception)
    else:
        print("File Read successfully")
    
    return pages

def chunk_files(pages,chunk_size,chunk_overlap):
    """
    
    Chunks pages into smaller documents, using recurvise character text splitter.
    returns splitted documents
    """

    print(f"CHUNKING FILES.\nCHUNK SIZE = {chunk_size}\nChunk overlap = {chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 40)

    splitted_docs = text_splitter.split_documents(pages)

    print(f"\nTOTAL CHUNKS = {len(splitted_docs)}")
    
    return splitted_docs

def preprocess_pdf(pdf_path, chunk_size =1000, chunk_overlap = 40):

    print("-"*20)
   
    pages = read_pdf(pdf_path)

    print("-"*20)

    splitted_docs = chunk_files(pages, chunk_size, chunk_overlap)

    return splitted_docs


llama_api_key, cohere_api_key, pinecone_api_key = get_api_keys()
llama_base_url = "https://api.llama-api.com"


def setup_embedding(embed_model = "embed-english-light-v3.0"):

    print("Setting up Embedding model")

    embed_model = embed_model

    embeddings = CohereEmbeddings(model = embed_model, cohere_api_key = cohere_api_key )

    print(f"Setup Successful. Model Name = {embed_model}")
    print("-"*20)
    
    dimension = 384 # HARDCODING OUTPUT DIMENSION OF MODEL : "embed-english-light-v3.0

    return embeddings


def create_index(pinecone,index_name,dimension):

    if index_name in [index.name for index in pinecone.list_indexes()]:
        pinecone.delete_index(index_name)

    try:
        pinecone.create_index(name = index_name, 
                                dimension = dimension ,
                                metric = 'cosine', 
                                spec = PodSpec(environment="gcp-starter"))
    except:
        print("INDEX CREATION FAILED")
    else:
        print(f"Index '{index_name}' created successfully")
    
    index = pinecone.Index(index_name)

    index_stats = index.describe_index_stats()

    # print(index_stats)

    print(f"Index name: {index_name} \nDimension: {index_stats.dimension}\n")

    print("-" *20)

def insert_vectors(splitted_docs, embeddings, index_name):

    print("EMBEDDING AND INSERTING")

    pvs = PineconeVectorStore.from_documents(splitted_docs, embeddings, index_name=index_name)

    print("VECTORSTORE SETUP DONE")

    print("Added Vectors into Pinecone")
    print("-"*20)

    return pvs


def setup_pvs(splitted_docs ,embeddings, index_name = "rag-pinecone-test-101", dimension = 384):


    print("Setting up Pinecone VectorDataStore ")

    pinecone = Pinecone(api_key = pinecone_api_key)

    index_name = index_name

    create_index(pinecone,index_name,dimension)

    pvs = insert_vectors(splitted_docs, embeddings, index_name)

    return pvs


def create_chat_model(model_name = "mistral-7b-instruct" ,temperature = 0.0, max_tokens = 2000):

    llm = ChatOpenAI(
    openai_api_key = llama_api_key,
    base_url= llama_base_url,
    model_name = model_name,
    temperature=temperature,
    max_tokens= max_tokens
    )


    return llm



def summarize_text(query,llm):
    # prompt1 = f"Summarize the give text delimitted between single quotes efficiently  '{query}'  Summary:"

    # print("USING CREDITS for summarizing!!!! ")

    template = (" You are a helpful assistant excelled in text summarization. Summarize the given text")
    
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    
    human_template = "{text}"
    
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)

    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    response = llm.invoke(chat_prompt.format_prompt(text = query).to_messages())

    return response.content

def setup_rag(pvs,model_name = "mistral-7b-instruct" ,temperature = 0.0, max_tokens = 2000):
    
    llm = ChatOpenAI(
    openai_api_key = llama_api_key,
    base_url= llama_base_url,
    model_name = model_name,
    temperature=temperature,
    max_tokens= max_tokens
    )

    retreiver = pvs.as_retriever()

    qa_model = RetrievalQA.from_chain_type( 
        llm = llm,
        chain_type = "stuff",
        retriever = retreiver

    )

    return llm, qa_model



def ask_model(query, qa_model):


    prompt = f"""Answer the user's question based on following rules.

    Rule1: If you can answer the question with the available context, answers the user's question, you answer the quesstion.
    Rule2: If you can not answer the quesstion with the availble context, then say Context not available.

    Follow the rules and Answer the question below.
    Question:{query}
    Answer:
    """

    print("utilizing credits")

    output = qa_model.invoke(prompt)['result']

    return output



def setup_summ_model(docs, model_name = "llama-7b-chat",temperature = 0.0, max_tokens = 2000):

 

    prompt_template = """Given the contents of a Pdf, Write a concise summary with a suitable heading with atmost 300 words.
    
    Text:
    {text}
    CONCISE SUMMARY:"""
    

    docs = docs
    
    llm = create_chat_model()

    prompt = PromptTemplate.from_template(prompt_template)
    # print(prompt)

    chain = load_summarize_chain(llm, chain_type="refine", question_prompt = prompt,input_key="input_documents",
        output_key="output_text")
    
    print("WARNING! - USING CREDITS")

    result = chain({"input_documents": docs}, return_only_outputs=True)
    output = result["output_text"]
    # print(output)
    return output


