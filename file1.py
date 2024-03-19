import warnings
warnings.filterwarnings('ignore',category=DeprecationWarning)


from openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from pinecone import Pinecone, PodSpec
from langchain_community.embeddings import CohereEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter, TokenTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
import os, time
from langchain.docstore.document import Document
import api_tokens

def get_api_keys():

    llama_api_token = os.environ.get("LLAMA_API_KEY")
    cohere_api_token = os.environ.get("COHERE_API_KEY")
    pinecone_api_token = os.environ.get("PINECONE_API_KEY")


    return llama_api_token, cohere_api_token, pinecone_api_token


def preprocess_pdf(pdf_path = "D:\Data\Official\RAG document\pdf\LLaMA2_Paper.pdf", chunk_size =1000, chunk_overlap = 40):

    print("-"*20)
    print("READING PDF")
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
    except Exception:
        print(Exception)
    else:
        print("File Read successfully")

    print("-"*20)

    print(f"CHUNKING FILES.\n CHUNK SIZE = {chunk_size}\nChunk overlap = {chunk_overlap}")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 40)

    splitted_docs = text_splitter.split_documents(pages)

    print(f"CHUNK COMPLETE.\n TOTAL CHUNKS = {len(splitted_docs)}")
    print("-"*20)

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

def setup_pvs(splitted_docs ,embeddings, index_name = "rag-pinecone-test-101", dimension = 384):
    print("Setting up Pinecone VectorDataStore ")

    pinecone = Pinecone(api_key = pinecone_api_key)

    index_name = index_name

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
    index = pinecone.Index(index_name)
    print(" EMBEDDING AND INSERTING")

    pvs = PineconeVectorStore.from_documents(splitted_docs, embeddings, index_name=index_name)

    print("VECTORSTORE SETUP DONE")
    # print("REFRESHING")
    index = pinecone.Index(index_name)
    index_stats = index.describe_index_stats()
    # print(f"Index name: {index_name} \nDimension: {index_stats.dimension}\nTotal_vector_count:{index_stats.total_vector_count}")

    print("Added Vectors into Pinecone")
    print("-"*20)

    return pvs

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

    return qa_model

def summarize_text(query,qa_model):
    prompt1 = f"Summarize the give text delimitted between single quotes efficiently  '{query}'  Summary:"

    print("USING CREDITS for summarizing!!!! ")
    output = qa_model.invoke(prompt1)

    return output


def query_model(query, qa_model):
    prompt1 = f"""
    You are a information extractor. You retreive information from the given text based on the user's query.

    If the context is not available STRICTLY return "Context not Available".
    If the context is available you retreive appropriate information.

    QUESTION: {query}

    Answer:
    """

    print("WARNING! -QUERYING USES CREDITS for querying")

    output = qa_model.invoke(prompt1)['result']

    return output

def setup_summ_model(docs, model_name = "llama-7b-chat",temperature = 0.0, max_tokens = 2000):

 

    prompt_template = """Write a concise summary of the following with atmost 300 words.

    Remove Any Personal Identifiable Information from the summary such as mobile no, email etc...

    Names are an exception.
    Text:
    {text}
    CONCISE SUMMARY:"""

    docs = docs[0:5]
    llm = ChatOpenAI(
        openai_api_key = llama_api_key,
        base_url= llama_base_url,
        model_name = model_name,
        temperature=temperature,
        max_tokens= max_tokens
        )


    prompt = PromptTemplate.from_template(prompt_template)
    # print(prompt)

    chain = load_summarize_chain(llm, chain_type="refine", question_prompt = prompt,input_key="input_documents",
        output_key="output_text")
    
    print("WARNING! - USING CREDITS")

    result = chain({"input_documents": docs}, return_only_outputs=True)
    output = result["output_text"]
    # print(output)
    return output
