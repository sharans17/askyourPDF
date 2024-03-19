import streamlit as st

import os 

import tempfile

from streamlit_lottie import st_lottie,st_lottie_spinner


from file1 import preprocess_pdf, summarize_text, setup_rag,setup_embedding,setup_pvs, setup_summ_model, query_model

import time


temp_dir =r"D:\Data\Official\RAG document\Uploaded_documents"

def change_file():
    for i in st.session_state.keys():
        del st.session_state[i]


# print(change_file())
def save_file(uploaded_file):
        path = os.path.join(temp_dir, uploaded_file.name)
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
        print("FILE SAVED HERE", path)

        return path

def file_setup(path):

    splitted_docs = preprocess_pdf(path)

    summary_docs = splitted_docs[:5]

    embeddings = setup_embedding(embed_model="embed-english-light-v3.0")

    pvs = setup_pvs(splitted_docs = splitted_docs,
                        index_name = "rag-pinecone-test-101",
                        embeddings = embeddings,
                        dimension = 384)
                
    model = setup_rag(pvs = pvs, model_name = "mistral-7b-instruct")
    
    st.session_state['truncated_docs'] = summary_docs
    st.session_state['model'] = model

    return summary_docs, model



st.title("ASK YOUR PDF! ")
st.divider()

# st.balloons()
# time.sleep(1)

if 'file_saved' not in st.session_state:
     st.session_state['file_saved'] = False
     
file_saved = st.session_state['file_saved']

file_upload = st.file_uploader("Drop your files", type = "pdf",disabled = file_saved)

if (st.button("Change file")):
    change_file()


if file_upload is None:
    file_saved = False
if (file_saved):

    summary_docs = st.session_state['truncated_docs']
    model = st.session_state['model']

    qa, sum = st.tabs(["Question and Answering", "Summary"])

    with qa:

        
        st.subheader("Ask your question")

        query = st.text_input("Question goes here!",)


        button1 = st.button("Ask")

        if button1: 
            st.session_state['ques'] = query 

            with st.spinner("Thinking"):
                output = query_model(query, model)
            st.write(output)

    with sum:
        st.subheader("Lets Summarize")
        if 'sum_text' not in st.session_state:
            st.session_state['sum_text'] = False

        sum_text = st.session_state['sum_text']

        if not sum_text:
        # button3 = st.button("Summarize Text")
            col1, col2 = st.columns(2)
            with col1:
                button4 = st.button("Summarize Document")
            
            with col2:
                button5 = st.button("Summarize text instead")

            if button5:
                st.session_state['sum_text'] = True
                sum_text = True

            if button4:
                with st.spinner("Summarizing"):
                    output = setup_summ_model(docs = summary_docs , model_name = "mistral-7b-instruct")
                st.write("Summary of the document")
                st.write(output)
            

        if sum_text:

            text1 = st.text_input("Enter the text here")

            button6 = st.button("Summarize it")

            if button6:
                with st.spinner("Summarizing text"):
                    output = summarize_text(text1, model)['result']
                    st.write(output)
                
                st.session_state['sum_text'] = False
            

else:
     
     if file_upload is None:
          pass
     else:
        path = save_file(file_upload)
        
        lottie1_url = "https://lottie.host/4070a451-5b71-40fe-bb6d-c0d90a78d01a/9XMNiFOBbX.json"
        # with st.spinner("Reading"):

        col5, col6 = st.columns(2)
        boo1 = True
        with col5:
            with st_lottie_spinner(lottie1_url, height =150, width = 150):
                with st_lottie_spinner("https://lottie.host/b186b520-b63e-4dd5-92ae-2a8866b75cde/SkSPgKZdd2.json"):
                    summary_docs, model = file_setup(path)
                    boo1 = False
            
        file_saved = True
        st.session_state['file_saved'] = True
        st.rerun()
