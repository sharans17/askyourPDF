
import os 

import streamlit as st

from streamlit_lottie import st_lottie, st_lottie_spinner

import api_tokens

from Main import preprocess_pdf, summarize_text, setup_rag,setup_embedding,setup_pvs, setup_summ_model, ask_model

import time

          

def save_file(uploaded_file):
        path = os.path.join(temp_dir, uploaded_file.name)
       
        with open(path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        print("FILE SAVED HERE", path)

        return path

def file_setup(path):

    with st.spinner("Loading"):
        splitted_docs = preprocess_pdf(path)
    
    with st.spinner("Still loading..."):
        embeddings = setup_embedding(embed_model="embed-english-light-v3.0")

    with st.spinner("Drink some water"):

        pvs = setup_pvs(splitted_docs = splitted_docs,
                            index_name = "rag-pinecone-test-101",
                            embeddings = embeddings,
                            dimension = 384)
        
    with st.spinner("Almost done..."):


        llm, model = setup_rag(pvs = pvs, model_name = "mistral-7b-instruct")

        st.session_state["qa_model"] = model

    count = llm.get_num_tokens(" ".join([i.page_content for i in splitted_docs]))

    decrease = -1
    print("Count:", count)
    if count<=1200:
        summary_docs = splitted_docs
    else:
        with st.spinner("CUTTING DOWN DOCUMENTS"):
            while count > 1200:
                print(count)
                summary_docs = splitted_docs[:decrease]
                count = llm.get_num_tokens(" ".join([i.page_content for i in summary_docs]))
                decrease-= 1

    st.write("Yep, Ready")
    time.sleep(2)

    return model, summary_docs, llm

temp_dir = r"D:\Data\Official\Ask you Pdf\Version 1\Uploaded_pdfs"

st.title("ASK YOUR PDF! ")
st.write("Why read pdfs, when you can directly ask them 😲")

st.divider()

holder = st.empty()

if "file" not in st.session_state:
     st.session_state.file = False

if not st.session_state.file:
    with holder:
        with st.form("Form1",):
            
            file_upload = st.file_uploader("Drop your PDF here", type = "pdf", accept_multiple_files= False)
            
            submit = st.form_submit_button("Submit")

            if submit:
                path = save_file(file_upload)
                st.write("File Saved at : ",path)
                
                st.session_state["file"] = True
                st.session_state["path"] = path            

if st.session_state.file:
    holder.empty()

    with holder:
            
        path = st.session_state.path
        if 'read' not in st.session_state:
            st.session_state.read = False

        if not st.session_state.read:
            
            model, summary_docs, llm = file_setup(path)
            st.session_state.model = model
            st.session_state.summary_docs = summary_docs
            st.session_state.llm = llm
            st.session_state.read = True
            st.write("read done")

    if st.session_state.read:
        holder.empty()

        summary_docs = st.session_state.summary_docs
        model = st.session_state.model

        llm = st.session_state.llm

        with holder: 
            qa, sum, sum_text= st.tabs(["Question and Answering", "Summarize document","Summarize text"])

            with qa:

                st.subheader("Ask your question")

                query = st.text_input("Question goes here!",)

                button1 = st.button("Ask")

                if button1: 
                    st.session_state['ques'] = query 

                    with st.spinner("Thinking"):
                        output = ask_model(query, model)
                    st.success(output)
            
            with sum:

                st.subheader("Summarize Document")

                button2 = st.button("Summarize document")

                if button2:

                    with st.spinner("Summarizing"):
                        output = setup_summ_model(docs = summary_docs , model_name = "mistral-7b-instruct")
                    st.write("Summary of the document")
                    st.success(output)

            with sum_text:

                st.subheader("summarize Text")


                with st.form("form2"):
                    text1 = st.text_area("Enter text")

                    button10 = st.form_submit_button("Summarize it")
                    if button10:
                        with st.spinner("Summarizing text"):
                            output = summarize_text(text1, llm)
                            st.success(output)           
                            