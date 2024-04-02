import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader

from haystack.document_stores import InMemoryDocumentStore
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import FARMReader
from haystack.pipelines import ExtractiveQAPipeline

# create sidebar
with st.sidebar:
    st.title("LLM Document Chat App")
    st.markdown('''
    ### About 
    This app is an LLM-powered chatbot capable of answering user queries based on information in a given PDF file. Primarily based on Haystack v1.25 and presented using Streamlit.
                
    ### Libraries
    - [Streamlit](https://streamlit.io/)
    - [Haystack](https://haystack.deepset.ai/)
                
    ### How it works:
    1. Takes a pdf
    2. Extracts text data and stores it in DocumentStore 
    3. Initialises BM25Retriever and FARMReader
    4. Connects reader & retriever with ExtractiveQAPipeline
    5. Process user query & display answer
    ''')

    add_vertical_space(9)
    st.write("Author: Muhammad Naufal Al Ghifari")


def main():
    st.header("Chat with your PDF🗣️📄")
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    if pdf is not None:
        # read the pdf
        pdf_reader = PdfReader(pdf)
        
        # extract the text from each page
        full_text = ""
        for page in pdf_reader.pages:
            full_text += page.extract_text()

        # save the text data
        with open(f"data.txt", "w", encoding="utf-8") as f:
            f.write(full_text)

        # initialise DocumentStore 
        document_store = InMemoryDocumentStore(use_bm25=True)

        # store corpus as haystack document object
        path = ["./data.txt"]
        indexing_pipeline = TextIndexingPipeline(document_store)
        indexing_pipeline.run_batch(file_paths=path)

        # initialise retriever and reader
        my_retriever = BM25Retriever(document_store=document_store)
        my_reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2", use_gpu=True)

        # connect retriever and reader using ExctractiveQAPipeline
        pipe = ExtractiveQAPipeline(my_reader, my_retriever)

        # get user query
        user_query = st.text_input("Ask a question about your PDF here:")

        if user_query:
            # process query
            prediction = pipe.run(
                query=user_query,
                params= {"Retriever":{"top_k": 10}, "Reader":{"top_k": 5}},
                debug=True
            )

            # display the answers            
            answers = prediction['answers']

            st.write(f"#### {len(answers)} answers found:\n\n")

            for i in range(len(answers)):
                ans = answers[i]
                ans_text = ans.answer
                context = ans.context
                score = ans.score

                st.write(f"**Answer {i+1}**:\n{ans_text}")
                st.write(f"**Context {i+1}**:\n{context}")
                st.markdown("""---""") 

if __name__ == "__main__":
    main()
