import streamlit as st
import langchain_helper as lch
import textwrap

st.title("Youtube Assistant")

with st.sidebar:
    with st.form(key='myform'):
        youtube_url = st.sidebar.text_area(
            label= "What is the video's url",
            max_chars=50
        )
        query = st.sidebar.text_area(
            label= "Ask me about the video",
            max_chars= 100,
            key= "query"
        )
        submit_button = st.form_submit_button(label="Submit")

if query and youtube_url:
    db = lch.create_vector_db(youtube_url)
    response, docs = lch.get_response_from_query(db, query)
    st.subheader("Answer")
    st.text(textwrap.fill(response, width= 80))