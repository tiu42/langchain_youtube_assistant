from langchain_community.document_loaders import YoutubeLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv()

embeddings = OpenAIEmbeddings()

def create_vector_db(video_url: str) -> FAISS:
    loader = YoutubeLoader.from_youtube_url(video_url)
    transcript = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_documents(transcript)
    db = FAISS.from_documents(docs,embeddings)
    return db

def get_response_from_query(db, query, k=4):
    docs = db.similarity_search(query,k=k)
    docs_page_content = " ".join([d.page_content for d in docs])
    llm = OpenAI()
    prompt = PromptTemplate(
        input_variables = ["question","docs"],
        template = """
        You are a helpful assistant that that can answer questions about youtube videos 
        based on the video's transcript.
        
        Answer the following question: {question}
        By searching the following video transcript: {docs}
        
        Only use the factual information from the transcript to answer the question.
        
        If you feel like you don't have enough information to answer the question, say "I don't know".
        
        Your answers should be verbose and detailed.
        """,
    )
    chain = LLMChain(llm=llm,prompt=prompt,output_key="answer")

    response = chain.invoke({'question': query, 'docs': docs_page_content})
    response['answer'] = response['answer'].replace("\n","")
    return response['answer'],docs