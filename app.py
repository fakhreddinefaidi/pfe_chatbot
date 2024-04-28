import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub
from langchain.callbacks import get_openai_callback
from langchain.text_splitter import SpacyTextSplitter
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage


def get_pdf_text(pdf_docs):
    text = ""
    for pdf_file in pdf_docs:
        pdf_reader = PdfReader(pdf_file)
        for page in pdf_reader.pages:
            text += page.extract_text()

    # Replace newline characters with spaces
    text = text.replace("\n", " ")

 

    return text


def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore


# prompt = "Vous êtes un assistant Monoprix virtuel. Vous avez accès à un vaste ensemble de données sur les produits et services de Monoprix. Votre objectif principal est de fournir des réponses utiles et positives aux clients. Vous devez être amical, poli et serviable dans toutes vos interactions. Vous pouvez utiliser les informations du dataset pour fournir des détails sur les produits, les promotions, les horaires d'ouverture, les services offerts, etc. Assurez-vous de toujours donner une réponse qui encourage le client et le guide vers une solution satisfaisante. Évitez les réponses négatives ou décourageantes\n\n si tu ne trouve pas la réponse il faut guider le client à contacter le service client Monoprix ou le rediriger vers le site web www.monoprix.tn"

general_system_template = r""" 
As an assistant for our clients, you're here to make sure they're dressed to impress for any occasion! Whether it's a casual outing, a  professional meeting, or a lively party, you've got them covered. Just let them know the event and the weather forecast, and you'll tailor some stylish suggestions to suit both the vibe and the climate. Let's make sure they shine wherever they go.you 're going to answer the client with the given language 
 ----
{context}
----
"""
general_user_template = "Question:```{question}```"
messages = [
    SystemMessagePromptTemplate.from_template(general_system_template),
    HumanMessagePromptTemplate.from_template(general_user_template),
]
qa_prompt = ChatPromptTemplate.from_messages(messages)


def get_conversation_chain(vectorstore):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(
            search_type="similarity",
        ),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        verbose=True,
    )
    return conversation_chain


def handle_userinput(user_question):
    with get_openai_callback() as cb:
        response = st.session_state.conversation({"question": user_question})
        print(cb)
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")

    user_question = st.chat_input("Ask a question about your documents:")
    
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                text_chunks = get_text_chunks(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)


if __name__ == "__main__":
    main()
