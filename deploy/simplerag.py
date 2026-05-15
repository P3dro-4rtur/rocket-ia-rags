#!/usr/bin/env python
# coding: utf-8

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableSequence
from langchain_chroma import Chroma
import os
import json


OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

embedding_model = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-3.5-turbo", max_tokens=200)


def loadData():
    pdf_link = os.path.join(
        os.environ.get("LAMBDA_TASK_ROOT", "."), "anexo-projeto-de-lei.pdf"
    )

    loader = PyPDFLoader(pdf_link, extract_images=False)

    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=20,
        length_function=len,
        add_start_index=True,
    )

    chunks = text_splitter.split_documents(pages)

    Chroma.from_documents(
        chunks, embedding=embedding_model, persist_directory="/tmp/text_index"
    )

    vectordb = Chroma(
        persist_directory="/tmp/text_index", embedding_function=embedding_model
    )

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})
    return retriever


# Inicializado uma vez na inicialização do container
retriever = loadData()


def getRelevantDocs(question):
    context = retriever.invoke(question)
    return context


def ask(question):
    TEMPLATE = """
    Você é um especialista em legislação e tecnologia. Responda a pergunta abaixo utilizando o contexto informado:

    Contexto: {context}

    Pergunta: {question}
    """

    prompt = PromptTemplate(input_variables=["context", "question"], template=TEMPLATE)
    sequence = RunnableSequence(prompt | llm)
    context = getRelevantDocs(question)
    response = sequence.invoke({"context": context, "question": question})
    return response


def lambda_handler(event, context):
    body = json.loads(event.get("body", {}))

    query = body.get("question")

    response = ask(query).content

    return {
        "status_code": 200,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps({"message": "Tarefa Concluída", "details": response}),
    }
