# chatIKE.py

import os
import platform

import openai
import chromadb
import langchain

from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.chains import ConversationalRetrievalChain
from langchain.document_loaders import TextLoader, DirectoryLoader, GutenbergLoader
from llm import load_local_files
from consts import OPENROUTER_API_BASE # Import OPENROUTER_API_BASE

query_list = {
    "IKE_States" : "As a IKE protocol specialist, your task is to extract IKE protocol states and how many IKE protocol states in these code snippets of IKE implementation?  Represent them with json.",
    "IKE_Pathrule" : "As a IKE protocol specialist, your task is to extract the IKE protocol state from the IKE implementation code and the constraints on the packets that can be accepted in different states. These constraints include the type and amount of payload in the IKE packet, and whether the payload is encrypted. Represent them with json.",
}

model_list = {
    "gpt-3.5-turbo" : "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k" : "gpt-3.5-turbo-16k",
    "gpt-4" : "gpt-4",
    "gpt-4-1106-preview" : "gpt-4-1106-preview", # Added from original file
}

def chatGPT():
    # Set the API base for openai library
    openai.api_base = OPENROUTER_API_BASE 
    
    llm = OpenAI(openai_api_key="", openai_api_base=OPENROUTER_API_BASE) # Set API base for OpenAI LLM
    chat_model = ChatOpenAI(model="gpt-3.5-turbo", openai_api_base=OPENROUTER_API_BASE) # Set API base for ChatOpenAI

    # 加载文件
    codeLoader = TextLoader(file_path="/home/why/sec_sea/protocols/ike/strongswan/src/libcharon/encoding/message.c")
    # codeLoader = TextLoader(file_path="/home/why/sec_sea/protocols/ike/strongswan/src/libcharon/sa/ike_sa.c")
    # codeLoader = TextLoader(file_path="/home/why/sec_sea/llm/hello.c")
    codeDoc = codeLoader.load()
    
    # print(codeDoc[0].page_content)
    
    # 分割代码
    cppSpliter = RecursiveCharacterTextSplitter.from_language(language=Language.CPP, chunk_size=2000, chunk_overlap=100)
    cDocs = cppSpliter.split_documents(codeDoc)
    print(len(cDocs))
    for i in cDocs:
        print(i.page_content)
        print("-------------------------------------")
    print(llm.predict("hello"))
    # print(llm.predict(
    #     "Analyze the following C language  code of IKE protocol implement. If these codes are related to the IKE status, specify the IKE protocol status in json format."
    #      + str(codeDoc[0].page_content)
    #     ))


def chatIKE():
    persist_dir = "/home/why/sec_sea/llm/IPsecGpt/remeo"
    loader = TextLoader(file_path="/home/why/sec_sea/protocols/ike/strongswan/src/libcharon/sa/ike_sa.c")
    # loader = TextLoader(file_path="/home/why/sec_sea/protocols/ike/strongswan/src/charon/charon.c")
    
    # codeLoader = TextLoader(file_path="/home/why/sec_sea/llm/hello.c")
    # loader = DirectoryLoader('/home/why/sec_sea/protocols/ike/strongswan/', glob="**/*.c", use_multithreading=False, loader_cls=TextLoader)
    code_data = loader.load()
    # doc_sources = [doc.metadata['source']  for doc in code_data]
    # print("number of C files: " + str(len(doc_sources)))
    # for c_source in doc_sources:
    #     print(c_source)

    cppSpliter = RecursiveCharacterTextSplitter.from_language(
        language=Language.CPP, 
        chunk_size=2000, 
        chunk_overlap=100)
    code_doc = cppSpliter.split_documents(code_data)

    # Set API base for OpenAIEmbeddings
    embeddings = OpenAIEmbeddings(openai_api_base=OPENROUTER_API_BASE) 
    vectordb = Chroma.from_documents(code_doc, embeddings, persist_directory=persist_dir)
    vectordb.persist()

    code_qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model_name=model_list["gpt-3.5-turbo-16k"], openai_api_key="", openai_api_base=OPENROUTER_API_BASE), # Set OpenRouter API base
        retriever=vectordb.as_retriever(),
        return_source_documents=True,
        
        )
    
    query = query_list["IKE_Pathrule"]
    result = code_qa({"question": query, "chat_history": ""})
    print(result["source_documents"][0].page_content)
    print("-------------------------------------")
    print(result["answer"])
    
def test():
    docs = load_local_files("/home/why/sec_sea/Fuzzers/ProtocolGPT/ProtocolGPT/test_code/test")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=28000,
        chunk_overlap=256)
    cpp_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.CPP,
        chunk_size=28000,
        chunk_overlap=256
    )
    texts = text_splitter.split_documents(docs)
    print(len(texts))
    for t in texts[:2]:
        print(t.page_content)
        print("\n---------------------------\n\n\n")
    
    
if __name__=="__main__":
    # chatIKE()
    # chatGPT()
    test()