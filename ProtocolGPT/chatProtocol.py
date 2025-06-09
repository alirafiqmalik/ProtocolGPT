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

query_list = {
    "IKE_States" : "As a IKE protocol specialist, your task is to extract IKE protocol states and how many IKE protocol states in these code snippets of IKE implementation?  Represent them with json.",
    "IKE_Pathrule" : "As a IKE protocol specialist, your task is to extract the IKE protocol state from the IKE implementation code and the constraints on the packets that can be accepted in different states. These constraints include the type and amount of payload in the IKE packet, and whether the payload is encrypted. Represent them with json.",
}

model_list = {
    "gpt-3.5-turbo" : "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k" : "gpt-3.5-turbo-16k",
    "gpt-4" : "gpt-4",
    "gpt-4-1106-preview" : "gpt-4-1106-preview",
}

def chatGPT():
    # openai.api_base = "https://openai.api2d.net/v1"
    # llm = OpenAI()
    # chat_model = ChatOpenAI(model="gpt-3.5-turbo")
    # print(llm.predict("hello"))
    
    completion = openai.ChatCompletion.create(model="gpt-3.5-turbo", messages=[{"role": "user", "content": "Hello world!"}])
    print(completion.choices[0].message.content)

def test():
    import http.client
    import json

    conn = http.client.HTTPSConnection("oa.api2d.net")
    payload = json.dumps({
    "model": "gpt-3.5-turbo",
    "messages": [
            {
                "role": "user",
                "content": "讲个笑话"
            }
        ],
        "safe_mode": False
    })
    headers = {
        'Authorization': 'Bearer fk....',
        'User-Agent': 'Apifox/1.0.0 (https://apifox.com)',
        'Content-Type': 'application/json'
    }
    conn.request("POST", "/v1/chat/completions", payload, headers)
    res = conn.getresponse()
    data = res.read()
    print(data.decode("utf-8"))




def chatIKE():
    persist_dir = "/home/why/sec_sea/Fuzzers/ProtocolGPT/ProtocolGPT"
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

    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(code_doc, embeddings, persist_directory=persist_dir)
    vectordb.persist()

    code_qa = ConversationalRetrievalChain.from_llm(
        llm=ChatOpenAI(temperature=0, model_name=model_list["gpt-4"], openai_api_key=""), 
        retriever=vectordb.as_retriever(),
        return_source_documents=True)
    
    query = query_list["IKE_Pathrule"]
    result = code_qa({"question": query, "chat_history": ""})
    print(result["source_documents"][0].page_content)
    print("-------------------------------------")
    print(result["answer"])
    
    
def test_path():
    path1 = "./testcode/charon"
    path2 = "./testcode/charon/"
    print(os.path.normpath(path1).split("/")[-1])
    print(type(os.path.abspath(path1)))
    print(os.path.normpath(path2).split("/")[-1])

if __name__=="__main__":
    # chatIKE()
    # chatGPT()
    test_path()
    