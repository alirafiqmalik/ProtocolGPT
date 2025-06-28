# llm.py

import os
import time
import logging
from typing import Optional

# import gpt4all
import questionary
from halo import Halo
from langchain.vectorstores import FAISS
from langchain.callbacks.manager import CallbackManager
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceEmbeddings
from langchain.llms import LlamaCpp
from langchain.text_splitter import RecursiveCharacterTextSplitter, Language
from langchain.memory import ConversationSummaryMemory

from consts import MODEL_TYPES, OPENROUTER_API_BASE # Import OPENROUTER_API_BASE
from utils import load_local_files, get_local_vector_store, calculate_cost, StreamStdOut


class BaseLLM:

    def __init__(self, root_dir, config):
        self.config = config
        self.llm = self._create_model()
        self.root_dir = root_dir
        self.vector_store = self._create_store(root_dir)

    def _create_store(self, root_dir):
        raise NotImplementedError("Subclasses must implement this method.")

    def _create_model(self):
        raise NotImplementedError("Subclasses must implement this method.")

    def embedding_search(self, query, k):
        return self.vector_store.search(query, k=k, search_type="similarity")

    #todo: Add vector_DB selection
    def _create_vector_store(self, embeddings, index, root_dir):
        k = int(self.config.get("k"))
        index_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"vector_store/{index}_"+os.path.normpath(root_dir).split("/")[-1]+"_"+str(self.config.get("chunk_size")) )
        new_db = get_local_vector_store(embeddings, index_path)
        if new_db is not None:
            return new_db.as_retriever(search_kwargs={"k": k})

        docs = load_local_files(root_dir)
        
        if len(docs) == 0:
            print("✘ No documents found")
            exit(0)
        # text_splitter = RecursiveCharacterTextSplitter(
        #     chunk_size=int(self.config.get("chunk_size")),
        #     chunk_overlap=int(self.config.get("chunk_overlap"))
        # )
        text_splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.CPP,
            chunk_size=int(self.config.get("chunk_size")),
            chunk_overlap=int(self.config.get("chunk_overlap"))
        )
        texts = text_splitter.split_documents(docs)
        if index == MODEL_TYPES["OPENAI"] or index == MODEL_TYPES["OPENROUTER"]: # Adjust condition
            cost = calculate_cost(docs, self.config.get("openai_model_name"))
            approve = questionary.select(
                f"Creating a vector store will cost ~${cost:.5f}. Do you want to continue?",
                choices=[
                    {"name": "Yes", "value": True},
                    {"name": "No", "value": False},
                ]
            ).ask()
            if not approve:
                exit(0)

        spinners = Halo(text=f"Creating vector store", spinner='dots').start()
        db = FAISS.from_documents([texts[0]], embeddings)
        for i, text in enumerate(texts[1:]):
            spinners.text = f"Creating vector store ({i + 1}/{len(texts)})"
            db.add_documents([text])
            db.save_local(index_path)
            time.sleep(0.1)

        spinners.succeed(f"Created vector store")
        return db.as_retriever(search_kwargs={"k": k})

    def send_query(self, query):
        retriever = self._create_store(self.root_dir)
        # qa = RetrievalQA.from_chain_type(
        #     llm=self.llm,
        #     chain_type="stuff",
        #     retriever=retriever,
        #     return_source_documents=True
        # )
        
        memory = ConversationSummaryMemory(
            llm=self.llm, 
            memory_key="chat_history", 
            return_messages=True
            )
        qa = ConversationalRetrievalChain.from_llm(
            self.llm, 
            retriever=retriever, 
            memory=memory, 
            output_key="answer",
            return_source_documents=True,
            get_chat_history=lambda h :h,
            )
        docs = qa(query)
        print("----------------------------")
        print(docs)
        print("----------------------------")
        file_paths = [os.path.abspath(s.metadata["source"]) for s in docs['source_documents']]
        print('\n'.join([f'📄 {file_path}:' for file_path in file_paths]))

    def chat_loop(self):
        retriever = self._create_store(self.root_dir)
        memory = ConversationSummaryMemory(
            llm=self.llm, 
            memory_key="chat_history", 
            return_messages=True,
            output_key="answer"
            )
        # qa = RetrievalQA.from_chain_type(
        #     llm=self.llm,
        #     chain_type="stuff",
        #     retriever=retriever,
        #     return_source_documents=True
        # )
        qa = ConversationalRetrievalChain.from_llm(
            self.llm, 
            retriever=retriever, 
            memory=memory, 
            return_source_documents=True,
            get_chat_history=lambda h :h,
            )
        while True:
            query = input("👉 ").lower().strip()
            if not query:
                print("🤖 Please enter a query")
                continue
            if query in ('exit', 'quit'):
                break
            print("----------------------------")
            
            # docs{question,chat_history,answer,source_documents{Document[page_content,metadata]}
            docs = qa(query)
            # print("----------------------------")
            file_paths = [os.path.abspath(s.metadata["source"]) for s in docs['source_documents']]
            print('\n'.join([f'📄 {file_path}:' for file_path in file_paths]))
            logging.info("Prompt: \n" + docs['question'])
            logging.info("Answer: \n" + docs['answer'])
            logging.info('\n'.join([f'📄 {file_path}:' for file_path in file_paths]))
            logging.info("---------------------------------------------------------------------------")


# class LocalLLM(BaseLLM):

#     def _create_store(self, root_dir: str) -> Optional[FAISS]:
#         embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
#         return self._create_vector_store(embeddings, MODEL_TYPES["LOCAL"], root_dir)

#     def _create_model(self):
#         os.makedirs(self.config.get("model_path"), exist_ok=True)
#         gpt4all.GPT4All.retrieve_model(model_name=self.config.get("local_model_name"),
#                                        model_path=self.config.get("model_path"))
#         model_path = os.path.join(self.config.get("model_path"), self.config.get("local_model_name"))
#         model_n_ctx = int(self.config.get("max_tokens"))
#         model_n_batch = int(self.config.get("n_batch"))
#         callbacks = CallbackManager([StreamStdOut()])
#         llm = LlamaCpp(model_path=model_path, n_ctx=model_n_ctx, n_batch=model_n_batch, callbacks=callbacks,
#                        verbose=False)
#         llm.client.verbose = False
#         return llm


class OpenAILLM(BaseLLM):
    def _create_store(self, root_dir: str) -> Optional[FAISS]:
        # Conditionally set openai_api_base for embeddings based on model_type
        if self.config.get("model_type") == MODEL_TYPES["OPENROUTER"]:
            embeddings = OpenAIEmbeddings(openai_api_key=self.config.get("api_key"), openai_api_base=OPENROUTER_API_BASE)
        else:
            embeddings = OpenAIEmbeddings(openai_api_key=self.config.get("api_key"))
        return self._create_vector_store(embeddings, self.config.get("model_type"), root_dir) # Pass model_type as index

    def _create_model(self):
        # Conditionally set openai_api_base for ChatOpenAI based on model_type
        if self.config.get("model_type") == MODEL_TYPES["OPENROUTER"]:
            return ChatOpenAI(model_name=self.config.get("openai_model_name"),
                            openai_api_key=self.config.get("api_key"),
                            openai_api_base=OPENROUTER_API_BASE, # Set OpenRouter API base
                            streaming=True,
                            max_tokens=int(self.config.get("max_tokens")),
                            callback_manager=CallbackManager([StreamStdOut()]),
                            temperature=float(self.config.get("temperature")))
        else:
            return ChatOpenAI(model_name=self.config.get("openai_model_name"),
                            openai_api_key=self.config.get("api_key"),
                            streaming=True,
                            max_tokens=int(self.config.get("max_tokens")),
                            callback_manager=CallbackManager([StreamStdOut()]),
                            temperature=float(self.config.get("temperature")))


def factory_llm(root_dir, config):
    return OpenAILLM(root_dir, config)
    # if config.get("model_type") == "openai":
    #     return OpenAILLM(root_dir, config)
    # else:
    #     return LocalLLM(root_dir, config)