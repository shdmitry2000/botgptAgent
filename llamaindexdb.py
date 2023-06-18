import os

from langchain.embeddings import OpenAIEmbeddings
from langchain.embeddings import FakeEmbeddings
from langchain.vectorstores import Chroma
from langchain.vectorstores.faiss import FAISS
from dotenv import load_dotenv
from abc import ABCMeta, abstractmethod
from langchain.document_loaders import DirectoryLoader, TextLoader
# from langchain.retrievers import BaseRetriever
from langchain.indexes import VectorstoreIndexCreator
from  langchain.vectorstores import VectorStore
import pandas as pd
from langchain.text_splitter import CharacterTextSplitter
from llama_index.node_parser import SimpleNodeParser
import os
# from langchain.llms.base import LLM
from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from llama_index import StorageContext, load_index_from_storage
from llama_index import VectorStoreIndex,ListIndex
from langchain.llms import OpenAI
from langchain.cache import SQLiteCache
import hashlib ,langchain
import logging
import sys

# from langchain.llms.openai import OpenAI
# import openai
from transformers import pipeline
from typing import Optional, List, Mapping, Any

# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI

# # from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
# #                                                      QA_PROMPT)
# """Create a ChatVectorDBChain for question/answering."""
# # from langchain.callbacks.base import AsyncCallbackManager
# from langchain.callbacks.manager import AsyncCallbackManager
# # from langchain.callbacks.tracers import LangChainTracer
# from langchain.chains import ChatVectorDBChain
# from langchain.chains.llm import LLMChain
# from langchain.chains.question_answering import load_qa_chain
# from langchain.llms import OpenAI
# from langchain.vectorstores.base import VectorStore
# from langchain.callbacks.tracers import LangChainTracer
# import  openai
import indexdb 
import openai

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

load_dotenv()


class LLamaindexdb(indexdb.baseindexdb):
    def __init__(self,embedding = None,llm_predictor =None,persist_directory = 'llama_db') -> None:
        super().__init__(embedding,persist_directory)
        # bug handling
        openai.api_key =os.environ["OPENAI_API_KEY"]
        
        if  llm_predictor is None:
            self.llm_predictor = self.createDifllmpredictor()
        else:
            self.llm_predictor=llm_predictor
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

        
    def getDefaultEmbadding(self): 
        print(os.environ.get("OPENAI_API_KEY"))
        return OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
    

    def createDifllmpredictor(self):
         return LLMPredictor(llm=OpenAI(temperature=0.4,max_tokens=450 ))#, model_name="text-davinci-003"))

    def createPromptHelper(self):
        # # Configure prompt parameters and initialise helper
        # max_input_size = 4096
        # num_output = 256
        # max_chunk_overlap = 0.2


         # set maximum input size
        max_input_size = 4096
        # set number of output tokens
        num_outputs = 256
        # set maximum chunk overlap
        max_chunk_overlap = 40
        # set chunk size limit
        # chunk_size_limit = 600

        prompt_helper = PromptHelper()
        # prompt_helper = PromptHelper(max_input_size, num_outputs, max_chunk_overlap)#, chunk_size_limit=chunk_size_limit)
        return prompt_helper

    
    def getDb(self):
        from llama_index import StorageContext, load_index_from_storage
        # rebuild storage context
        storage_context = StorageContext.from_defaults(persist_dir=self.persist_directory)

        prompt_helper = PromptHelper()

        # query = "how i can get to the moon?"
        service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor, prompt_helper=self.createPromptHelper())

        # load index
        index = load_index_from_storage(storage_context=storage_context,service_context=service_context)


        return index

        
    
    
    def create_db(self,path,persist_directory=None,embedding=None  ):   
        if persist_directory is None:
            persist_directory = self.persist_directory
        if embedding is None:
            embedding = self.embedding

        

        documents = SimpleDirectoryReader(path,recursive=True, exclude_hidden=True).load_data()

        storage_context = StorageContext.from_defaults()
        


        # query = "how i can get to the moon?"
        service_context = ServiceContext.from_defaults(llm_predictor=self.llm_predictor, prompt_helper=self.createPromptHelper())



        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(documents)
        storage_context.docstore.add_documents(nodes)


# index1 = VectorStoreIndex(nodes, storage_context=storage_context,service_context=service_context)
# index2 = ListIndex(nodes, storage_context=storage_context,service_context=service_context)

# index1.storage_context.persist(persist_dir="llama_node_vectorstore")
# index2.storage_context.persist(persist_dir="llama_node_listindex")

# # build index
        vectordb = VectorStoreIndex.from_documents(
            documents ,storage_context=storage_context , service_context=service_context
        )
        vectordb.storage_context.persist(persist_dir=self.persist_directory)
        
        # storage_context.persist(persist_dir=self.persist_directory)
        return vectordb
    
    def checkdb(self):
        
        # query = "What did the president say about Justice Breyer"
        query = "מה היא משכנתאה?"
        # query = "לירח"

        docs = self.getDb().as_retriever().retrieve(query)
        return len(docs)>0
    



if __name__ == "__main__":
    
    
    
    cdb= LLamaindexdb()
    cdb.create_db(path=os.getcwd()+'/mashkanta/')
    print("LLamaindexdb",cdb.checkdb())

    
    # query = "איך אני מגיע לירח"
    # cdb=Faissindexdb()
    # cdb=LLamaindexdb()
    # answer=cdb.getDb().similarity_search(query)
    query = "מה היא משכנתא?"
    
    # print(cdb.getDb().as_retriever().retrieve(query))
    query_engine = cdb.getDb().as_query_engine()
    response = query_engine.query(query)
    print(response)
    




        