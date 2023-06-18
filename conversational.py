import os

from dotenv import load_dotenv
from abc import ABCMeta, abstractmethod


from langchain.embeddings import OpenAIEmbeddings
import indexdb
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)

from langchain import HuggingFaceHub
# from utilities import load_api_key


# from langchain.vectorstores import Chroma
# from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI , ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory,ConversationSummaryMemory,ConversationSummaryBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

# from langchain.chains.chat_vector_db.prompts import (CONDENSE_QUESTION_PROMPT,
#                                                      QA_PROMPT)
"""Create a ChatVectorDBChain for question/answering."""
# from langchain.callbacks.base import AsyncCallbackManager
from langchain.callbacks.manager import AsyncCallbackManager
# from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ChatVectorDBChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain 
from langchain.chains.summarize import load_summarize_chain
from langchain.llms import OpenAI
from langchain.vectorstores.base import VectorStore
from langchain.callbacks.tracers import LangChainTracer
import  openai

from transformers import TFAutoModelForQuestionAnswering
from langchain.chains import AnalyzeDocumentChain 
from langchain.chains.summarize import load_summarize_chain
from langchain.cache import SQLiteCache
from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain.cache import GPTCache
import hashlib ,langchain
# from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
import logging
import sys


load_dotenv()
langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

class conversatioalBase(metaclass=ABCMeta):
    def __init__(self,embedding,retriver) -> None:
        if embedding is None:
            print("default encoding set!")
            self.embedding = self.getDefaultEmbadding()
        else:
            self.embedding=embedding
        if retriver is None:
            print("default retriver set!")
            self.retriver = indexdb.Faissindexdb(embedding=self.embedding)
        else:
            self.retriver=retriver   
       
     
    @abstractmethod   
    def  getDefaultEmbadding(self): 
        pass

class conversatioalChatGPT(conversatioalBase):
    
    def __init__(self,embedding=None,retriver=None,chat_prompt_template=os.environ.get("CONDENSE_TEMPLATE"),qa_template=os.environ.get("QA_TEMPLATE"),map_template=os.environ.get("summarize_TEMPLATE"),refine_template=os.environ.get("refine_TEMPLATE")) -> None:
        super().__init__(embedding,retriver)
        self.chat_prompt_template=chat_prompt_template
        self.qa_template=qa_template
        self.map_template=map_template
        self.refine_template=refine_template
        logging.basicConfig(stream=sys.stdout, level=logging.INFO)
        logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))      
    
    def getDefaultEmbadding(self): 
        print ("create OpenAIEmbeddings")
        return OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
    
    
    def getConversational(self,tracing:bool=True)-> ConversationalRetrievalChain:

    
        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(self.chat_prompt_template)
        QA_PROMPT= PromptTemplate.from_template(template=self.qa_template)
        MAP_PROMPT=PromptTemplate.from_template(self.map_template)
        REFINE_PROMPT=PromptTemplate.from_template(self.refine_template)
        
        
        
        manager = AsyncCallbackManager([])
        
        # if tracing:
        #     tracer = LangChainTracer()
        #     # tracer.load_default_session()
        #     manager.add_handler(tracer)
        #     # question_manager.add_handler(tracer)
        #     # stream_manager.add_handler(tracer)

        
        # memory = ConversationBufferMemory(
        #     memory_key="chat_history", 
        #     return_messages=True
        # )
        
        llm = ChatOpenAI(temperature=0,verbose=True,cache=True,model_name="gpt-3.5-turbo")
        memory = ConversationSummaryBufferMemory(
            llm=llm,
            output_key='answer',
            max_token_limit=750,
            memory_key='chat_history',
            return_messages=True)
                
        


        question_generator = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)#prompt=CONDENSE_QUESTION_PROMPT)
        # streaming=True, callbacks=[StreamingStdOutCallbackHandler()],
        streaming_llm = OpenAI( temperature=0,verbose=True,cache=True)
        doc_chain = load_summarize_chain(streaming_llm , chain_type="map_reduce")
        # doc_chain = load_qa_chain(streaming_llm, chain_type="map_reduce")

        return ConversationalRetrievalChain(
            retriever=self.retriver.getDb().as_retriever(),
            question_generator=question_generator,
            combine_docs_chain=doc_chain,
            memory=memory,
            get_chat_history=lambda h: h,
            verbose=True,
            # return_source_documents=True
            
        )
        
        
        
        
        
        
        # question_gen_llm =ChatOpenAI(temperature=0, model="gpt-4",verbose=True,cache=True) 
        # question_generator = LLMChain(llm=question_gen_llm)#, prompt=CONDENSE_QUESTION_PROMPT)
        
        # streaming_llm = OpenAI(streaming=True, callbacks=[StreamingStdOutCallbackHandler()], temperature=0,model='gpt-3.5-turbo',verbose=True,cache=True)
        # doc_chain = load_qa_chain(streaming_llm, chain_type="map_reduce")
        
        # return ConversationalRetrievalChain(
        #     question_generator=question_generator,
        #     retriever=self.retriver.getDb().as_retriever(),
        #     combine_docs_chain=doc_chain ,
        #     memory=memory,
        #     get_chat_history=lambda h: h,
        #     verbose=True,
        #     return_source_documents=True
        # )
        
        
if __name__ == "__main__":
    query = "איך אני לוקח משכנתא?"
    # query ="איך אני טס לירח?"
    # query = "מיסים"
   
    print("conversatioalChatGPT", conversatioalChatGPT().getConversational()
          (
                {
                    'question': query
                    # 'chat_history': []
                }
            )
          )
   

    # print("conversatioalChatGPT", conversatioalChatGPT().getConversational()
    #       (
    #             {
    #                 'question': "מה התהליך?",
    #                 'chat_history':[['איך אני לוקח משכנתא?', 'לקחת משכנתא מתחיל בחיפוש אחר חברות משכנתא מסוגים שונים שניתן לבחור מתוכן. יש לבדוק את התנאים של כל חברת משכנתא ולבחור את החברה המתאימה ביותר לצרכים שלך. יש לצרף את המסמכים הנדרשים לבקשת המשכנתא, כולל פרטי הכנסה והה']]
    #             }
    #         )
    #       )
    