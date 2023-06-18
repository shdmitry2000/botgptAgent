import os

from dotenv import load_dotenv
from abc import ABCMeta, abstractmethod


from langchain.embeddings import OpenAIEmbeddings

from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)

from llama_index import GPTVectorStoreIndex, SimpleDirectoryReader, LLMPredictor, PromptHelper, ServiceContext
from llama_index.indices.query.base import BaseQueryEngine
from langchain import HuggingFaceHub
# from utilities import load_api_key


# from langchain.vectorstores import Chroma
# from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI , ChatAnthropic
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory,ConversationSummaryMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import llamaindexdb
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

load_dotenv()
# langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

import logging
import sys

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")

# logging.basicConfig(stream=sys.stdout, level=logging.INFO)
# logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

class QuestionalBase(metaclass=ABCMeta):
    # def __init__(self,embedding,retriver) -> None:
    #     if embedding is None:
    #         print("default encoding set!")
    #         self.embedding = self.getDefaultEmbadding()
    #     else:
    #         self.embedding=embedding
    #     if retriver is None:
    #         print("default retriver set!")
    #         self.retriver = indexdb.Faissindexdb(embedding=self.embedding)
    #     else:
    #         self.retriver=retriver   
       
     
    @abstractmethod   
    def  getDefaultEmbadding(self): 
        pass

class QuestionalLlama(QuestionalBase):
    
    # def __init__(self,embedding=None,retriver=None,chat_prompt_template=os.environ.get("CONDENSE_TEMPLATE"),qa_template=os.environ.get("QA_TEMPLATE"),map_template=os.environ.get("summarize_TEMPLATE"),refine_template=os.environ.get("refine_TEMPLATE")) -> None:
    #     super().__init__(embedding,retriver)
    #     self.chat_prompt_template=chat_prompt_template
    #     self.qa_template=qa_template
    #     self.map_template=map_template
    #     self.refine_template=refine_template
        
    
    def getDefaultEmbadding(self): 
        print ("create OpenAIEmbeddings")
        return OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
    
    

    
    def getQuestional(self,tracing:bool=True)->BaseQueryEngine:

    
        # CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(self.chat_prompt_template)
        # QA_PROMPT= PromptTemplate.from_template(template=self.qa_template)
        # MAP_PROMPT=PromptTemplate.from_template(self.map_template)
        # REFINE_PROMPT=PromptTemplate.from_template(self.refine_template)
        
        # This example uses text-davinci-003 by default; feel free to change if desired
        # max_tokens
        # https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjl3s-Wu73_AhUeX_EDHffzAFUQz40FegQIChA3&url=https%3A%2F%2Fwww.youtube.com%2Fwatch%3Fv%3D82Fc9Olldnw&usg=AOvVaw3qBXsix9MyhqKjbsdaIHah
        
        llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003",verbose=True,cache=True,max_tokens=460))



        return llamaindexdb.LLamaindexdb(llm_predictor=llm_predictor).getDb().as_query_engine()
        
    
        
        
        
if __name__ == "__main__":
    query = "איך אני לוקח משכנתא?"
    # query ="איך אני טס לירח?"
    # query = "מיסים"
   
    print("conversatioalllama", QuestionalLlama().getQuestional().query(query))
          
   

    # print("QuestionalLlama", QuestionalLlama().getConversational()
    #       (
    #             {
    #                 'question': "מה התהליך?",
    #                 'chat_history':[['איך אני לוקח משכנתא?', 'לקחת משכנתא מתחיל בחיפוש אחר חברות משכנתא מסוגים שונים שניתן לבחור מתוכן. יש לבדוק את התנאים של כל חברת משכנתא ולבחור את החברה המתאימה ביותר לצרכים שלך. יש לצרף את המסמכים הנדרשים לבקשת המשכנתא, כולל פרטי הכנסה והה']]
    #             }
    #         )
    #       )
    