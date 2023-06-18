import os

from dotenv import load_dotenv
from abc import ABCMeta, abstractmethod

import langchain
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
from langchain.memory import ConversationBufferMemory,ConversationSummaryMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

from searchTools import SearchTool

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

from langchain.cache import InMemoryCache
from langchain.callbacks import get_openai_callback
# We can do the same thing with a SQLite cache
from langchain.cache import SQLiteCache
from langchain.chat_models import ChatOpenAI
from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain import LLMMathChain

from llama_index import ListIndex
from llama_index.langchain_helpers.memory_wrapper import GPTIndexChatMemory
import logging
import sys

import searchTools
import mathTools
import llamaTools
import llamaindexdb

from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory

import searchTools
import llamaTools
import stockTools
import llamaindexdb
import baseToolsHandler
import mathTools
from langchain.agents import initialize_agent
from langchain.agents import AgentType


# https://github.com/sudarshan-koirala/youtube-stuffs/blob/main/langchain/cache_llm_calls.ipynb


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
        
    
    def getDefaultEmbadding(self): 
        print ("create OpenAIEmbeddings")
        return OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
    
    
    def getConversational(self)-> ConversationalRetrievalChain:

    
    
        # memory =ConversationSummaryMemory

        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=False
        )
    
        # def getretriver(self):
        #     # return dbCroma
        #     # return Faissindexdb(embedding=self.getDefaultEmbadding())
        #     return self.retriver
    

        CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(self.chat_prompt_template)
        QA_PROMPT= PromptTemplate.from_template(self.qa_template)
        MAP_PROMPT=PromptTemplate.from_template(self.map_template)
        REFINE_PROMPT=PromptTemplate.from_template(self.refine_template)
        
        # # define two LLM models from OpenAI
        # question_gen_llm = ChatOpenAI(
        #     temperature=0.2,
        #     # max_tokens=2000,
        # #    model_name="gpt-3.5-turbo" #text-davinci-003",
        # )
        

        



        llm = OpenAI(batch_size=5,temperature=0,cache=True,model="text-davinci-003",max_tokens=4096)
        question_gen_llm = LLMChain(llm=llm, prompt=CONDENSE_QUESTION_PROMPT)
        

        # qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=qa_chain)
        # streaming=True, callbacks=[StreamingStdOutCallbackHandler()]
        streaming_llm = OpenAI( streaming=True, callbacks=StreamingStdOutCallbackHandler(), temperature=0.2,verbose=True,cache=True,max_tokens=400)#,model="text-davinci-003")
        # streaming=True, callbacks=[StreamingStdOutCallbackHandler()]
        doc_chain = load_qa_chain(streaming_llm, 
                                    # chain_type="refine",
                                    chain_type="map_reduce",
                                    # return_refine_steps=True,
                                    # question_prompt=QA_PROMPT,
                                    # refine_prompt=REFINE_PROMPT
                                    )

        # doc_chain = load_qa_chain(llm, chain_type="map_reduce")
        # doc_chain = load_qa_chain(llm, chain_type="map_reduce")
        # doc_chain = load_summarize_chain(llm, chain_type="map_reduce",verbose=True,map_prompt=MAP_PROMPT,combine_prompt=CONDENSE_QUESTION_PROMPT,return_intermediate_steps=True)
        # doc_chain = load_summarize_chain(llm, chain_type="map_reduce",verbose=True,return_intermediate_steps=True)
        
        # qa_document_chain = AnalyzeDocumentChain(combine_docs_chain=doc_chain)
    
        # summary_chain = load_summarize_chain(llm, chain_type="map_reduce")

        return  ConversationalRetrievalChain(
            retriever=self.retriver.getDb().as_retriever(),
            question_generator=question_gen_llm,
            combine_docs_chain=doc_chain,
            memory=memory,
            get_chat_history=lambda h: h,
            verbose=True,
            # return_source_documents=True
        )



        # chain = load_summarize_chain(llm, chain_type="map_reduce",verbose=True,map_prompt=PROMPT,combine_prompt=COMBINE_PROMPT)
        
        # return  ConversationalRetrievalChain.from_llm(
        #     llm=question_gen_llm,
        #     # chain_type='stuff',
        #     chain_type="map_reduce",
        #     # chain_type="refine" ,
        #     # retriever=self.retriver.getDb().as_retriever(refine_template=os.environ.get("refine_TEMPLATE")),
        #     # retriever=self.retriver.getDb().as_retriever(qa_template=QA_PROMPT, question_generator_template=CONDENSE_QUESTION_PROMPT),
        #     retriever=self.retriver.getDb().as_retriever(qa_template=QA_PROMPT, question_generator_template=CONDENSE_QUESTION_PROMPT,prompt_template=os.environ.get("summarize_TEMPLATE")),
        #     #  retriever=self.retriver.getDb().as_retriever(qa_template=QA_PROMPT, question_generator_template=CONDENSE_QUESTION_PROMPT),
        #     # retriever=self.retriver.getDb().as_retriever(prompt_template=os.environ.get("summarize_TEMPLATE")),
            
        #     memory=memory,
        #     get_chat_history=lambda h: h,
        #     verbose=True,
        #     # return_source_documents=True
        # )
    
    
    
    # def getTestConversational():
        
    #     CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(self.chat_prompt_template)
    #     QA_PROMPT= PromptTemplate.from_template(self.qa_template)
    #     # condense_template = """Given the following chat history and a follow up question, rephrase the follow up input question to be a standalone question.Or end the conversation if it seems like it's done.Chat History:\"""{chat_history} \"""Follow Up Input: \"""{question}\"""Standalone question:"""
    
    #     # condense_question_prompt = PromptTemplate.from_template(template)
    
    #     # qa_template = """You are a friendly, conversational banking assistant. Talk about your  data  and answer any questions.It's ok if you don't know the answer.Context:\"""{context}\"""Question:\"{question}\"""Helpful Answer :"""
    
    #     # qa_prompt= PromptTemplate.from_template(template)

        
    #     # define two LLM models from OpenAI
    #     question_gen_llm = OpenAI(
    #         temperature=0.2,
    #         # max_tokens=2000,
    #         model_name="text-curie-001" ,#text-davinci-003",
    #     )
        
    #     # # llm = ChatOpenAI(temperature=0.0,model_name='gpt-3.5-turbo')
    #     # question_gen_llm = openai.Completion.create(
    #     #     engine="text-davinci-003",
    #     #     # prompt='\n'.join([f"{m['role']}: {m['content']}" for m in message_history]),
    #     #     temperature=0.2,
    #     #     max_tokens=1024,
    #     #     n=1,
    #     #     stop=None,
    #     #     timeout=30,
        
    #     # use the LLM Chain to create a question creation chain
    #     question_generator = LLMChain(
    #         llm=question_gen_llm,
    #         prompt=CONDENSE_QUESTION_PROMPT
    #     )
        
        
    #     streaming_llm = OpenAI(
    #         streaming=True,
    #         callback_manager=CallbackManager([
    #             StreamingStdOutCallbackHandler()
    #         ]),
    #         verbose=True,
    #         max_tokens=2000,
    #         temperature=0.2
    #     )
    #     # use the streaming LLM to create a question answering chain
    #     doc_chain = load_qa_chain(
    #         llm=streaming_llm,
    #         chain_type="stuff",
    #         prompt=QA_PROMPT
    #     )

    #     chatbot = ConversationalRetrievalChain(
    #         retriever=getretriver().as_retriever(),
    #         combine_docs_chain=doc_chain,
    #         question_generator=question_generator
    #     )
    
    #     return chatbot


    # # qa =getStandartConversational()
    # # return qa
    
 
    
        
        
       
if __name__ == "__main__":
    query = "איך אני לוקח משכנתא?"
    query="hi, i am bob"
    # query = "מיסים"
   
    # print("conversatioalChatGPT", conversatioalChatGPT().getConversational()
    #       (
    #             {
    #                 'question': query,
    #                 'chat_history': []
    #             }
    #         )
    #       )
   

    # print("conversatioalChatGPT", conversatioalChatGPT().getConversational()
    #       (
    #             {
    #                 'question': "מה התהליך?",
    #                 'chat_history':[['איך אני לוקח משכנתא?', 'לקחת משכנתא מתחיל בחיפוש אחר חברות משכנתא מסוגים שונים שניתן לבחור מתוכן. יש לבדוק את התנאים של כל חברת משכנתא ולבחור את החברה המתאימה ביותר לצרכים שלך. יש לצרף את המסמכים הנדרשים לבקשת המשכנתא, כולל פרטי הכנסה והה']]
    #             }
    #         )
    #       )
    

    # query="how much i need to pay for 5 meter of ceramic with price $100 per meter?"
    print("conversatioalAgentsChatGPT", conversatioalAgentsChatGPT().getConversational().run(query))
    
