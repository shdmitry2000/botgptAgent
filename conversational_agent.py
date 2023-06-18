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

from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.schema import AIMessage, HumanMessage, SystemMessage

# https://github.com/sudarshan-koirala/youtube-stuffs/blob/main/langchain/cache_llm_calls.ipynb


load_dotenv()

langchain.llm_cache = SQLiteCache(database_path=".langchain.db")



class conversatioalAgentsChatGPT():
    
    # def __init__(self,embedding=None,retriver=None,chat_prompt_template=os.environ.get("CONDENSE_TEMPLATE"),qa_template=os.environ.get("QA_TEMPLATE"),map_template=os.environ.get("summarize_TEMPLATE"),refine_template=os.environ.get("refine_TEMPLATE")) -> None:
    #     super().__init__(embedding,retriver)
    #     self.chat_prompt_template=chat_prompt_template
    #     self.qa_template=qa_template
    #     self.map_template=map_template
    #     self.refine_template=refine_template
    #     logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    #     logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

        
    
    def getDefaultEmbadding(self): 
        print ("create OpenAIEmbeddings")
        return OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
        
    def load_agent_chain(self,llm,tools,memory):
        return  initialize_agent(tools=tools, llm=llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)

    def load_agent_chain_w_planer(self,llm,tools,memory,planner_templait):

        planner = load_chat_planner(llm)
        executor = load_agent_executor(llm, tools, verbose=True)
        return PlanAndExecute(planner=planner, executor=executor, verbose=True, memory=memory)

    
    def getConversational(self):

        tools = [
            # searchTools.SearchTool().getToolDefinition(),
            stockTools.StockTool().getToolDefinition(),
            llamaTools.LlamaTool(llamaindexdb=llamaindexdb.LLamaindexdb(),withmemory=False).getToolDefinition(),
            mathTools.MathTool().getToolDefinition(),
        ]

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        # system_message_prompt_content="You are a friendly, conversational banking assistant. Talk about your  data  and answer any questions.It's ok if you don't know the answer. Answer on the same language as question. Use search only if noone know answer. "
        #planner_template=""
        llm = ChatOpenAI(temperature=0,model="gpt-3.5-turbo")
       
        agent_chain =self.load_agent_chain(llm,tools,memory)
        
        # agent_chain=self.load_agent_chain_w_planer(llm,tools,memory,planner_templait=system_message_prompt_content)
        

        


        # model = ChatOpenAI(temperature=0)
        # planner = load_chat_planner(model)
        # executor = load_agent_executor(model, tools, verbose=True)
        # agent = PlanAndExecute(planner=planner, executor=executor, verbose=True)

        return agent_chain




        
        
       
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
    
