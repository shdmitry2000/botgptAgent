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
from langchain.agents import AgentType
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
# from langchain.experimental.plan_and_execute import PlanAndExecute, load_agent_executor, load_chat_planner
from langchain.llms import OpenAI
from langchain import SerpAPIWrapper
from langchain.agents.tools import Tool
from langchain import LLMMathChain

from llama_index import ListIndex
from llama_index.langchain_helpers.memory_wrapper import GPTIndexChatMemory
import logging
import sys
from langchain.llms import OpenAIChat
from langchain.agents import initialize_agent

# https://github.com/sudarshan-koirala/youtube-stuffs/blob/main/langchain/cache_llm_calls.ipynb
load_dotenv()
import baseToolsHandler

# class projectBaseTool():

#     def __init__(self,withmemory=False) -> None:
#         if withmemory :
#             index = ListIndex([])
#             memory = GPTIndexChatMemory(
#                 index=index, 
#                 memory_key="chat_history", 
#                 query_kwargs={"response_mode": "compact"},
#                 # return_source returns source nodes instead of querying index
#                 return_source=True,
#                 # return_messages returns context in message format
#                 return_messages=True
#             )
#             self.memory =memory
#         else:
#             self.memory = ConversationBufferMemory(memory_key="chat_history")
       


    
class SearchTool(baseToolsHandler.projectBaseTool):

    def __init__(self,memory_name="search_data",withmemory=False) -> None:
        super().__init__(memory_name=memory_name ,withmemory=withmemory)
        OpenAI.api_key =os.environ["OPENAI_API_KEY"]
        
    


    def getToolDefinition(self):
        
        print("SERPAPI_API_KEY",os.environ["SERPAPI_API_KEY"])
        search = SerpAPIWrapper(serpapi_api_key=os.environ["SERPAPI_API_KEY"])
        return Tool(
                name = "Current Search",
                func=search.run,
                description="useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term. Not for translation questions."
            )
        

    
        


        
    def getExecuter(self):
        # set Logging to DEBUG for more detailed outputs
        llm=ChatOpenAI( temperature=0)
        tools=[self.getToolDefinition()]
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True, memory=memory)
        
        return agent_chain
      

        
       
if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    executer=SearchTool().getExecuter()
    executer.run(input="hi, i am bob")
    executer.run(input="what's my name?")
    # print(SearchTool().getExecuter().run("what is time in uk now?"))
    
    
    
    



     