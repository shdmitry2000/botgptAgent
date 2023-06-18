# https://github.com/hwchase17/langchain/blob/master/docs/getting_started/getting_started.md
import os

from dotenv import load_dotenv

# do imports
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.agents import initialize_agent

from llama_index.langchain_helpers.agents import LlamaToolkit, create_llama_chat_agent, IndexToolConfig

from abc import ABCMeta, abstractmethod

from langchain import OpenAI
from langchain.llms import OpenAIChat
from langchain.agents import load_tools#,BaseTool
from langchain.agents import initialize_agent
from langchain.agents import AgentType,AgentExecutor
from langchain.llms import OpenAI
from langchain.utilities import BingSearchAPIWrapper
from langchain.chat_models import ChatOpenAI

from langchain.agents import LLMSingleActionAgent
# from langchain.executors import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory

# from langchain.templates import CustomPromptTemplate

from langchain.tools import bing_search

# import pinecone
from langchain.chains import RetrievalQA
from langchain import OpenAI
from langchain.tools import BaseTool, StructuredTool, Tool, tool

from langchain.agents import LLMSingleActionAgent
# from langchain.executors import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from llama_index.langchain_helpers.agents import IndexToolConfig, LlamaIndexTool

import indexdb
import logging
import sys

import baseToolsHandler




load_dotenv()
    
class IndexTool(baseToolsHandler.projectBaseTool):

    def __init__(self,indexdb,embedding=None,memory_name="Index", withmemory=False) -> None:
        super().__init__(withmemory=withmemory)
        OpenAI.api_key =os.environ["OPENAI_API_KEY"]
        self.indexdb=indexdb

       
    
    def getSimilarityEngine(self):
        # llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003",verbose=True,cache=True,max_tokens=460))
        return self.indexdb.getDb()
     

    def getToolDefinition(self):
        qe=self.getSimilarityEngine()
        return Tool(
                name = "Finance education",
                func=lambda q: str(qe.similarity_search(q)),
                description="Useful for general questions about finance , and how to start business or save the money."
                )
    
       



        
    


    def getExecuter(self):
        # set Logging to DEBUG for more detailed outputs
        llm=ChatOpenAI( temperature=0)
        tools=[self.getToolDefinition()]
        
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        agent_chain = initialize_agent(tools, llm, agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION, verbose=True ,max_tokens=2000,memory=memory)
        
        return agent_chain

        
       
if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    executer= IndexTool(indexdb=indexdb.Faissindexdb(),withmemory=False).getExecuter()
    executer.run(input="איך אני פוטתח עסק?")
    

    
    
    