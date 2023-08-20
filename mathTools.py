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
import baseToolsHandler

# https://github.com/sudarshan-koirala/youtube-stuffs/blob/main/langchain/cache_llm_calls.ipynb
load_dotenv()



    
class MathTool(baseToolsHandler.projectBaseTool):

    def __init__(self,withmemory=False) -> None:
        super().__init__(withmemory)
        OpenAI.api_key =os.environ["OPENAI_API_KEY"]
        llm = OpenAI(temperature=0)
        self.llm_math_chain = LLMMathChain.from_llm(llm=llm, verbose=True)
        
    
    def getToolDefinition(self):
        

        return Tool(
                name="Calculator",
                func=self.llm_math_chain.run,
                description="useful for when you need to answer questions about math"
            )
    
        


        
    def getExecuter(self):
        # set Logging to DEBUG for more detailed outputs
        llm = ChatOpenAI(temperature=0)
        tools=[self.getToolDefinition()]
        agent_executor = initialize_agent(tools, llm, agent="conversational-react-description", memory=self.memory)
        return agent_executor

      

        
       
if __name__ == "__main__":


    print(MathTool().getExecuter().run("2*2"))
    
    
    
    



     