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

from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

from pydantic import BaseModel, Field
import yfinance as yf

import baseToolsHandler
# https://github.com/sudarshan-koirala/youtube-stuffs/blob/main/langchain/cache_llm_calls.ipynb
load_dotenv()

from langchain.tools import BaseTool
from typing import Optional, Type
from langchain.agents import initialize_agent, Tool
from langchain.agents import AgentType
from langchain.chat_models import ChatOpenAI

from pydantic import BaseModel, Field

# https://colab.research.google.com/drive/1o5N38guLVSLHWQvLGUTW8uxjvLeAOhx9?usp=sharing#scrollTo=KMtU1sWTbnZB


# # input params
# class StockPriceCheckInput(BaseModel):
#     """Input for Stock price check."""

#     stockticker: str = Field(..., description="Ticker symbol for stock or index")


# # function
# class StockPriceTool(BaseTool):
#     name = "get_stock_ticker_price"
#     description = "Useful for when you need to find out the price of stock. You should input the stock ticker used on the yfinance API"

#     def _run(self, stockticker: str):
#         # print("i'm running")
#         price_response = get_stock_price(stockticker)

#         return price_response

#     def _arun(self, stockticker: str):
#         raise NotImplementedError("This tool does not support async")

#     args_schema: Optional[Type[BaseModel]] = StockPriceCheckInput


    
class StockTool(baseToolsHandler.projectBaseTool):

    def __init__(self,memory_name="stock_data", withmemory=False) -> None:
        super().__init__(withmemory)
        OpenAI.api_key =os.environ["OPENAI_API_KEY"]
        

    # @staticmethod
    def get_stock_price(self,symbol):
        print("get ticker price for",symbol)
        ticker = yf.Ticker(symbol)
        todays_data = ticker.history(period='1d')
        return round(todays_data['Close'][0], 2)



    def getToolDefinition(self):
        
        StockPriceTool = Tool(
            name='Get Stock Ticker price',
            func= self.get_stock_price,
            # func=lambda q: str(self.get_stock_price(q)),
            description="Useful for when you need to find out the price of stock. You should input the stock ticker used on the yfinance API"
        )

        return StockPriceTool
    
        


        
    def getExecuter(self):
        # set Logging to DEBUG for more detailed outputs
        llm = ChatOpenAI(temperature=0)
        tools=[self.getToolDefinition()]
        agent_executor = initialize_agent(tools, llm, agent="conversational-react-description", memory=self.memory)
        return agent_executor

      

        
       
if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))
    print(StockTool().getExecuter().run(input="What is the price of Google stock"))
    
    
    
    



     