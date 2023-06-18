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

import llamaindexdb
import logging
import sys

import baseToolsHandler
import llamaTools



load_dotenv()

# class ToolBase(metaclass=ABCMeta):
   



   
   
     
    # @abstractmethod   
    # def  getDefaulToolDesc(self): 
    #     pass

# class  searchTool():

#     template_with_history = """
#     Thought: {{thought}}
#     Action: {{action}}
#     Action Input: {{action_input}}
#     {% for name, result in action_outputs.items() %}
#     Observation: {{result.observation}}
#     {% endfor %}
#     Final Answer: {{answer}}
#     """

#     prompt_with_history = CustomPromptTemplate(
#         template=template_with_history,
#         tools=[],
#         input_variables=["input", "intermediate_steps", "history"]
#     )


#     def createToolkit()->BaseTool :

#         llm = OpenAI(temperature=0.7)


#         # Create an instance of the Bing search tool
#         return   Bing(api_key=os.environ.get("BING_SUBSCRIPTION_KEY"))
    
        
#     def getToolDefinition():
#         return Tool(
#                 name="Search",
#                 func=searchTool().createToolkit().run,
#                 description="useful for when you need to answer questions about current events"
#                 )

# class  pineconeTool():

#     template_with_history = """
#     Thought: {{thought}}
#     Action: {{action}}
#     Action Input: {{action_input}}
#     {% for name, result in action_outputs.items() %}
#     Observation: {{result.observation}}
#     {% endfor %}
#     Final Answer: {{answer}}
#     """

#     prompt_with_history = CustomPromptTemplate(
#         template=template_with_history,
#         tools=[],
#         input_variables=["input", "intermediate_steps", "history"]
#     )


#     def createToolkit()->BaseTool :

#        # Initialize Pinecone
#         pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"))

#         # Create a new Pinecone service
#         index_name = "knowledge-base"
#         if index_name not in pinecone.list_indexes():
#             pinecone.create_index(name=index_name, dimension=300, metric="cosine")
        
#         # Initialize an OpenAI agent
#         llm = OpenAI(temperature=0)

#         # Create a Pinecone retriever tool
#         knowledge_base_tool = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=index_name)
#         return knowledge_base_tool

    
#     def getToolDefinition():
#         return Tool(
#                 name='Knowledge Base',
#                 func=pineconeTool().createToolkit().run,
#                 description="Useful for general questions about how to do things and for details on interesting topics. Input should be a fully formed question."
#                 )
    
class LlamaTool(baseToolsHandler.projectBaseTool):

    def __init__(self,llamaindexdb,embedding=None,memory_name="mashcanta", withmemory=True) -> None:
        super().__init__(withmemory=withmemory)
        OpenAI.api_key =os.environ["OPENAI_API_KEY"]
        self.llamaindexdb=llamaindexdb

        # if  llm_predictor is None:
        #     self.llm_predictor =  LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003",verbose=True,cache=True,max_tokens=460))
        # else:
        # self.llm_predictor=llm_predictor



    
    # def createToolkit()->BaseTool :
       

    
    def getQueryEngine(self):
        # llm_predictor = LLMPredictor(llm=OpenAI(temperature=0, model_name="text-davinci-003",verbose=True,cache=True,max_tokens=460))
        return self.llamaindexdb.getDb().as_query_engine()

    def getToolDefinition(self):
        qe=self.getQueryEngine()
        return Tool(
                name = "llamaindexdb",
                func=lambda q: str(qe.query(q)),
                description="Useful for general questions about morgage , buying and selling house."
                )
    
        # tool_config = IndexToolConfig(
        #     query_engine=self.getLlamadb().as_query_engine(), 
        #     name=f"morgage Index",
        #     description=f"useful for when you want to answer queries about morgage ",
        #     tool_kwargs={"return_direct": True}
        # )

        # tool = LlamaIndexTool.from_tool_config(tool_config)
        # return tool


    # def createToolkit()->BaseTool :


    # def createToolkitbyconfig()->BaseTool :

    #     tool_config = IndexToolConfig(
    #         query_engine=query_engine, 
    #         name=f"Vector Index",
    #         description=f"useful for when you want to answer queries about X",
    #         tool_kwargs={"return_direct": True}
    #     )

    #     return  LlamaIndexTool.from_tool_config(tool_config)




        
    
    def getExecuter(self):
        # set Logging to DEBUG for more detailed outputs
        llm = ChatOpenAI(temperature=0)
        tools=[LlamaTool(llamaindexdb=self.llamaindexdb).getToolDefinition()]
        agent_executor = initialize_agent(tools, llm, agent="conversational-react-description", memory=self.memory)
        return agent_executor

      

        
       
if __name__ == "__main__":

    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

    executer= LlamaTool(llamaindexdb=llamaindexdb.LLamaindexdb(),withmemory=True).getExecuter()
    print(executer.run(input="איך אני לוקח משכנתא?"))
    