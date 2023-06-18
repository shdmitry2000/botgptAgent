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


load_dotenv()


class baseindexdb:
    
    def __init__(self,embedding,persist_directory ) -> None:
        if embedding is None:
            self.embedding = self.getDefaultEmbadding()
        else:
            self.embedding=embedding
        self.persist_directory=persist_directory
    
    def create_doc_chunks_split(self,loader):
        splitter = CharacterTextSplitter(separator=" ", chunk_size=1900, chunk_overlap=150)
        source_chunks = []
        source_chunks.extend(splitter.split_documents(loader))    
        return source_chunks
    
    
    @abstractmethod   
    def  getDefaultEmbadding(self): 
        pass

    
    @abstractmethod
    def create_db(self,loaders,persist_directory=None,embedding=None ):
        pass
    
    @staticmethod
    def load_data(path=os.getcwd()+'/bnhp/'):
    # path='trainhebrew'
        pdf_loader = DirectoryLoader(path, glob="**/*.pdf",silent_errors = True ,recursive=True,show_progress=True)
        txt_loader = DirectoryLoader(path, glob="**/*.txt",silent_errors = True ,recursive=True,show_progress=True)
        word_loader = DirectoryLoader(path, glob="**/*.docx",silent_errors = True ,recursive=True,show_progress=True)
        ppt_loader = DirectoryLoader(path, glob="**/*.pptx",silent_errors = True ,recursive=True,show_progress=True)

        loaders = [ppt_loader,pdf_loader, txt_loader, word_loader]
        return loaders
    
    def create_sentences(segments, MIN_WORDS, MAX_WORDS):

            # Combine the non-sentences together
            sentences = []

            is_new_sentence = True
            sentence_length = 0
            sentence_num = 0
            sentence_segments = []

            for i in range(len(segments)):
                if is_new_sentence == True:
                    is_new_sentence = False
                    # Append the segment
                    sentence_segments.append(segments[i])
                    segment_words = segments[i].split(' ')
                    sentence_length += len(segment_words)
                    
                # If exceed MAX_WORDS, then stop at the end of the segment
                # Only consider it a sentence if the length is at least MIN_WORDS
                if (sentence_length >= MIN_WORDS and segments[i][-1] == '.') or sentence_length >= MAX_WORDS:
                    sentence = ' '.join(sentence_segments)
                    sentences.append({
                        'sentence_num': sentence_num,
                        'text': sentence,
                        'sentence_length': sentence_length
                    })
                    # Reset
                    is_new_sentence = True
                    sentence_length = 0
                    sentence_segments = []
                    sentence_num += 1

            return sentences

    def create_chunks(sentences, CHUNK_LENGTH, STRIDE):

        sentences_df = pd.DataFrame(sentences)
        
        chunks = []
        for i in range(0, len(sentences_df), (CHUNK_LENGTH - STRIDE)):
            chunk = sentences_df.iloc[i:i+CHUNK_LENGTH]
            chunk_text = ' '.join(chunk['text'].tolist())
            
            chunks.append({
            'start_sentence_num': chunk['sentence_num'].iloc[0],
            'end_sentence_num': chunk['sentence_num'].iloc[-1],
            'text': chunk_text,
            'num_words': len(chunk_text.split(' '))
            })
            
        chunks_df = pd.DataFrame(chunks)
        return chunks_df.to_dict('records')
    
    def parse_title_summary_results(results):
            out = []
            for e in results:
                e = e.replace('\n', '')
                if '|' in e:
                    processed = {'title': e.split('|')[0],
                                    'summary': e.split('|')[1][1:]
                                    }
                elif ':' in e:
                    processed = {'title': e.split(':')[0],
                                    'summary': e.split(':')[1][1:]
                                    }
                elif '-' in e:
                    processed = {'title': e.split('-')[0],
                                    'summary': e.split('-')[1][1:]
                                    }
                else:
                    processed = {'title': '',
                                    'summary': e
                                    }
                out.append(processed)
            return out

    
    @abstractmethod
    def checkdb(self):
        return False
    
    @abstractmethod
    def create_db(self,loaders):
        pass 

class Chromaindexdb(baseindexdb):
    def __init__(self,embedding=None,persist_directory = 'db') -> None:
        super().__init__(embedding,persist_directory)
        
    def getDefaultEmbadding(self): 
        return OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
   
    def getDb(self):
        return  Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding
        )
     
    def create_db(self,loaders,persist_directory=None,embedding=None ):   
        if persist_directory is None:
            persist_directory = self.persist_directory
        if embedding is None:
            embedding = self.embedding
            
        source_chunks = []
        for loader in loaders:  
            source_chunks.extend(self.create_doc_chunks_split(loader.load()))
        # source_chunks=create_doc_chunks_split(loaders.load())
        vectordb = Chroma.from_documents(source_chunks, embedding,persist_directory=persist_directory)
        vectordb.persist()
        return vectordb
    
    
    # persist_directory = 'feiss_db'
    # dbFaiss =FAISS.load_local(
    #     persist_directory, 
    #     getDefaultEmbadding()
    # )
    
    def checkdb(self):
        
        query = "איך אני לוקח משכנתא?"
        docs = self.getDb().similarity_search(query)
        return len(docs)>0

class Faissindexdb(baseindexdb):
    def __init__(self,embedding = None,persist_directory = 'feiss_db') -> None:
        super().__init__(embedding,persist_directory)
        
    def getDefaultEmbadding(self): 
        return OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
    
    def getDb(self):
        return FAISS.load_local(
            self.persist_directory, 
            self.embedding
        )
    
    
    
    def create_db(self,loaders,persist_directory=None,embedding=None ):   
        if persist_directory is None:
            persist_directory = self.persist_directory
        if embedding is None:
            embedding = self.embedding
            
        source_chunks = []
        for loader in loaders:  
            
            source_chunks.extend(self.create_doc_chunks_split(loader.load()))
        # source_chunks=create_doc_chunks_split(loaders.load())
        
        # vectordb = Chroma.from_documents(source_chunks, embedding,persist_directory=persist_directory)
        vectordb = FAISS.from_documents(source_chunks, embedding)
        # vectordb.save_local(folder_path: str, index_name: str = 'index') 
        vectordb.save_local(persist_directory)
        return vectordb

    def extend_db(self,loaders,persist_directory=None,embedding=None ):   
        if persist_directory is None:
            persist_directory = self.persist_directory
        if embedding is None:
            embedding = self.embedding
            
        source_chunks = []
        for loader in loaders:  
            
            source_chunks.extend(self.create_doc_chunks_split(loader.load()))
        # source_chunks=create_doc_chunks_split(loaders.load())
        
        # vectordb = Chroma.from_documents(source_chunks, embedding,persist_directory=persist_directory)
        vectordb = FAISS.load_local(
            self.persist_directory, 
            self.embedding
        )

        vectordb.from_documents(source_chunks, embedding)
        # vectordb.save_local(folder_path: str, index_name: str = 'index') 
        vectordb.save_local(persist_directory)
        return vectordb
    
    def checkdb(self):
        
        query = "איך אני לוקח משכנתא?"
        docs = self.getDb().similarity_search(query)
        print(docs)
        return len(docs)>0
    


class Complexindexdb(baseindexdb):
    def __init__(self,embedding = None,persist_directory = 'complex_db') -> None:
        super().__init__(embedding,persist_directory)
        
    def getDefaultEmbadding(self): 
        return OpenAIEmbeddings(
            openai_api_key=os.environ.get("OPENAI_API_KEY")
        )
    
    def getDb(self):
        return FAISS.load_local(
            self.persist_directory, 
            self.embedding
        )
    
   
    # retriever = FAISS.as_retriever(
    # search_type="similarity",
    # search_kwargs={"k": 4, "include_metadata": True})

    
    
    def create_db(self,loaders,persist_directory=None,embedding=None ):   
        if persist_directory is None:
            persist_directory = self.persist_directory
        if embedding is None:
            embedding = self.embedding

        langchain_documents=[]

        for loader in loaders:  
            try:
                data = loader.load()
                langchain_documents.extend(data)
            except Exception:
                continue

        vectordb = FAISS.from_documents(langchain_documents, embedding)
        # vectordb.save_local(folder_path: str, index_name: str = 'index') 
        vectordb.save_local(persist_directory)
        return vectordb
    
    def checkdb(self):
        
        query = "What did the president say about Justice Breyer"
        query = "מה היא משכנתאה?"
        query = "לירח"
        docs = self.getDb().similarity_search(query)
        print(len(docs),docs)
        return len(docs)>0
    



if __name__ == "__main__":
    
    
    # query = "איך אני לוקח משכנתא?"
    # print("Faissindexd",Faissindexdb(embedding=OpenAIEmbeddings()).checkdb())
    # print("Chromaindexdb",Chromaindexdb(embedding=OpenAIEmbeddings()).checkdb())
    # # embeddings = FakeEmbeddings(size=1352)
    # # fidb=Faissindexdb(embedding=embeddings)
    # # print(fidb.getDb().similarity_search(query))
    # fidb=Faissindexdb()
    # print(fidb.getDb().similarity_search(query))


    # cdb=Complexindexdb(embedding=OpenAIEmbeddings())
    # # cdb.create_db(loaders=Complexindexdb.load_data(path=os.getcwd()+'/datatoload/'))
    # print("Faissindexd",Complexindexdb(embedding=OpenAIEmbeddings()).checkdb())

    
    # cdb=Faissindexdb(embedding=OpenAIEmbeddings())
    # cdb.create_db(loaders=Faissindexdb.load_data(path=os.getcwd()+'/mashkanta/'))
    # print("Faissindexd",Faissindexdb(embedding=OpenAIEmbeddings()).checkdb())

    # query = "איך אני מגיע לירח"
    # cdb=Faissindexdb()
    # print(cdb.getDb().similarity_search(query))


    # cdb= Chromaindexdb(embedding=OpenAIEmbeddings())
    # cdb.create_db(loaders=Chromaindexdb.load_data(path=os.getcwd()+'/testdata/'))
    # print("Faissindexd",cdb.checkdb())

    cdb=Faissindexdb(embedding=OpenAIEmbeddings())
    cdb.extend_db(loaders=Faissindexdb.load_data(path=os.getcwd()+'/testdata/'))
    print("Faissindexd",Faissindexdb(embedding=OpenAIEmbeddings()).checkdb())

    # cdb=Faissindexdb(embedding=OpenAIEmbeddings())
    # cdb.create_db(loaders=Faissindexdb.load_data(path=os.getcwd()+'/bnhp_all_data/'))
    # print("Faissindexd",Faissindexdb(embedding=OpenAIEmbeddings()).checkdb())


    query = "איך אני מגיע לירח"
    # cdb=Faissindexdb()
    # cdb=Chromaindexdb()
    # answer=cdb.getDb().similarity_search(query)
    # query = "איך אני לוקח משכנתא?"
    # answer=cdb.getDb().similarity_search_with_relevance_scores(query)
    answer=cdb.getDb().similarity_search(query)
    print (answer)

        