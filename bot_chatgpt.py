 #!/usr/bin/env python3

import gradio as gr
import os
from os.path import join, dirname
from dotenv import load_dotenv
from dotenv import load_dotenv
from sklearn import externals
import conversational

# from conversational import conversatioalChatGPT

# from conversation import create_conversation
# dotenv_path = join(dirname(__file__), '.env')
# print(load_dotenv(dotenv_path))
load_dotenv()
print ("OPENAI_API_KEY",os.environ.get("OPENAI_API_KEY"))

# CONDENSE_TEMPLATE = """Given the following chat history and a follow up question, rephrase the follow up input question to be a standalone question.Or end the conversation if it seems like it's done.Chat History:\"""{chat_history} \"""Follow Up Input: \"""{question}\"""Standalone question:"""
# QA_TEMPLATE = """You are a friendly, conversational banking assistant. Talk about your  data  and answer any questions.It's ok if you don't know the answer.Context:\"""{context}\"""Question:\"{question}\"""Helpful Answer :"""
      

# OPENAI_API_KEY = "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"
# HUGGINGFACEHUB_API_TOKEN = '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'
# server_name="0.0.0.0"
#getpass('Enter your OpenAI key: ')
# os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACEHUB_API_TOKEN
# os.environ["GRADIO_SERVER_NAME"] = server_name
# os.environ["GRADIO_SERVER_PORT"] = "8888"

qa = conversational.conversatioalChatGPT().getConversational()

def create_conversational():
    # externals (qa)
    qa = conversational.conversatioalChatGPT().getConversational()


create_conversational()


def add_text(history, text):
    history = history + [(text, None)]
    return history, ""


def bot(history):
    question=history[-1][0]
    chat_history=history[:-1]
    res=qa(
                {
                    'question': question
                    # 'chat_history': []
                }
            )
          
    print("res",res)
    history[-1][1] = res['answer']
    return history


with gr.Blocks() as demo:
    
    chatbot = gr.Chatbot([], elem_id="chatbot",
                         label='Document GPT').style(height=750)
    
    with gr.Row():
        with gr.Column(scale=0.80):
            txt = gr.Textbox(
                show_label=False,
                placeholder="Enter text and press enter",
            ).style(container=False)
        with gr.Column(scale=0.10):
            submit_btn = gr.Button(
                'Submit',
                variant='primary'
            )
        with gr.Column(scale=0.10):
            clear_btn = gr.Button(
                'Clear',
                variant='stop'
            )
            
   
            

    txt.submit(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )

    submit_btn.click(add_text, [chatbot, txt], [chatbot, txt]).then(
        bot, chatbot, chatbot
    )
    # create_conversational_btn.click(create_conversational, [CONDENSE_TEMPLATE_txt, QA_TEMPLATE_txt], [CONDENSE_TEMPLATE_txt, QA_TEMPLATE_txt]).then(
    #     lambda: None, None, chatbot, queue=False
    # ).then (
    #     bot, chatbot, chatbot
    # )

    clear_btn.click(lambda: None, None, chatbot, queue=False).then(
        create_conversational
    )

if __name__ == '__main__':
    
    demo.queue(concurrency_count=3)
    demo.launch(share=True,server_name=os.environ.get("GRADIO_SERVER_NAME"),server_port=int(os.getenv("GRADIO_SERVER_PORT",8888)))
