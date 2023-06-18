 #!/usr/bin/env python3

import textwrap
import gradio as gr
import os
from os.path import join, dirname
from dotenv import load_dotenv
from dotenv import load_dotenv
import llamaquestional
# from conversational import conversatioalChatGPT

# from conversation import create_conversation
# dotenv_path = join(dirname(__file__), '.env')
# print(load_dotenv(dotenv_path))
load_dotenv()



import gradio as gr
import sys
qa = llamaquestional.QuestionalLlama().getQuestional()

def qa_func( question):
    # return model.predict(context, question)["answer"]
    return qa.query(question)

samples = [
    ["איך אני לוקח משכנתא?",
     "מה זה ביטוח משכנתא?",
     "מה כדי ללמוד?"]
]

gr.Interface(qa_func, 
    [
        # gr.inputs.Textbox(lines=7, label="Context"), 
        gr.inputs.Textbox(label="Question"), 
    ], 
    gr.outputs.Textbox(label="Answer"),
    title="Ask Me Anything",
    examples=samples).launch(share=True,server_name=os.environ.get("GRADIO_SERVER_NAME"),server_port=int(os.getenv("GRADIO_SERVER_PORT",8888)))




# if __name__ == '__main__':
    
#     demo.queue(concurrency_count=3)
#     demo.launch(share=True,server_name=os.environ.get("GRADIO_SERVER_NAME"),server_port=int(os.getenv("GRADIO_SERVER_PORT",8888)))
