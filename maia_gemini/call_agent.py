import argparse
import openai
from io import BytesIO
import base64
import pandas as pd
import PIL.Image
from tqdm import tqdm
from IPython import embed
import time
from random import random, uniform
import warnings
from vertexai.preview.generative_models import Part, Content
import os
import requests
import ast
import dotenv
dotenv.load_dotenv('.env')
warnings.filterwarnings("ignore")

# User inputs:
# Load your API key from an environment variable or secret management service
# openai.api_key = os.getenv("OPENAI_API_KEY")
# OR 
# Load your API key manually:
# openai.api_key = API_KEY

#openai.api_key = os.getenv("OPENAI_API_KEY")
#openai.organization = os.getenv("OPENAI_ORGANIZATION")

import google.generativeai as genai

genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))

def ask_agent(model, history):
    count = 0
    try:
        count+=1
        if model in ['gpt-4-vision-preview','gpt-4-turbo', 'gemini-1.5-flash']:
            '''params = {
            "model": model,
            "messages": history,
            "max_tokens": 4096,
            }'''
            #r = openai.ChatCompletion.create(**params)
            '''
            text_part = Part.from_text("Why is sky blue?")
            image_part = Part.from_image(Image.load_from_file("image.jpg"))
            '''
            '''text_part = Content(
                role='user',
                parts=[
                    Part.from_text('System: ' + history[0]['content'][0]['text'] + ", Assistant: Understood, User: " + history[1]['content'][0]['text']),
                ]
            )'''
            #text = ''
            prompt = []
            for exp in history:
                prompt += [exp['role'] + ": "]
                for content in exp['content']:
                    if content['type']=='text':
                        prompt += [content[content['type']]]
                    elif content['type']=='image_url':
                        data = content[content['type']]
                        decoded_bytes = base64.b64decode(data.replace('\\n', ''))
                        im = PIL.Image.open(BytesIO(decoded_bytes))
                        prompt += [im]
                #text = 'System: ' + history[0]['content'][0]['text'] + ", Assistant: Understood, User: " + history[1]['content'][0]['text']
            model = genai.GenerativeModel(model_name="models/gemini-1.5-flash")
            response = model.generate_content(prompt)
            resp = response.text #r['choices'][0]['message']['content']
        else:
            print(f"Unrecognize model name: {model}")
            return 0
    except Exception as e:
        print('openai-api error: ',e)
        if ((e.http_status==429) or (e.http_status==502) or (e.http_status==500)) :
            time.sleep(60+10*random()) # wait for 60 seconds and try again
            if count < 25:
                resp = ask_model(model,history)
            else: return e
        elif (e.http_status==400): # images cannot be process due to safty filter
            if (len(history) == 4) or (len(history) == 2): # if dataset exemplars are unsafe
                return e
            else: # else, remove the lase experiment and try again
                resp = ask_model(model,history[:-2])
    return resp