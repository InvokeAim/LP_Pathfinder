# first AI try
import sys
from openai import OpenAI
import openai
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


client = OpenAI(organization='org-tpDC2uJAfAPeuh171S7EW8nr',
                project='proj_mAgiJfHZ8zFbuYQsYpFMFJ8Y')

def chat():
    print("Start chatting with the AI. Type 'exit' to end the session.")

    # initialize the AI behavior
    messages = [
        {"role": "system", 
         "content": "You are a helpful asistant."},
         {"role": "system",
          "content": read_user_file()}
    ]
    memorize_user_input = ''

    while True:
        user_input = input('You: ')
        messages.append({"role": "user",
                         "content": user_input})
        memorize_user_input += f'{user_input}\n'
        
        if user_input.lower() == 'exit':
            print("Ending the chat session!")
            summarize_user_info(memorize_user_input)
            break
        
        try:
            completion = client.chat.completions.create(
                model='gpt-4o-mini',
                messages=messages
            )

            assistant_message = completion.choices[0].message.content
            print('AI:', assistant_message)
            print("Usage:", completion.usage.total_tokens)
        except openai.RateLimitError as e:
            print(f"Rate limit exceeded: {e}")
        except openai.OpenAIError as e:
            print(f'An error occurred: {e}')


def summarize_user_info(messages):
    messages = (f'Summarize the following user information in a concise manner: {messages}'
                f'Format it into yaml'
    )
    completion = client.chat.completions.create(
        model='gpt-4o-mini',
        messages=[
            {"role": "user",
             "content": messages}
                ]
    )
    summary = completion.choices[0].message.content
    with open('/home/aim/LP/users/aim.info', 'a') as fhand:
        fhand.write(summary)

def get_embedding(text, model="text-embedding-3-small"):
    response = openai.embeddings.create(
        input=text,
        model=model
    )
    return response.data[0].embedding

def compare_embeddings(emb1, emb2):
    emb1 = np.array(emb1).reshape(1, -1)
    emb2 = np.array(emb2).reshape(1, -1)
    return cosine_similarity(emb1, emb2)[0][0]

def read_user_file():
    with open('/home/aim/LP/users/aim.info', 'r') as fhand:
        return fhand.read()

if __name__ == "__main__":
    chat()
