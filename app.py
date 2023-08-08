#pip install langchain==0.0.150 pypdf pandas transformers openai faiss-cpu gdown flask flask_cors python-dotenv tiktoken
import os

import pandas as pd
from transformers import GPT2TokenizerFast
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os
import gdown

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

load_dotenv()

# api_key = os.getenv("OPENAI_API_KEY")
# os.environ["OPENAI_API_KEY"] = "sk-2LT"

api_key = os.getenv("OPENAI_API_KEY")
os.environ["OPENAI_API_KEY"] = api_key

url = "https://docs.google.com/document/d/1W_n0U2RsXtKan5Rcp-nWP1Tz6I2cN-ezvaLUIOkRlrg/export?format=txt"
output_path = "./document.txt"
gdown.download(url, output_path, quiet=False)
with open(output_path,'r', encoding='utf-8') as f:
    text = f.read()

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")


def count_tokens(text: str) -> int:
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 100,
    chunk_overlap  = 25,
    length_function = count_tokens,
)

chunks = text_splitter.create_documents([text])
type(chunks[0])

token_counts = [count_tokens(chunk.page_content) for chunk in chunks]
df = pd.DataFrame({'Token Count': token_counts})

embeddings = OpenAIEmbeddings()
db = FAISS.from_documents(chunks, embeddings)
query = "where did you study?"
docs = db.similarity_search(query)
docs[0]
chain = load_qa_chain(OpenAI(temperature=0.5), chain_type="stuff")
query = "where did you study?"
docs = db.similarity_search(query)
chain.run(input_documents=docs, question=query)

qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0.5), db.as_retriever())

@app.route('/api/chatbot', methods=['POST'])
def chatbot():
    data = request.get_json()
    query = data['query']
    history= data['history']
    
    if query.lower() == 'exit':
        return jsonify({'response': "Thank you for using the State of the Union chatbot!"})
    
    result = qa({"question": query+". Give very short precise answer.", "chat_history": history})
    # chat_history.append((query, result['answer']))
    
    return jsonify({'response': result['answer']})

if __name__ == '__main__':
    app.run()
