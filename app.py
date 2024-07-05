from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
import os
import torch
# Determine if a GPU is available
device = "cuda" if torch.cuda.is_available() else "cpu"
print(torch.cuda.is_available()) 
app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')


embeddings = download_hugging_face_embeddings()

#Initializing the Pinecone
pinecone.init(api_key=PINECONE_API_KEY,
              environment=PINECONE_API_ENV)

index_name="medical-bot"

#Loading the index
docsearch=Pinecone.from_existing_index(index_name, embeddings)


PROMPT=PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs={"prompt": PROMPT}

llm=CTransformers(model="model/llama-2-7b-chat.ggmlv3.q4_0.bin",
                  model_type="llama",
                  config={'max_new_tokens':512,
                          'temperature':0.8})


qa=RetrievalQA.from_chain_type(
    llm=llm, 
    chain_type="stuff", 
    retriever=docsearch.as_retriever(search_kwargs={'k': 2}),
    return_source_documents=True, 
    chain_type_kwargs=chain_type_kwargs)



@app.route("/")
def index():
    return render_template('chat.html')



@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result=qa({"query": input})
    print("Response : ", result["result"])
    return str(result["result"])

@app.route('/upload', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the POST request has the file part
        if 'file' not in request.files:
            return render_template('upload.html', message='No file part')

        file = request.files['file']

        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return render_template('upload.html', message='No selected file')

        if file:
            # Here, you would connect to the PicOne Vector Database API
            # and upload the file. This part will depend on the API
            # documentation provided by PicOne.

            # Placeholder for PicOne Vector Database API upload code
            # Replace this with actual API upload code
            # upload_to_picone_database(file)

            # Provide feedback to the user
            return render_template('upload.html', message='File uploaded successfully')

    return render_template('upload.html')

# if __name__ == '__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    app.run(host="0.0.0.0", port= 1500, debug= True)


