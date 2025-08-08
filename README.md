# HuggingFaceChat02Rag

This is a simple example of a chatbot using the HuggingFace Chat API and a local vector database.

## Steps to build index then prompt
1. run `python build_index.py`
2. run `python rag_chat_prompt.py "Your question here"`
- optional: add -k 50 to retrieve more context
- optional: add --debug to print retrieved context
- example: `python rag_chat_prompt.py "What has my sleep been like for the past week?" -k 50 --debug`
3. copy the prompt from the output
4. paste the prompt into your chat model


## Steps to perform RAG techinique

### 1.  ingest data
- load the CSV file containing raw data
- format each row into a short text "document" for processing
- occurs in `build_index.py`

### 2. 	Embed data
- use a small local embedding model to generate vectors
- convert each document into its vector representation
- occurs in `build_index.py`

### 3. 	Index
- store the generated vectors in a FAISS index file.
- occurs in `build_index.py`

### 4.	Retrieve
In this step we actually interact with the model.
- embed the user query using the same embedding model
- perform a k-NN search in the FAISS index to find relevant documents
- send the retrieved context to the Friendli endpoint for response generation
- occurs in `rag_chat.py`

## installation

```bash
pip install pandas faiss-cpu sentence-transformers python-dotenv requests
```

## Set up

1. Create a `.env` file in the root directory 
2. create folder for context data - this stores the csv raw data for context
3. create folder for index data - this stores the faiss index and docs.json

