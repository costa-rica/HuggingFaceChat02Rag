# HuggingFaceChat02Rag

This is a simple example of a chatbot using the HuggingFace Chat API and a local vector database.


## Steps
### 1.  ingest data
- load data from a file
- load the CSV file containing raw data
- format each row into a short text "document" for processing
- run: `python build_index.py`
- output: creates faiss.index and docs.json in the index directory
### 2. 	Embed data
- use a small local embedding model to generate vectors
- cache the embedding model locally for faster access
- convert each document into its vector representation

### 3. 	Index
- store the generated vectors in a FAISS index
- save the FAISS index to disk for persistence
- prepare the index for efficient similarity search

### 4.	Retrieve
- embed the user query using the same embedding model
- perform a k-NN search in the FAISS index to find relevant documents
- send the retrieved context to the Friendli endpoint for response generation