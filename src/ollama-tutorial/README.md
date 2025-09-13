# AI playground

### Select devcontainer 
- https://github.com/asantiola/developer-notes/tree/master/python/python-ai/gpu/.devcontainer
- https://github.com/asantiola/developer-notes/tree/master/python/python-ai/nogpu/.devcontainer

### create virtual env
- ctrl-shift-P, Python: Create Environment
- python3 -m venv .venv; source .venv/bin/activate;

### Packages
- pip install --upgrade pip
- pip install -r requirements.txt

### using ollama/ollama
- https://hub.docker.com/r/ollama/ollama
- https://github.com/ollama/ollama
```
docker compose -f docker-compose-ollama-gpu.yml create
docker compose -f docker-compose-ollama-gpu.yml start
docker compose -f docker-compose-ollama-gpu.yml stop
```

### Ollama API
- https://github.com/ollama/ollama/blob/main/docs/api.md
- list local models
    ```
    curl http://ollama:11434/api/tags
    ```
- list running models
    ```
    curl http://ollama:11434/api/ps
    ```
- pull llama3: 
    ```
    curl http://ollama:11434/api/pull -d '{"model": "llama3" }'
    ```
- sample call: 
    ```
    curl http://ollama:11434/api/generate -d '{ "model": "llama3", "prompt": "Why is the sky blue?" }'
    ```

### Vector database
- https://redis.io/solutions/vector-database/
- https://www.mongodb.com/resources/basics/databases/vector-databases
- https://www.timescale.com/learn/postgresql-extensions-pgvector
- https://python.langchain.com/docs/integrations/vectorstores/sklearn/
- https://stackoverflow.com/questions/79210867/is-there-a-way-to-load-a-saved-sklearn-vectorstore-using-langchain

### RAG With Llama 3.1 8B, Ollama, and Langchain: Tutorial
- reference: https://www.datacamp.com/tutorial/llama-3-1-rag?utm_source=google&utm_medium=paid_search&utm_campaignid=19589720821&utm_adgroupid=157098104375&utm_device=c&utm_keyword=&utm_matchtype=&utm_network=g&utm_adpostion=&utm_creative=726015683427&utm_targetid=dsa-2264919291989&utm_loc_interest_ms=&utm_loc_physical_ms=9062549&utm_content=&utm_campaign=230119_1-sea~dsa~tofu_2-b2c_3-row-p1_4-prc_5-na_6-na_7-le_8-pdsh-go_9-nb-e_10-na_11-na-jan25&gad_source=1&gclid=CjwKCAiAp4O8BhAkEiwAqv2UqMpEf0h0Eh8QIt-Vavc2NJ3R1sBqv_sMXyt2oVYrzIXnKb4nikXQ1xoCboEQAvD_BwE
- https://github.com/langchain-ai/langchain/discussions/24647
- https://medium.com/@sametarda.dev/deep-dive-into-corrective-rag-implementations-and-workflows-111c0c10b6cf
- https://huggingface.co/learn/cookbook/en/advanced_rag
- https://sj-langchain.readthedocs.io/en/latest/embeddings/langchain.embeddings.huggingface.HuggingFaceEmbeddings.html#langchain.embeddings.huggingface.HuggingFaceEmbeddings
- https://ollama.com/blog/embedding-models
- https://medium.com/hackademia/how-to-use-local-embedding-models-and-sentence-transformers-c0bf80a00ce2#:~:text=after%20the%20sentence%20transformer%20have,fetch%20and%20install%20the%20model.
- https://github.com/NirDiamant/RAG_Techniques

### rag/rag-sklearn-text.py
- Questions are taken from a text file
- Create documents at data/documents/*.txt.
- Create questions at data/questions.txt. Each line is a query.
- Create config at data/ollama_conf.json. 
    - Supported LLM models: llama3, mistral, llama3.2, phi3
    - Supported Embeddings models: sentence-transformers/all-mpnet-base-v2, thenlper/gte-small
- Web Pages:
    ```
    from langchain_community.document_loaders import WebBaseLoader
    WebBaseLoader(url).load()
    ```
- TextLoader:
    - pip install langchain langchain_community langchain-huggingface scikit-learn langchain-ollama
    ```
    from langchain_community.document_loaders import TextLoader
    TextLoader(file).load()
    ```
- Word Documents:
    - pip install unstructured python-docx
    - pip install spacy
    - python -m spacy downlfrom langchain_community.document_loaders
    - pip install NLTK
    ```
    import UnstructuredWordDocumentLoader
    import nltk
    nltk.download("punkt_tab")
    nltk.download("averaged_perceptron_tagger_eng")oad en_core_web_sm
    UnstructuredWordDocumentLoader(
        file, 
        mode="elements", 
        strategy="fast"
    ).load()
    ```
- PDF:
    ```
    from langchain_community.document_loaders import PyPDFLoader
    docs = [PyPDFLoader(
        file,
        # password = "my-pasword",
        # extract_images = True,
        # headers = None
        # extraction_mode = "plain",
        # extraction_kwargs = None,
    ).load() for file in files]
    ```

### rag/rag-sklearn-rest.py
- Implement a simple REST API to receive questions
- pip install flask
- while experimenting with array of questions
    - pip install flask-restful
    - seems to only work if array of questions are in 1 line like:
      ```
      {
        "questions": ["Who is X?", "Where is Y?"]
      }
      ```

### rag/rag-mongodb*.py - Using mongodb-atlas-local as vector store
- https://www.mongodb.com/docs/atlas/cli/current/atlas-cli-deploy-docker/
- https://www.mongodb.com/docs/atlas/atlas-vector-search/rag/
- https://www.mongodb.com/resources/products/fundamentals/examples
- https://www.mongodb.com/docs/mongodb-shell/connect/
- https://python.langchain.com/docs/integrations/vectorstores/mongodb_atlas/
- Based on rag01.py, Use mongodb-atlas-local as vector store
- pip install pymongo langchain_mongodb
- rag03-save.py
    - save vector store into mongodb-atlas-local
- open mongodb terminal - sample commands below
    - mongosh "mongodb://mongo_admin:mongo_passwd@localhost:27017"
        - show dbs
        - use rag_db
        - show collections
        - db.test.countDocuments()
        - db.test.findOne()
- rag03.py
    - assumes rag03-save.py ran successfully
    - use vector store example

### rag/rag-redis*py - Using redis-stack as vector store
- https://python.langchain.com/docs/integrations/vectorstores/redis/
- TODO: add authentication?
- pip install langchain_redis 
- rag04-save.py
    - pattern after rag03-save
    - save vector store into mongodb-atlas-local
- rag04.py 
    - assumes rag04-save.py ran successfully
    - use vector store example

### Agents
- https://www.anthropic.com/engineering/building-effective-agents
- https://github.com/anthropics/anthropic-cookbook/tree/main/patterns%2Fagents
- https://github.com/ollama/ollama-python
- https://ollama.com/blog/tool-support
- https://ollama.com/blog/structured-outputs
- https://github.com/ollama/ollama-python/blob/main/examples/tools.py
- https://python.langchain.com/docs/how_to/custom_tools/

### Prompt Chaining
- https://www.datacamp.com/tutorial/prompt-chaining-llm
- https://youtu.be/iDu0FB3nwig?si=AtKSVKg3IndJmHtR

### Structure Output
- https://github.com/ollama/ollama/issues/6314
- phi4: doesnt work.
- llama3.1: test OK
- mistral: partially OK??
