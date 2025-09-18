from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import os

embeddings_model = "thenlper/gte-small"

# Warning seen: huggingface/tokenizers: The current process just got forked, after parallelism has already been used. Disabling parallelism to avoid deadlocks...
# To disable this warning, you can either:
# - Avoid using `tokenizers` before the fork if possible
# - Explicitly set the environment variable TOKENIZERS_PARALLELISM=(true | false)
os.environ["TOKENIZERS_PARALLELISM"] = ""

hf_embeddings = HuggingFaceEmbeddings(
    cache_folder="/Users/asantiola/repo/playground-ai-ml/.cache",
    model_name=embeddings_model,
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": False},
)

doc_path = "/Users/asantiola/repo/playground-ai-ml/data/documents-txt"
documents = []
for filename in os.listdir(doc_path):
    if filename.endswith('.txt'):
        file_path = os.path.join(doc_path, filename)
        loader = TextLoader(file_path)
        documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
)
splits = text_splitter.split_documents(documents=documents)

vector_store = Chroma.from_documents(
    documents=splits,
    embedding=hf_embeddings,
)

prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
        If you don't know the answer, just say that you don't know. 
        Use three sentences maximum and keep the answer concise:
        Question: {question} 
        Documents: {documents} 
        Answer: 
        """,
    input_variables=["question", "documents"],
)

llm = ChatOpenAI(
    model="ai/phi4",
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

retriever = vector_store.as_retriever()
rag_chain = prompt | llm

def query(question):
    print(f"Question: \"{question}\"\n")
    documents = retriever.invoke(question)
    answer = rag_chain.invoke({
        "documents": documents,
        "question": question,
    })
    print(f"Answer: {answer.content}\n\n")

questions = "/Users/asantiola/repo/playground-ai-ml/data/questions.txt"
with open(questions) as file:
    for line in file:
        query(line.rstrip())
