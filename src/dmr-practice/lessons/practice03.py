import sqlite3
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate

def create_tables(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        )
    ''')

    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_id INTEGER,
            name TEXT NOT NULL,
            salary REAL,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        )
    ''')
    conn.commit()

def populate_departments(cursor):
    cursor.execute("INSERT INTO departments VALUES (1, 'IT');")
    cursor.execute("INSERT INTO departments VALUES (2, 'HR');")
    cursor.execute("INSERT INTO departments VALUES (3, 'Marketing');")
    cursor.execute("INSERT INTO departments VALUES (4, 'Finance');")

def populate_employees(cursor):
    cursor.execute("INSERT INTO employees VALUES(1, 1, 'Alex', 10000);")
    cursor.execute("INSERT INTO employees VALUES(2, 1, 'John', 9000);")
    cursor.execute("INSERT INTO employees VALUES(3, 1, 'Mary', 9500);")
    cursor.execute("INSERT INTO employees VALUES(4, 1, 'Joseph', 9700);")
    cursor.execute("INSERT INTO employees VALUES(5, 1, 'Jane', 9200);")
    
    cursor.execute("INSERT INTO employees VALUES(6, 2, 'Monique', 8000);")
    cursor.execute("INSERT INTO employees VALUES(7, 2, 'Owen', 8100);")

    cursor.execute("INSERT INTO employees VALUES(8, 3, 'Fred', 7200);")
    cursor.execute("INSERT INTO employees VALUES(9, 3, 'Michelle', 8100);")
    cursor.execute("INSERT INTO employees VALUES(10, 3, 'Janice', 9800);")

def get_tables_columns(cursor):
    cursor.execute('''
        SELECT name FROM sqlite_master WHERE type='table'
    ''')
    tables = cursor.fetchall()
    info = ""
    for table_name_tuple in tables:
        table_name = table_name_tuple[0]

        cursor.execute(f"PRAGMA table_info(\'{table_name}\')")
        columns = cursor.fetchall()

        column_names = [column_name_tuple[1] for column_name_tuple in columns]
        info += f"table name: {table_name}, column names: {column_names}. "
    return info

def convert_to_vector_store(info):
    embeddings_model = "ai/mxbai-embed-large"

    oa_embeddings = OpenAIEmbeddings(
        model=embeddings_model,
        base_url="http://localhost:12434/engines/v1",
        api_key="docker",
        # disable check_embedding_ctx_length if your local model has different constraints
        check_embedding_ctx_length=False,
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
    )
    splits = text_splitter.split_text(info)

    vector_store = Chroma.from_texts(
        texts=splits,
        embedding=oa_embeddings,
    )
    return vector_store

llm = ChatOpenAI(
    model="ai/llama3.1",
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

def generate_sql(vector_store, question):
    prompt = PromptTemplate(
        template="""You are an assistant that generates SQL code for sqlite3 database.
            Use the documents to generate the SQL that will answer the question.
            Running the SQL should generate a result for a human reader.
            Return the resulting SQL without any extra information or text.
            Question: {question} 
            Documents: {documents}  
            """,
        input_variables=["question", "documents"],
    )
    retriever = vector_store.as_retriever()
    rag_chain = prompt | llm
    documents = retriever.invoke(question)
    answer = rag_chain.invoke({
        "documents": documents,
        "question": question,
    })
    return answer.content

def run_sql(cursor, sql):
    cursor.execute(sql)
    result = cursor.fetchall()
    return result

def setup(cursor):
    create_tables(cursor)
    populate_departments(cursor)
    populate_employees(cursor)
    conn.commit()

def print_tables(cursor):
    cursor.execute("select * from departments")
    res = cursor.fetchall()
    print(f"departments: {res}")

    cursor.execute("select * from employees")
    res = cursor.fetchall()
    print(f"employees: {res}")

def ask_question(cursor, question):
    info = get_tables_columns(cursor)
    vector_store = convert_to_vector_store(info)
    sql = generate_sql(vector_store=vector_store, question=question)
    return run_sql(cursor=cursor, sql=sql)

try:
    conn = sqlite3.connect("/Users/asantiola/repo/playground-ai-ml/data/practice03.db")
    cursor = conn.cursor()
    # setup(cursor=cursor)

    print_tables(cursor=cursor)

    result = ask_question(question="What is the department with the largest number of employees?", cursor=cursor)
    print(f"result: {result}")

    result = ask_question(question="Give me the top 3 employees based on salary.", cursor=cursor)
    print(f"result: {result}")
except sqlite3.Error as e:
    print(f"Error caught: {e}")
finally:
    conn.close()
