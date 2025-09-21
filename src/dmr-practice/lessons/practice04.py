import os
import sqlite3
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field

def populate_departments(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL
        )
    ''')

    cursor.execute("INSERT INTO departments VALUES (1, 'IT');")
    cursor.execute("INSERT INTO departments VALUES (2, 'HR');")
    cursor.execute("INSERT INTO departments VALUES (3, 'Marketing');")
    cursor.execute("INSERT INTO departments VALUES (4, 'Finance');")

def populate_employees(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS employees (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            department_id INTEGER,
            name TEXT NOT NULL,
            salary REAL,
            FOREIGN KEY (department_id) REFERENCES departments(id)
        )
    ''')

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

def populate_contacts(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS contacts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            employee_id INTEGER,
            phone TEXT,
            email TEXT,
            address TEXT,
            FOREIGN KEY (employee_id) REFERENCES employees(id)
        )
    ''')

    cursor.execute("INSERT INTO contacts VALUES(1, 2, '1234-5678', 'john@email.com', 'binondo, manila');")
    cursor.execute("INSERT INTO contacts VALUES(2, 8, '1111-2222', 'fred@email.com', 'ermita, manila');")
    cursor.execute("INSERT INTO contacts VALUES(3, 10, '3456-1234', 'janice@email.com', 'makati, manila');")
    cursor.execute("INSERT INTO contacts VALUES(4, 7, NULL, 'owen@email.com', NULL);")

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

class SqlQuery(BaseModel):
    sql_query: str = Field(
        description="The generated SQL query."
    )
    score: float = Field(
        description="Confidence score on the scale of 0 to 1 on your answer."
    )

def generate_sql(retriever, question) -> SqlQuery:
    prompt = PromptTemplate(
        template="""You are a database expert that generates SQL query for sqlite3 database.
            Your response should be a JSON string with 2 fields: sql_query and score.
            Analyze the question and documents to generate the SQL query into sql_query field.
            For the SQL query, you can perform table joins if needed.
            Do not break the SQL string into multiple lines.
            No need to enclose the JSON in backticks or quotes.
            Provide a confidence score on the scale of 0 to 1 on the generated SQL and relevance to the question.
            Question: {question} 
            Documents: {documents}  
            """,
        input_variables=["question", "documents"],
    )
    rag_chain = (
        {"documents": retriever, "question": lambda x: x} 
        | prompt 
        | llm.with_structured_output(SqlQuery)
    )
    sql_query = rag_chain.invoke(question)
    # print(f"sql: {sql_query}")
    return sql_query

def format_answer(question, sql_response):
    prompt = PromptTemplate(
        template="""You are an assistant that formats SQL result from sqlite3 database.
            If you think the response is valid, you can revise it so it is expresed in in proper sentences, no need for explanations.
            If you think the response is invalid, you can say that you have insufficient information to answer the question.
            Question: {question} 
            SQL Response: {sql_response}  
            """,
        input_variables=["question", "sql_response"],
    )
    rag_chain = prompt | llm
    answer = rag_chain.invoke({
        "question": question,
        "sql_response": sql_response,
    })
    return answer.content

def run_sql(cursor, sql):
    cursor.execute(sql)
    result = cursor.fetchall()
    return result

def setup(cursor):
    populate_departments(cursor)
    populate_employees(cursor)
    populate_contacts(cursor)

def print_tables(cursor):
    cursor.execute("select * from departments")
    res = cursor.fetchall()
    print(f"departments: {res}")

    cursor.execute("select * from employees")
    res = cursor.fetchall()
    print(f"employees: {res}")

    cursor.execute("select * from contacts")
    res = cursor.fetchall()
    print(f"contacts: {res}")

    print("")

def ask_question(cursor, question):
    info = get_tables_columns(cursor)
    vector_store = convert_to_vector_store(info)
    sql_query = generate_sql(retriever=vector_store.as_retriever(), question=question)
    sql_response = ""
    if sql_query.score > 0.7:
        sql_response = run_sql(cursor=cursor, sql=sql_query.sql_query)
    answer = format_answer(question=question, sql_response=sql_response)
    print(f"\nquestion: {question}\nanswer: {answer}\n")

db_name = "/Users/asantiola/repo/playground-ai-ml/data/practice04.db"
do_setup = False
try:
    do_setup = not os.path.exists(db_name)
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    if do_setup:
        setup(cursor=cursor)
        conn.commit()
    
    print_tables(cursor=cursor)

    question="Which department has the most number of employees, and how many employees does it have?"
    ask_question(question=question, cursor=cursor)

    question="Give me the top 3 employees based on salary."
    ask_question(question=question, cursor=cursor)

    question="Give me the email of the employee with highest salary and has a contact information."
    ask_question(question=question, cursor=cursor)

    question="What is the largest bone in the human body?"
    ask_question(question=question, cursor=cursor)
except sqlite3.Error as e:
    print(f"Error caught: {e}")
finally:
    conn.close()
