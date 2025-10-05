from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.tools import tool
import os
import sqlite3

# practice rag selection using a selector agent, with SQL agent

class MySqlite3Db:
    """A database object."""
    
    def __init__(self, **kwargs):
        self.conn = None
        self.cursor = None
        self.dbname = ""
        self.do_setup = False
        self.connect(**kwargs)
    
    def __del__(self):
        if self.conn == None:
            return
        
        self.conn.close()
    
    def get_dbtype(self):
        return "sqlite3"
    
    def get_dbname(self):
        return self.dbname

    def execute(self, sql):
        if self.conn == None:
            return
        
        if self.cursor == None:
            return
        
        self.cursor.execute(sql)
    
    def fetchall(self):
        if self.conn == None:
            return None
        
        if self.cursor == None:
            return None
        
        return self.cursor.fetchall()
    
    def execute_fetchall(self, sql: str):
        """
        Executes a SQL for this DB instance and returns the results.

        Args:
            sql: The SQL query to be done.
        """
        if self.conn == None:
            return None
        
        if self.cursor == None:
            return None
        
        self.cursor.execute(sql)
        return self.cursor.fetchall()

    def fetchone(self):
        if self.conn == None:
            return None
        
        if self.cursor == None:
            return None
        
        return self.cursor.fetchone()
    
    def fetchmany(self, size):
        if self.conn == None:
            return None
        
        if self.cursor == None:
            return None
        
        return self.cursor.fetchmany(size)
    
    def commit(self):
        self.conn.commit()

    def connect(self, **kwargs):
        if "dbname" in kwargs:
            self.dbname = kwargs["dbname"]
        if len(self.dbname) == 0:
            return
        
        if not os.path.exists(self.dbname):
            self.do_setup = True
        
        self.conn = sqlite3.connect(self.dbname)
        self.cursor = self.conn.cursor()
    
    def setup_tables(self):
        if self.conn == None:
            return
        
        if self.cursor == None:
            return
        
        if not self.do_setup:
            return
        self._populate_departments()
        self._populate_employees()
        self._populate_contacts()
    
    def print_tables(self):
        if self.conn == None:
            return
        
        if self.cursor == None:
            return
        
        self.cursor.execute("select * from departments")
        response = self.cursor.fetchall()
        print(f"departments: {response}")    

        self.cursor.execute("select * from employees")
        response = self.cursor.fetchall()
        print(f"employees: {response}")  

        self.cursor.execute("select * from contacts")
        response = self.cursor.fetchall()
        print(f"contacts: {response}\n")
    
    def _populate_departments(self):
        self.cursor.execute("""CREATE TABLE IF NOT EXISTS departments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL)
            """)

        self.cursor.execute("INSERT INTO departments VALUES (1, 'IT');")
        self.cursor.execute("INSERT INTO departments VALUES (2, 'HR');")
        self.cursor.execute("INSERT INTO departments VALUES (3, 'Marketing');")
        self.cursor.execute("INSERT INTO departments VALUES (4, 'Finance');")

        self.conn.commit()

    def _populate_employees(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS employees (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                department_id INTEGER,
                name TEXT NOT NULL,
                salary REAL,
                FOREIGN KEY (department_id) REFERENCES departments(id)
            )
            """)

        self.cursor.execute("INSERT INTO employees VALUES(1, 1, 'Alex', 10000);")
        self.cursor.execute("INSERT INTO employees VALUES(2, 1, 'John', 9000);")
        self.cursor.execute("INSERT INTO employees VALUES(3, 1, 'Mary', 9500);")
        self.cursor.execute("INSERT INTO employees VALUES(4, 1, 'Joseph', 9700);")
        self.cursor.execute("INSERT INTO employees VALUES(5, 1, 'Jane', 9200);")
        self.cursor.execute("INSERT INTO employees VALUES(6, 2, 'Monique', 8000);")
        self.cursor.execute("INSERT INTO employees VALUES(7, 2, 'Owen', 8100);")
        self.cursor.execute("INSERT INTO employees VALUES(8, 3, 'Fred', 7200);")
        self.cursor.execute("INSERT INTO employees VALUES(9, 3, 'Michelle', 8100);")
        self.cursor.execute("INSERT INTO employees VALUES(10, 3, 'Janice', 9800);")

        self.conn.commit()

    def _populate_contacts(self):
        self.cursor.execute("""
            CREATE TABLE IF NOT EXISTS contacts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                employee_id INTEGER,
                phone TEXT,
                email TEXT,
                address TEXT,
                FOREIGN KEY (employee_id) REFERENCES employees(id)
            )
            """)

        self.cursor.execute("INSERT INTO contacts VALUES(1, 2, '1234-5678', 'john@email.com', 'binondo, manila');")
        self.cursor.execute("INSERT INTO contacts VALUES(2, 8, '1111-2222', 'fred@email.com', 'ermita, manila');")
        self.cursor.execute("INSERT INTO contacts VALUES(3, 10, '3456-1234', 'janice@email.com', 'makati, manila');")
        self.cursor.execute("INSERT INTO contacts VALUES(4, 7, NULL, 'owen@email.com', NULL);")

        self.conn.commit()

def agent_paraphraser(llm, input, format):
    prompt = PromptTemplate(
        template="""You are a helpful assistant that paraphrase the input into sentence form.
            Take note of the input description to help in your conversion.
            Input: {input}
            Input format: {format}
            """,
        input_variables=["input", "format",],
    )
    chain = (prompt | llm)
    response = chain.invoke({
        "input": input,
        "format": format,
    })
    return response.content

def agent_tableinfo(llm: ChatOpenAI, db: MySqlite3Db):
    format = "[table1, (column1, column2, ...), table2, (column1, column2, ...), ...]"
    prompt = PromptTemplate(
        template="""You are a helpful assistant, with expertise in {dbtype} database.
            Given a database, generate a SQL that can be called with the tool.
            No need for explanation, call the tool with the SQL.
            The SQL query provide me information in the following format:
            {format}
            """,
        input_variables=["dbtype", "format"],
    )
    db_execute_fetchall = tool(db.execute_fetchall)

    tool_list = [
        db_execute_fetchall,
    ]

    tools_map = {
        "execute_fetchall": db_execute_fetchall,
    }
    
    chain = (prompt | llm.bind_tools(tool_list))
    response = chain.invoke({
        "dbtype": db.get_dbtype(),
        "format": format,
    })

    info = ""
    for tool_call in response.tool_calls:
        if function_to_call := tools_map.get(tool_call["name"]):
            tool_response = function_to_call.invoke(tool_call["args"])
            info += " " + agent_paraphraser(llm=llm, input=tool_response, format=format)
    
    return info

embeddings_model = "ai/mxbai-embed-large"
HOME=os.environ["HOME"]

oa_embeddings = OpenAIEmbeddings(
    model=embeddings_model,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
    # disable check_embedding_ctx_length if your local model has different constraints
    check_embedding_ctx_length=False,
)

def create_retriever(doc_path):
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
        embedding=oa_embeddings,
    )

    return vector_store.as_retriever()

retriever_billiards = create_retriever(HOME + "/repo/playground-ai-ml/data/routing-txt/billiards")
retriever_guitars = create_retriever(HOME + "/repo/playground-ai-ml/data/routing-txt/guitars")
retriever_technologies = create_retriever(HOME + "/repo/playground-ai-ml/data/routing-txt/technologies")
retrievers = (None, retriever_billiards, retriever_guitars, retriever_technologies)

llm = ChatOpenAI(
    model="ai/gpt-oss:latest",
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

selection_prompt = PromptTemplate(
    template="""You are helpful assistant that analyzes questions.
        You only return integers: 0, 1, 2, 3.
        If the question is not related to Lex, return 0.
        Else if the question is related to billiards, return 1.
        Else if the question is related to guitars, return 2.
        Else if the question is related to software engineering or programming, return 3.
        Otherwise, if it doesn't match any of these topics, return 0.
        Question: {question}
        """,
    input_variables=["question"]
)

chain = (selection_prompt | llm)

def agent_selector(question: str):
    response=chain.invoke({ "question": question})
    return retrievers[int(response.content)]

rag_prompt = PromptTemplate(
    template="""You are an assistant helping answer questions.
        Analyze the given question and the database information.
        If database has the needed tables and columns for the questions, execute the tool with a SQL to get the answer.
        You may use joins and unions in your SQL query if needed.
        If you cannot use the database, use your training data and augmented by the context to answer.
        Format your answer in setence form.
        Question: {question}
        Database information: {dbinfo}
        Context: {context}
        """,
    input_variables=["question", "dbinfo", "context"],
)

def agent_expert(dbinfo: str, db: MySqlite3Db, question: str):
    db_execute_fetchall = tool(db.execute_fetchall)

    tool_list = [
        db_execute_fetchall,
    ]

    tools_map = {
        "execute_fetchall": db_execute_fetchall,
    }
    
    rag_chain = (
        { "context": agent_selector, "question": RunnablePassthrough(), "dbinfo": RunnablePassthrough() } | 
        rag_prompt | 
        llm.bind_tools(tool_list)
    )
    
    response = rag_chain.invoke(question, { "dbinfo": dbinfo})
    
    if len(response.content) > 0:
        print(f"question: {question}\nanswer: {response.content}\n")
        return
    
    format = f"The input is SQL query output to the question {question}."
    info = ""
    for tool_call in response.tool_calls:
        if function_to_call := tools_map.get(tool_call["name"]):
            tool_response = function_to_call.invoke(tool_call["args"])
            info += " " + agent_paraphraser(llm=llm, input=tool_response, format=format)
    
    print(f"question: {question}\nanswer: {info}\n")

dbname = HOME + "/repo/playground-ai-ml/data/sql-rag-routing.db"

try:
    db = MySqlite3Db(dbname=dbname)

    db.setup_tables()
    db.print_tables()
    dbinfo = agent_tableinfo(llm=llm, db=db)

    agent_expert(dbinfo=dbinfo, db=db, question="Who are the top three highest earning employees, and what are their salaries?")
    agent_expert(dbinfo, db, "What are Lex's break cues?")
    agent_expert(dbinfo, db, "Where is Cebu City?")
    agent_expert(dbinfo, db, "Among the employees with contact information, who has the highest salary and what is the email address?.")
    agent_expert(dbinfo, db, "What is the largest bone in the human body?")
except sqlite3.Error as e:
    print(f"Error caught: {e}")
    exit(1)

# # results using gpt-oss
# departments: [(1, 'IT'), (2, 'HR'), (3, 'Marketing'), (4, 'Finance')]
# employees: [(1, 1, 'Alex', 10000.0), (2, 1, 'John', 9000.0), (3, 1, 'Mary', 9500.0), (4, 1, 'Joseph', 9700.0), (5, 1, 'Jane', 9200.0), (6, 2, 'Monique', 8000.0), (7, 2, 'Owen', 8100.0), (8, 3, 'Fred', 7200.0), (9, 3, 'Michelle', 8100.0), (10, 3, 'Janice', 9800.0)]
# contacts: [(1, 2, '1234-5678', 'john@email.com', 'binondo, manila'), (2, 8, '1111-2222', 'fred@email.com', 'ermita, manila'), (3, 10, '3456-1234', 'janice@email.com', 'makati, manila'), (4, 7, None, 'owen@email.com', None)]
# question: Who are the top three highest earning employees, and what are their salaries?
# answer:  Alex earns $10,000, Janice earns $9,800, and Joseph earns $9,700, making them the top three highest‑earning employees.
# question: What are Lex's break cue?
# answer: Lex owns two break‑cues: an **Action ACT 56 Break and Jump cue** (black metallic stained Birdseye Maple sleeve, 29″ Hardrock Maple 10‑12″ Pro Taper, 13 mm phenolic tip) and a **Mezz Power Break Kai** (Deep Impact 2 shaft, 13 mm tip, composite wood forearm and butt sleeve).
# question: Where is Cebu City?
# answer: Cebu City is located in the Philippines, on the island of Cebu in the Central Visayas region. It serves as the capital of Cebu province.
# question: Among the employees with contact information, who has the highest salary and what is the email address?.
# answer:  Janice, whose email address is janice@email.com, has the highest salary among employees with contact information.
# question: What is the largest bone in the human body?
# answer: The largest bone in the human body is the femur, the thigh bone.
