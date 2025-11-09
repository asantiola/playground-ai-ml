import os
import sqlite3
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import tool

# practice code agentic ai
# OOP approach of sql-agentic-ai-01.py

HOME=os.environ["HOME"]

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

def agent_expert(llm: ChatOpenAI, dbinfo: str, db: MySqlite3Db, question: str):
    prompt = PromptTemplate(
        template="""You are a helpful assistant that helps answer questions.
            Assess the given question.
            If the database has enough to answer it, execute the tool with a SQL to get the answer.
            You can use joins and unions in your SQL.
            Otherwise, you do not have to call the tool, you can answer the question based on your training data.
            Question: {question}
            Database information: {dbinfo}
            """,
        input_variables=["question", "dbinfo",],
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
        "question": question,
        "dbinfo": dbinfo,
    })

    format = f"The input is SQL query output to the question {question}."
    info = ""
    for tool_call in response.tool_calls:
        if function_to_call := tools_map.get(tool_call["name"]):
            tool_response = function_to_call.invoke(tool_call["args"])
            info += " " + agent_paraphraser(llm=llm, input=tool_response, format=format)
    
    if info == "":
        info = response.content
    
    print(f"question: {question}\nanswer: {info}\n")


llm = ChatOpenAI(
    model="ai/gpt-oss:latest",
    temperature=0,
    base_url="http://localhost:12434/engines/v1",
    api_key="docker",
)

dbname = HOME + "/repo/playground-ai-ml/data/sql-agentic-ai.db"

try:
    db = MySqlite3Db(dbname=dbname)

    db.setup_tables()
    db.print_tables()
    dbinfo = agent_tableinfo(llm=llm, db=db)
    
    agent_expert(llm=llm, dbinfo=dbinfo, db=db, question="Who is the employee with the highest salary, and how much was the salary?")
    agent_expert(llm=llm, dbinfo=dbinfo, db=db, question="Who are the top three highest earning employees, and what are their salaries?")
    agent_expert(llm=llm, dbinfo=dbinfo, db=db, question="What is the largest bone in the human body?")
    agent_expert(llm=llm, dbinfo=dbinfo, db=db, question="Which department has the most number of employees, and how many employees does it have?")
    agent_expert(llm=llm, dbinfo=dbinfo, db=db, question="Among the employees with contact information, who has the highest salary and what is the email address?.")
    agent_expert(llm=llm, dbinfo=dbinfo, db=db, question="How many moons does Jupiter have?")
except sqlite3.Error as e:
    print(f"Error caught: {e}")
    exit(1)

# # results using gpt-oss

# departments: [(1, 'IT'), (2, 'HR'), (3, 'Marketing'), (4, 'Finance')]
# employees: [(1, 1, 'Alex', 10000.0), (2, 1, 'John', 9000.0), (3, 1, 'Mary', 9500.0), (4, 1, 'Joseph', 9700.0), (5, 1, 'Jane', 9200.0), (6, 2, 'Monique', 8000.0), (7, 2, 'Owen', 8100.0), (8, 3, 'Fred', 7200.0), (9, 3, 'Michelle', 8100.0), (10, 3, 'Janice', 9800.0)]
# contacts: [(1, 2, '1234-5678', 'john@email.com', 'binondo, manila'), (2, 8, '1111-2222', 'fred@email.com', 'ermita, manila'), (3, 10, '3456-1234', 'janice@email.com', 'makati, manila'), (4, 7, None, 'owen@email.com', None)]

# database_query: SELECT '[' || group_concat(name || ', (' || cols || ')', ', ') || ']' AS result FROM (SELECT name, (SELECT group_concat(name, ', ') FROM pragma_table_info(t.name)) AS cols FROM sqlite_master t WHERE type='table' ORDER BY name)

# database_query: SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1

# question: Who is the employee with the highest salary, and how much was the salary?
# answer:  Alex earns the highest salary, which is $10,000.

# database_query: SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 3

# question: Who are the top three highest earning employees, and what are their salaries?
# answer:  The top three highest‑earning employees are Alex, who earns $10,000; Janice, who earns $9,800; and Joseph, who earns $9,700.

# question: What is the largest bone in the human body?
# answer: The largest bone in the human body is the **femur** (thigh bone).

# database_query: SELECT d.name, COUNT(e.id) AS employee_count FROM departments d JOIN employees e ON d.id = e.department_id GROUP BY d.id ORDER BY employee_count DESC LIMIT 1;

# question: Which department has the most number of employees, and how many employees does it have?
# answer:  The IT department has the most employees, with a total of five.

# database_query: SELECT e.name, c.email FROM employees e JOIN contacts c ON e.id = c.employee_id ORDER BY e.salary DESC LIMIT 1;

# question: Among the employees with contact information, who has the highest salary and what is the email address?.
# answer:  Janice, whose email address is janice@email.com, has the highest salary among employees with contact information.

# question: How many moons does Jupiter have?
# answer: Jupiter has **79 known moons** (as of the most recent counts in 2023).
