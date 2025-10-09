import os
import sqlite3
import time
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.tools import tool
from langchain_core.runnables import RunnablePassthrough, RunnableLambda, Runnable

# practice code agentic RAG
# OOP alternative of sql-agentic-ai-03.py and rag-routing-03.py

HOME=os.environ["HOME"]

class MySqlite3Db:
    """A database object."""
    
    def __init__(self, **kwargs):
        self.conn = None
        self.cursor = None
        self.db_name = ""
        self.connect(**kwargs)
    
    def __del__(self):
        if self.conn == None:
            return
        
        self.conn.close()
    
    def get_db_type(self):
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
    
    def execute_commit(self, sql:str):
        """
        Executes a SQL for this DB instance and calls commit.

        Args:
            sql: The SQL query to be done.
        """
        if self.conn == None:
            return None
        
        if self.cursor == None:
            return None
        
        self.cursor.execute(sql)
        self.conn.commit()

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
        if "db_name" in kwargs:
            self.db_name = kwargs["db_name"]
        if len(self.db_name) == 0:
            return
                
        self.conn = sqlite3.connect(self.db_name)
        self.cursor = self.conn.cursor()

class TokenManager:
    def __init__(self):
        self.total_tokens = 0

    def measure_tps(self, what, runnable: Runnable, *args, **kwargs):
        start_time = time.time()
        response = runnable.invoke(*args, **kwargs)
        end_time = time.time()
        time_taken = end_time - start_time
        total_tokens = response.response_metadata.get("token_usage", {}).get("total_tokens")
        self.total_tokens += total_tokens
        tokens_per_second = total_tokens / time_taken
        print(f"{what}: {total_tokens} total tokens @ {tokens_per_second:.2f} tokens / second.")
        return response
    
    def get_total_tokens(self):
        return self.total_tokens

token_manager = TokenManager()

class AgentSqlDeveloper:
    def __init__(self, llm, db_type, enable_history=False):
        self.llm = llm
        self.db_type = db_type
        self.enable_history = enable_history
        self.chat_history = []

    def run(self, description: str):
        system_msg =  """You are a helpful assistant expert in {db_type} database.
            You can use unions and joins if required.
            No need to enclose the SQL in quotes.
            No need to provide an explanation for your answer.
            """
        human_msg = "Create a SQL for the given description. Description: {description}"

        if self.enable_history:
            prompt = ChatPromptTemplate.from_messages([
                ("system", system_msg),
                MessagesPlaceholder(variable_name="chat_history"),
                ("human", human_msg),
            ])
            chain = (prompt | llm)

            response = token_manager.measure_tps(__class__.__name__, chain, {
                "db_type": self.db_type,
                "description": description,
                "chat_history": self.chat_history,
            })
            
            self.chat_history.append(HumanMessage(content=human_msg.format(description=description)))
            self.chat_history.append(AIMessage(content=response.content))

            return response.content
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", human_msg),
        ])
        chain = (prompt | llm)

        response = token_manager.measure_tps(__class__.__name__, chain, {
            "db_type": self.db_type,
            "description": description,
        })

        return response.content

class AgentParaphraser:
    def __init__(self, llm):
        prompt = PromptTemplate(
            template="""You are a helpful assistant that paraphrase the input into sentence form.
                Take note of the input description to help in your conversion.
                Input: {input}
                Input format: {format}
                """,
            input_variables=["input", "format",],
        )
        self.chain = (prompt | llm)
    
    def run(self, input, format):
        response = token_manager.measure_tps(__class__.__name__, self.chain, {
            "input": input,
            "format": format,
        })
        return response.content

class AgentDbExpert:
    def __init__(self, db, agent_db_viewer, agent_paraphraser, db_info):
        self.db = db
        self.agent_db_viewer = agent_db_viewer
        self.agent_paraphraser = agent_paraphraser
        self.db_info = db_info
    
    def run(self, question: str):
        """
        Calls underlying SQL query based on the question.

        Args:
            question (str): The question used as basis for the SQL to run.
        """
        question = f"Given these tables and respective columns: {db_info}; {question}"
        info_format = f"The input is SQL query output to the question {question}."
        sql=self.agent_db_viewer.run(question)
        raw_response = db.execute_fetchall(sql)
        return self.agent_paraphraser.run(input=raw_response, format=info_format)

class AgentRetrieverSelector:
    def __init__(self, llm, conditions, retrievers):
        prompt = PromptTemplate(
            template="""You are helpful assistant that analyzes questions.
                Conditions: {conditions}
                Question: {question}
                """,
            input_variables=["conditions", "question"],
        )
        self.conditions = conditions
        self.retrievers = retrievers
        self.chain = (prompt | llm)
    
    def run(self, question: str):
        response = token_manager.measure_tps(__class__.__name__, self.chain, {
            "conditions": self.conditions,
            "question": question,
        })
        return self.retrievers[int(response.content)]

class AgentExpert:
    def __init__(self, llm, db_info, agent_db_expert, agent_retriever_selector):
        self.llm = llm
        self.db_info = db_info
        self.agent_retriever_selector = agent_retriever_selector
        self.prompt = PromptTemplate(
            template="""You are an assistant helping answer questions.
                Analyze the given question and the database information.
                If database has the needed tables and columns for the questions,
                execute the tool with a SQL to get the answer.
                You may use joins and unions in your SQL query if needed.
                If the database does not have the needed tables and columns,
                then use your training data and documents to answer the question.
                Format your answer in sentence form.
                Question: {question}.
                Database information: {db_info}.
                Documents: {documents}.
                """,
            input_variables=["question", "db_info", "context"],
        )

        tool_agent_db_expert = tool(agent_db_expert.run)
        self.tool_list = [tool_agent_db_expert]
        self.tool_map = {
            "run": tool_agent_db_expert
        }
    
    def run(self, question: str):
        retriever = self.agent_retriever_selector.run(question)
        chain = (
            { "documents": retriever, "question": RunnablePassthrough(), "db_info": lambda x: self.db_info } | 
            self.prompt |
            llm.bind_tools(self.tool_list)
        )
        response = token_manager.measure_tps(__class__.__name__, chain, question)

        if len(response.content) > 0:
            return response.content
        
        info = ""
        for tool_call in response.tool_calls:
            if function_to_call := self.tool_map.get(tool_call["name"]):
                info += " " + function_to_call.invoke(tool_call["args"])
        return info

def create_retriever(oa_embeddings: OpenAIEmbeddings, doc_path: str, store_path: str):
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

    Chroma.from_documents(
        documents=splits,
        embedding=oa_embeddings,
        persist_directory=store_path,
    )

def get_retriever(oa_embeddings: OpenAIEmbeddings, store_path: str):
    vector_store = Chroma(
        embedding_function=oa_embeddings,
        persist_directory=store_path
    )
    return vector_store.as_retriever()

embeddings_model = "ai/mxbai-embed-large"
api_url = "http://localhost:12434/engines/v1"
api_key = "docker"

oa_embeddings = OpenAIEmbeddings(
    model=embeddings_model,
    base_url=api_url,
    api_key=api_key,
    # disable check_embedding_ctx_length if your local model has different constraints
    check_embedding_ctx_length=False,
)

docs_billiards = HOME + "/repo/playground-ai-ml/data/routing-txt/billiards"
docs_guitars = HOME + "/repo/playground-ai-ml/data/routing-txt/guitars"
docs_technologies = HOME + "/repo/playground-ai-ml/data/routing-txt/technologies"
db_billiards = HOME + "/repo/playground-ai-ml/data/billiards.db"
db_guitars = HOME + "/repo/playground-ai-ml/data/guitars.db"
db_technologies = HOME + "/repo/playground-ai-ml/data/technologies.db"

llm_model = "ai/gpt-oss:latest"

llm = ChatOpenAI(
    model=llm_model,
    temperature=0,
    base_url=api_url,
    api_key=api_key,
)

db_name = HOME + "/repo/playground-ai-ml/data/sql-agentic-ai.db"
do_setup = False

if do_setup and os.path.exists(db_name):
    try:
        os.remove(db_name)
        os.remove(db_billiards)
        os.remove(db_guitars)
        os.remove(db_technologies)
    except FileNotFoundError:
        pass

db = MySqlite3Db(db_name=db_name)
db_type = db.get_db_type()

agent_db_creator = AgentSqlDeveloper(llm, db_type, enable_history=True)

if do_setup:
    sql = agent_db_creator.run("""
        Create a table named 'departments' with the following columns: id, name.
        'id' is the primary key, it is an autoincrementing integer.
        'name' is a string and cannot be null.
    """)
    print(f"sql:\n{sql}\n")
    db.execute_commit(sql)

    sql = agent_db_creator.run("""
        Create another table named 'employees' with the following columns: id, department_id, name, salary.
        'id' is the primary key, it is an autoincrementing integer.
        'department_id' is an integer, it is a foreign key for table 'departments' column 'id'.
        'name' is a string and it cannot be null.
        'salary' is a floating point number.
    """)
    print(f"sql:\n{sql}\n")
    db.execute_commit(sql)

    sql = agent_db_creator.run("""
        Create another table named 'contacts' with the following columns: id, employee_id, phone, email, address.
        'id' is the primary key, it is an autoincrementing integer.
        'employee_id' is an integer, it is a foreign key for table 'employees' column 'id'.
        'phone' is a string.
        'email' is a string.
        'address' is a string.
    """)
    print(f"sql:\n{sql}\n")
    db.execute_commit(sql)

    sql = agent_db_creator.run("Add the following departments: IT, HR, Marketing, Finance.")
    print(f"sql:\n{sql}\n")
    db.execute_commit(sql)

    sql = agent_db_creator.run("""
        Add the following employees:
        Lex from the IT department, with a 10000 salary.
        John from the IT department, with a 9000 salary.
        Mary from the IT department, with a 9500 salary.
        Joseph from the IT department, with a 9700 salary.
        Jane from the IT department, with a 9200 salary.
        Monique from the HR department, with a 8000 salary.
        Owen from the HR department, with a 8100 salary.
        Fred from the Marketing department, with a 7200 salary.
        Michelle from the Marketing department, with a 8100 salary.
        Janice from the Marketing department, with a 9800 salary.
    """)
    print(f"sql:\n{sql}\n")
    db.execute_commit(sql)

    sql = agent_db_creator.run("""
        Add contacts for the following employees:
        John: phone: 1234-5678, email: john@email.com, address: 'binondo, manila'
        Fred: phone: 1111-2222, email: fred@email.com, address: 'ermita, manila'
        Janice: phone: 3456-1234, email: janice@email.com, address: 'ayala, makati'
        Owen: email: owen@email.com
    """)
    print(f"sql:\n{sql}\n")
    db.execute_commit(sql)

    create_retriever(oa_embeddings, docs_billiards, db_billiards)
    create_retriever(oa_embeddings, docs_guitars, db_guitars)
    create_retriever(oa_embeddings, docs_technologies, db_technologies)

retriever_billiards = get_retriever(oa_embeddings, db_billiards)
retriever_guitars = get_retriever(oa_embeddings, db_guitars)
retriever_technologies = get_retriever(oa_embeddings, db_technologies)
retrievers = [RunnableLambda(lambda x: ""), retriever_billiards, retriever_guitars, retriever_technologies]
retriever_conditions = [
    "If question is related to billiards, return 1.",
    "If question is related to guitars, return 2.",
    "If question is related to software engineering or programming, return 3.",
    "Otherwise, return 0.",
]

agent_db_viewer = AgentSqlDeveloper(llm, db_type)
agent_paraphraser = AgentParaphraser(llm)

sql = agent_db_viewer.run("Get all available tables and respective columns.")
db_info = db.execute_fetchall(sql)

info_format = "[table1, (column1, column2, ...), table2, (column1, column2, ...), ...]"
db_info_phrase = agent_paraphraser.run(input=db_info, format=info_format)
print(f"db_info:\n{db_info_phrase}\n")

agent_db_expert = AgentDbExpert(db, agent_db_viewer, agent_paraphraser, db_info)
agent_retriever_selector = AgentRetrieverSelector(llm, retriever_conditions, retrievers)
agent_expert = AgentExpert(llm, db_info, agent_db_expert, agent_retriever_selector)

question = "Who are the top three highest earning employees, and what are their salaries?"
answer = agent_expert.run(question)
print(f"question:\n{question}\n")
print(f"answer:\n{answer}\n")

question = "What are Lex's break cues?"
answer = agent_expert.run(question)
print(f"question:\n{question}\n")
print(f"answer:\n{answer}\n")

question = "Among the employees with contact information, who has the highest salary and where does he or she lives?"
answer = agent_expert.run(question)
print(f"question:\n{question}\n")
print(f"answer:\n{answer}\n")

question = "What is the largest bone in the human body?"
answer = agent_expert.run(question)
print(f"question:\n{question}\n")
print(f"answer:\n{answer}\n")

question = "Is Lex a programmer?"
answer = agent_expert.run(question)
print(f"question:\n{question}\n")
print(f"answer:\n{answer}\n")

print(f"Total tokens: {token_manager.get_total_tokens()}\n")

# # results:
# AgentSqlDeveloper: 684 total tokens @ 55.83 tokens / second.
# AgentParaphraser: 637 total tokens @ 90.41 tokens / second.
# db_info:
# The database schema includes three tables: a **departments** table with columns **id** and **name**; an **employees** table with columns **id**, **department_id**, **name**, and **salary**; and a **contacts** table with columns **id**, **employee_id**, **phone**, **email**, and **address**.

# AgentRetrieverSelector: 259 total tokens @ 104.21 tokens / second.
# AgentExpert: 435 total tokens @ 165.97 tokens / second.
# AgentSqlDeveloper: 378 total tokens @ 116.17 tokens / second.
# AgentParaphraser: 528 total tokens @ 102.52 tokens / second.
# question:
# Who are the top three highest earning employees, and what are their salaries?

# answer:
#  The query returned the names and salaries of the three highest‑paid employees: Lex earned 10,000.0, Janice earned 9,800.0, and Joseph earned 9,700.0.

# AgentRetrieverSelector: 247 total tokens @ 98.46 tokens / second.
# AgentExpert: 1068 total tokens @ 219.41 tokens / second.
# question:
# What are Lex's break cues?

# answer:
# Lex’s break cues are:

# - **Action ACT 56 Break** – a break and jump cue made by Action Cues.  
# - **Mezz Power Break Kai** – a break cue made by Mezz Cues.  
# - **Predator Air 2** – a jump cue made by Predator Cues.

# AgentRetrieverSelector: 216 total tokens @ 109.00 tokens / second.
# AgentExpert: 532 total tokens @ 136.76 tokens / second.
# AgentSqlDeveloper: 560 total tokens @ 100.84 tokens / second.
# AgentParaphraser: 567 total tokens @ 99.25 tokens / second.
# question:
# Among the employees with contact information, who has the highest salary and where does he or she lives?

# answer:
#  The employee with the highest salary is Janice, whose address is ayala, makati.

# AgentRetrieverSelector: 191 total tokens @ 111.24 tokens / second.
# AgentExpert: 411 total tokens @ 174.20 tokens / second.
# question:
# What is the largest bone in the human body?

# answer:
# The largest bone in the human body is the **femur** (thigh bone).

# AgentRetrieverSelector: 209 total tokens @ 103.10 tokens / second.
# AgentExpert: 715 total tokens @ 199.85 tokens / second.
# question:
# Is Lex a programmer?

# answer:
# Yes, Lex is a programmer. The document lists a wide range of programming languages (C++, Python, Golang, Java, C, C#, Perl, SQL, Visual Basic, JavaScript, Angular) and frameworks (Spring, Spring Boot, REST API, Microservices, etc.) that Lex has used in his software engineering career.

# Total tokens: 7637
# Note as of this test, gpt-oss:20b price is $0.000030 / 1K tokens
