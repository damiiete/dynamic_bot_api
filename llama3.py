# %%
print("Start Loading...")
import ollama
import bs4
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

load_dotenv(".env")

from groq import Groq
import os
from langchain_groq import ChatGroq 

#%%
# client = Groq(
#     api_key=os.getenv("GROQ_API_KEY"),
# )

client = Groq(
    api_key=os.environ["GROQ_API_KEY"],
)
#%%
llm_groq = ChatGroq(model='llama3-70b-8192')

# %%
#1. Load the data
# loader = WebBaseLoader(
#     web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
#     bs_kwargs=dict(
#         parse_only=bs4.SoupStrainer(
#             class_=("post-content", "post-title", "post-header")
#         )
#     ),
# )
# docs = loader.load()
# text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
# splits = text_splitter.split_documents(docs)

# %%
# 2. Create Ollama embeddings and vector store
# persist_directory = 'docs/chroma'
# embeddings = OllamaEmbeddings(model='llama3')
# vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings)
# vectorstore.persist()

# %% [markdown]
# ## Testing code from internet

# %%
# Vectorstore
persist_directory = 'docs/chroma'
vectorstore = Chroma(embedding_function=OllamaEmbeddings(model='llama3'), persist_directory=persist_directory)
retriever = vectorstore.as_retriever()
def combine_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#%%
# SQL Chain
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.pydantic_v1 import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from langchain_community.chat_models import ChatOllama
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from getpass import getpass
from pathlib import Path
import time


GROQ_API_KEY = os.environ["GROQ_API_KEY"]

# Start time
start_time = time.time()
print(f"Start time is: {start_time}")

# Add the LLM downloaded from Ollama
# ollama_llm = "llama2"
# llm = ChatOllama(model=ollama_llm)

llm = ChatGroq(temperature=0, model="llama3-70b-8192",
               groq_api_key=GROQ_API_KEY)

db_path = Path(__file__).parent / "patientHealthData.db"
rel = db_path.relative_to(Path.cwd())
db_string = f"sqlite:///{rel}"
db = SQLDatabase.from_uri(db_string, sample_rows_in_table_info=0)


def get_schema(_):
    return db.get_table_info()


def run_query(query):
    return db.run(query)


# Prompt

template = """Based on the table schema below, write a SQL query that would answer the user's question. :
{schema}

Question: {question}
SQL Query:"""  # noqa: E501

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", """Given an input question, convert it to a SQL query. 
         Your output must only include SQL query. Your output should not include the triple ticks: ''' ''' 

         """),
        MessagesPlaceholder(variable_name="history"),
        ("human", template),
    ]
)

memory = ConversationBufferMemory(return_messages=True)

# Chain to query with memory

sql_chain = (
    RunnablePassthrough.assign(
        schema=get_schema,
        history=RunnableLambda(
            lambda x: memory.load_memory_variables(x)["history"]),
    )
    | prompt
    | llm.bind(stop=["\nSQLResult:"])
    | StrOutputParser()
)

# response = sql_chain.invoke(
#     {"question": "What did the patient have for breakfast and dinner on 2023-12-21?"})
# print(response)


def save(input_output):
    output = {"output": input_output.pop("output")}
    memory.save_context(input_output, output)
    return output["output"]


sql_response_memory = RunnablePassthrough.assign(output=sql_chain) | save

# Chain to answer
template = """Based on the table schema below, question, sql query, and sql response, write a natural language response:
{schema}

Question: {question}
SQL Query: {query}
SQL Response: {response}"""  # noqa: E501
prompt_response = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Given an input question and SQL response, convert it to a natural "
            "language answer. No pre-amble.",
        ),
        ("human", template),
    ]
)


# Supply the input types to the prompt
class InputType(BaseModel):
    question: str


chain = (
    RunnablePassthrough.assign(query=sql_response_memory).with_types(
        input_type=InputType
    )
    | RunnablePassthrough.assign(
        schema=get_schema,
        response=lambda x: db.run(x["query"]),
    )
    | prompt_response
    | llm
)

# print(response)

# response1 = chain.invoke(
#     {"question": "What did the patient have for breakfast and dinner on 2023-12-21?"})
# print(response1.content)

# %%
### LLM

local_llm = "llama3"

# %%
### Retrieval Grader

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing relevance 
    of a retrieved document to a user question. If the document contains keywords related to the user question, 
    grade it as relevant. It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explaination.
     <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n <|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """,
    input_variables=["question", "document"],
)

retrieval_grader = prompt | llm_groq| JsonOutputParser()
# question = "agent memory"
# docs = retriever.invoke(question)
# doc_txt = docs[1].page_content
# print(retrieval_grader.invoke({"question": question, "document": docs}))

# %%
### Generate chain

from langchain.prompts import PromptTemplate
from langchain import hub
from langchain_core.output_parsers import StrOutputParser

# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise <|eot_id|><|start_header_id|>user<|end_header_id|>
    Question: {question} 
    Context: {context} 
    Answer: <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question", "context"],
)

llm = ChatOllama(model=local_llm, temperature=0)


# Post-processing
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Chain
rag_chain = prompt | llm_groq | StrOutputParser()

# Run
question = "agent memory"
# question = "hello, how do you do"
docs = retriever.invoke(question)
# generation = rag_chain.invoke({"context": docs, "question": question})
# print(generation)
#%%
print(f"This is the docs \n {docs}")
print(type(docs))

# %%
### Hallucination Grader chain

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Prompt
prompt = PromptTemplate(
    template=""" <|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether 
    an answer is grounded in / supported by a set of facts. Give a binary 'yes' or 'no' score to indicate 
    whether the answer is grounded in / supported by a set of facts. Provide the binary score as a JSON with a 
    single key 'score' and no preamble or explanation. <|eot_id|><|start_header_id|>user<|end_header_id|>
    Here are the facts:
    \n ------- \n
    {documents} 
    \n ------- \n
    Here is the answer: {generation}  <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "documents"],
)

hallucination_grader = prompt | llm_groq | JsonOutputParser()
# hallucination_grader.invoke({"documents": docs, "generation": generation})

# %%
### Answer Grader chain

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

# Prompt
prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> You are a grader assessing whether an 
    answer is useful to resolve a question. Give a binary score 'yes' or 'no' to indicate whether the answer is 
    useful to resolve a question. Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.
     <|eot_id|><|start_header_id|>user<|end_header_id|> Here is the answer:
    \n ------- \n
    {generation} 
    \n ------- \n
    Here is the question: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["generation", "question"],
)

answer_grader = prompt | llm_groq | JsonOutputParser()
#answer_grader.invoke({"question": question, "generation": generation})

# %%
### Router chain

from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama
from langchain_core.output_parsers import JsonOutputParser

# LLM
llm = ChatOllama(model=local_llm, format="json", temperature=0)

prompt = PromptTemplate(
    template="""<|begin_of_text|><|start_header_id|>system<|end_header_id|> 
    You are an expert at routing a users question to one of 4 sources. Use the vectorstore for questions on LLM  agents, 
    prompt engineering, and adversarial attacks. Use booking for input related to booking a meeting or appointment.\
    Use sql for questions related to information about the user's data. Otherwise, use web-search.
    Give a binary choice 'web_search', 'booking', 'vectorstore' or 'sql' based on the question. Return a JSON with a single key 'datasource' and 
    no premable or explaination. Question to route: {question} <|eot_id|><|start_header_id|>assistant<|end_header_id|>""",
    input_variables=["question"],
)

question_router = prompt | llm_groq | JsonOutputParser()
# question = "hello"
# question = "What did the patient have for breakfast and dinner on 2023-12-21?"
# print(question_router.invoke({"question": question}))

# %%
### web Search tool

from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import load_tools


web_search_tool = TavilySearchResults(k=3)


# %%
#### Booking function setup

from datetime import datetime, timedelta
import os.path

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

import pickle

from typing import Optional, Type
from langchain.tools import BaseTool
from langchain.agents import initialize_agent
from pydantic import BaseModel, Field
from langchain.agents import AgentType


# If modifying these SCOPES, delete the file token.pickle.
SCOPES = ['https://www.googleapis.com/auth/calendar']


"""Shows basic usage of the Google Calendar API.
Lists the next 10 events on the user's calendar.
"""
creds = None
# The file token.pickle stores the user's access and refresh tokens, and is
# created automatically when the authorization flow completes for the first
# time.
if os.path.exists('token.pickle'):
    with open('token.pickle', 'rb') as token:
        creds = pickle.load(token)
# If there are no (valid) credentials available, let the user log in.
if not creds or not creds.valid:
    if creds and creds.expired and creds.refresh_token:
        creds.refresh(Request())
    else:
        flow = InstalledAppFlow.from_client_secrets_file(
            'credentials.json', SCOPES)
        creds = flow.run_local_server(port=0)
    # Save the credentials for the next run
    with open('token.pickle', 'wb') as token:
        pickle.dump(creds, token)

service = build('calendar', 'v3', credentials=creds)

# Call the Calendar API
now = datetime.utcnow().isoformat() + 'Z' # 'Z' indicates UTC time

def create_event(doc_email, client_email, client_name, start_time):
    # Refer to the Python quickstart on how to setup the environment:
    # https://developers.google.com/calendar/quickstart/python
    # Change the scope to 'https://www.googleapis.com/auth/calendar' and delete any
    # stored credentials.

    
    start_time = start_time.replace('Z',"")
    end_time = datetime.strptime(start_time, '%Y-%m-%dT%H:%M:%S') + timedelta(hours=1)
    #end_time = start_time + timedelta(hours=1)

    event = {
    'summary': f'FundusAi Test meeting with {client_name}',
    'location': 'Virtual', #might not need it
    'description': 'Testing the event creation function.',
    'start': {
        'dateTime': start_time, #start_time.strptime('%Y-%m-%dT%H:%M:%S'), #start_date.format in date and time
        'timeZone': 'Europe/London', #users timezone input or give specific time zone
    },
    'end': {
        'dateTime': end_time.strftime('%Y-%m-%dT%H:%M:%S'),
        'timeZone': 'Europe/London',
    },
    # 'recurrence': [
    #     'RRULE:FREQ=DAILY;COUNT=2'
    # ],
    'attendees': [
        {'email': doc_email},
        {'email': client_email},
    ],
    'reminders': {
        'useDefault': False,
        'overrides': [
        {'method': 'email', 'minutes': 24 * 60},
        {'method': 'popup', 'minutes': 10},
        ],
    },
    #'id': f'{client_name}_FundusAi' 
    "conferenceData": {
    "createRequest": {
      "requestId": f"{now}", #to have a unique identifier
      "conferenceSolutionKey": {
        "type": 'hangoutsMeet'
      },
    },    
    }

    }

    event = service.events().insert(calendarId='primary', body=event, conferenceDataVersion=1).execute()
    return ('Event created: %s' % (event.get('htmlLink')))



# %%
### State
from typing_extensions import TypedDict
from typing import List
from langgraph.checkpoint import MemorySaver

### State

class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """

    question: str
    generation: str
    web_search: str
    documents: List[str]


from langchain.schema import Document


### Nodes

def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]

    # Retrieval
    documents = retriever.invoke(question)
    return {"documents": documents, "question": question}


def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]

    # RAG generation
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}


def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]

    # Score each doc
    filtered_docs = []
    web_search = "No"
    for d in documents:
        score = retrieval_grader.invoke(
            {"question": question, "document": d.page_content}
        )
        grade = score["score"]
        # Document relevant
        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        # Document not relevant
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            # We do not include the document in filtered_docs
            # We set a flag to indicate that we want to run web search
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}


def web_search(state):
    """
    Web search based based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state["documents"]

    # Web search
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join([d["content"] for d in docs])
    #web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    return {"documents": documents, "question": question}


def booking(state):
    """
    Performs the booking of the meeting for the client

    Args:
        state (dict): The current graph state

    Returns:
        state(dict): response of booking
    """
    print("---BOOKING---")

    question = state["question"]

    client_email = input("Please type your email")
    client_name = input("Please type your name")
    doc_email = input("Please add doc email")
    start_time = input("put start time in YY-MM-DDTH:m:s format")

    result = create_event(doc_email, client_email, client_name, start_time)
    link = result.split()[-1]

    
    return {"question": question, "generation": f'Booked Successfully. Check your calendar using the link {link}'}
    # else:
    #     return "not

def sql(state):
    """
    Answers user's question from the sql database

    Args:
        state (dict): The current graph state

    Returns:
        str: response of booking
    """
    print("---SQL---")

    question = state["question"]

    response = chain.invoke({"question": question})
    generation = response.content
    # print(response1.content)

    return {"question": question, "generation": generation}


### Conditional edge

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})
    print(source)
    print(source["datasource"])
    if source["datasource"] == "web_search":
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source["datasource"] == "vectorstore":
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    elif source["datasource"] == "booking":
        print("---ROUTE QUESTION TO Booking---")
        return "booking"
    elif source['datasource'] == "sql":
        print("---ROUTE QUESTION TO Sql---")
        return "sql"

def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        # All documents have been filtered check_relevance
        # We will re-generate a new query
        print(
            "---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---"
        )
        return "not relevant"
    else:
        # We have relevant documents, so generate answer
        print("---DECISION: GENERATE---")
        return "generate"
    #else - book

### Conditional edge


def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]

    score = hallucination_grader.invoke(
        {"documents": documents, "generation": generation}
    )
    grade = score["score"]

    # Check hallucination
    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        # Check question-answering
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question, "generation": generation})
        grade = score["score"]
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        pprint("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"

# %%
##Langgraph build
from langgraph.graph import END, StateGraph

workflow = StateGraph(GraphState)

# Define the nodes
workflow.add_node("websearch", web_search)  # web search
workflow.add_node("retrieve", retrieve)  # retrieve
workflow.add_node("grade_documents", grade_documents)  # grade documents
workflow.add_node("generate", generate)  # generate
workflow.add_node("booking", booking) #booking
workflow.add_node("sql", sql)


workflow.set_conditional_entry_point(
    route_question,
    {
        # string - output of fucntion call 'route_question'
        #string:node to call
        "booking": "booking",
        "vectorstore": "retrieve",
        "websearch": "websearch",
        "sql": "sql",
    },
)

workflow.add_edge("booking", END)
workflow.add_edge("sql", END)
workflow.add_edge("retrieve", "grade_documents")
# workflow.add_edge("booking", "generate")

workflow.add_conditional_edges(
    "grade_documents",
    decide_to_generate,
    {
        # string - output of fucntion call 'decided_to_generate'
        #string:node to call 
        "not relevant": "websearch", 
        "generate": "generate",
    },
)
workflow.add_edge("websearch", "generate")
workflow.add_conditional_edges(
    "generate",
    grade_generation_v_documents_and_question,
    {
        "not supported": "generate",
        "useful": END,
        "not useful": "websearch",
    },
)

#%%
checkpointer = MemorySaver()

# # %%
# # Compile test 1 
app = workflow.compile(checkpointer=checkpointer)
print("App Loaded. . .")
#%%
# res = app.invoke({"question": "What did the patient have for breakfast and dinner on 2023-12-21?"})
# res['generation']
#%%

# # Test
# from pprint import pprint

# inputs = {"question": "What are the types of agent memory?"}
# for output in app.stream(inputs):
#     for key, value in output.items():
#         pprint(f"Finished running: {key}:")
# pprint(value["generation"])

# inputs = app.invoke({"question": "what is the current weather in Abuja Nigeria"},
#         config={
#             "configurable": {
#             "thread_id": 244}}       
# )
# inputs
#%%

# for output in app.stream(inputs):
#     for key, value in output.items():
#         pprint(f"Finished running: {key}:")
# pprint(value["generation"])

# # %%
# # Compile test 2
# app = workflow.compile()

# # Test
# from pprint import pprint

# inputs = {"question": "Who are the Bears expected to draft first in the NFL draft?"}
# for output in app.stream(inputs):
#     for key, value in output.items():
#         pprint(f"Finished running: {key}:")
# pprint(value["generation"])

# # %%
# # compile test 3
# # you might get an error cos of the stream output but you can stil confirm the event was booked

# app = workflow.compile()

# from pprint import pprint

# inputs = {"question": "Please I want to book a meeting for 12 pm on 20th of May 2024"}
# for output in app.stream(inputs):
#     for key, value in output.items():
#         pprint(f"Finished running: {key}:")
# pprint(value["generation"])


# #%%
# #compile test 4 - SQL
# app = workflow.compile()
# inputs = {"question": "What did the patient have for breakfast and dinner on 2023-12-21?"}
# for output in app.stream(inputs):
#     for key, value in output.items():
#         pprint(f"Finished running: {key}:")
# print(value["generation"])

#%%

# from IPython.display import Image, display
# display(Image(app.get_graph().draw_mermaid_png()))
# try:
#     display(Image(app.get_graph().draw_mermaid_png()))
# except Exception:
#     # This requires some extra dependencies and is optional
#     print('error')




# %%
# if __name__ == "__main__":
#     app = workflow.compile()
#     from pprint import pprint

#     inputs = {"question": "Please I want to book a meeting for 12 pm on 20th of May 2024"}
#     for output in app.stream(inputs):
#         for key, value in output.items():
#             pprint(f"Finished running: {key}:")
#     pprint(value["generation"])




##############
# END OF WORKING WORKFLOW
#############
