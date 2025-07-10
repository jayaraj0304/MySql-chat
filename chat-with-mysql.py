from dotenv import load_dotenv
import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_google_genai import GoogleGenerativeAI
import streamlit as st

load_dotenv()

# llm
# openai = ChatOpenAI(model=os.environ["OPENAI_MODEL"], api_key=os.environ["OPENAI_API_KEY"])
gemini = GoogleGenerativeAI(model=os.environ["GEMINI_MODEL"], api_key=os.environ["GEMINI_API_KEY"])

# Database setting
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
  db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
  return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: Promedio de coste de soles por tipo de planta
    SQL Query: SELECT tipo, AVG(coste_soles) AS promedio_coste_soles FROM plantas GROUP BY tipo;
    Question: Plantas por minijuego
    SQL Query: SELECT m.nombre_minijuego AS Minijuego, p.nombre_planta AS Planta FROM minijuego_plantas mp JOIN minijuegos m ON mp.id_minijuego = m.id_minijuego JOIN plantas p ON mp.id_planta = p.id_planta ORDER BY m.nombre_minijuego, p.nombre_planta;
    Question: Ver todas las partidas con detalles de planta, zombi y escenario
    SQL Query: SELECT pa.id_partida, p.nombre_planta, z.nombre_zombie, e.nombre AS escenario, pa.fecha_partida, pa.resultado FROM partidas pa JOIN plantas p ON pa.id_planta = p.id_planta JOIN zombies z ON pa.id_zombie = z.id_zombie JOIN escenarios e ON pa.id_escenario = e.id_escenario;

    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
  prompt = ChatPromptTemplate.from_template(template)
  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | gemini
    # | openai
    | StrOutputParser()
  )

def get_response(user_query: str, db: SQLDatabase, chat_history: list):
  sql_chain = get_sql_chain(db)
  
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}"""
  
  prompt = ChatPromptTemplate.from_template(template)
  
  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | gemini
    # | openai
    | StrOutputParser()
  )
  
  return chain.invoke({
    "question": user_query,
    "chat_history": chat_history,
  })

# Streamlit config
st.set_page_config(page_title="Chat with MySQL", page_icon=":speech_balloon:")

st.title(":speech_balloon: Chat with MySQL")

with st.sidebar:
    st.markdown("## Settings")
    st.write("This is a simple chat application using MySQL. Connect to the database and start chatting.")
    
    st.text_input("Host", value="localhost", key="Host")
    st.text_input("Port", value="3306", key="Port")
    st.text_input("User", value="root", key="User")
    st.text_input("Password", type="password", value="root", key="Password")
    st.text_input("Database", value="pvz", key="Database")
    
    if st.button("Connect"):
        with st.spinner("Connecting to database..."):
            db = init_database(
                st.session_state["User"],
                st.session_state["Password"],
                st.session_state["Host"],
                st.session_state["Port"],
                st.session_state["Database"]
            )
            st.session_state.db = db
            st.success("Connected to database!")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [AIMessage(content="Hi! I'm a SQL assistant. Ask me anything about your database."),]

for message in st.session_state.chat_history:
    if isinstance(message, AIMessage):
        with st.chat_message(name="ai", avatar="img/chatbot.png"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message(name="human", avatar="img/boy.png"):
            st.markdown(message.content)

user_query = st.chat_input("Ask anything about your database...")

if user_query is not None and user_query.strip() != "":
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    
    with st.chat_message(name="human", avatar="img/boy.png"):
        st.markdown(user_query)

    with st.chat_message(name="ai", avatar="img/chatbot.png"):
        # Return a sql query
        sql_chain = get_sql_chain(st.session_state.db)
        sql_response = sql_chain.invoke({
            "chat_history": st.session_state.chat_history,
            "question": user_query
        })

        # Return a natural language response
        response = get_response(user_query, st.session_state.db, st.session_state.chat_history)

        st.markdown("### Generated SQL Query:")
        st.code(sql_response)
        st.markdown("### Query Result:")
        st.markdown(response)

    st.session_state.chat_history.append(AIMessage(content=sql_response))    
    st.session_state.chat_history.append(AIMessage(content=response))