import streamlit as st
from pathlib import Path
from langchain.agents import create_sql_agent
from langchain.sql_database import SQLDatabase
from langchain.agents.agent_types import AgentType
from langchain.callbacks import StreamlitCallbackHandler
from langchain.agents.agent_toolkits import SQLDatabaseToolkit
from sqlalchemy import create_engine
import sqlite3
from langchain_groq import ChatGroq

# Page configuration
st.set_page_config(page_title="Langchain: Chat with SQL DB", page_icon="🦜")
st.title("Langchain : Chat with SQL DB")

# Constants for database types
LOCALDB = 'USE_LOCALDB'
MYSQL = "USE_MYSQL"

# Sidebar for database selection
radio_opt = ["Use SQLite 3 Database - Student.db", "Connect to your MySQL Database"]
selected_opt = st.sidebar.radio(label="Choose the database you want to chat with", options=radio_opt)

# MySQL connection input fields
if selected_opt == "Connect to your MySQL Database":
    db_uri = MYSQL
    mysql_host = st.sidebar.text_input("Provide MySQL Host")
    mysql_user = st.sidebar.text_input("MySQL User")
    mysql_password = st.sidebar.text_input("MySQL Password", type="password")
    mysql_db = st.sidebar.text_input("MySQL Database")
else:
    db_uri = LOCALDB

# API Key input
api_key = st.sidebar.text_input(label="GROQ_API_KEY", type="password")

# Validation for DB URI and API key
if not db_uri:
    st.info("Please enter the database information and URI.")
if not api_key:
    st.info("Please add the Groq API key.")

# LLM Model setup (Groq API)
llm = ChatGroq(groq_api_key=api_key, model="Gemma2-9b-It", streaming=True)

# Function to configure the database connection
@st.cache_resource(ttl="2h")
def configure_db(db_url, mysql_host=None, mysql_user=None, mysql_password=None, mysql_db=None):
    try:
        if db_url == LOCALDB:
            db_filepath = Path(__file__).parent / "Student.db"
            creator = lambda: sqlite3.connect(f"file:{db_filepath}?mode=ro", uri=True)
            engine = create_engine("sqlite://", creator=creator)
        elif db_url == MYSQL:
            if not (mysql_host and mysql_user and mysql_password and mysql_db):
                st.error("Please provide all MySQL connection details.")
                st.stop()
            engine = create_engine(f"mysql+mysqlconnector://{mysql_user}:{mysql_password}@{mysql_host}/{mysql_db}")
        else:
            raise ValueError("Unsupported database type")
        return SQLDatabase(engine)
    except Exception as e:
        st.error(f"An error occurred while configuring the database: {str(e)}")
        st.stop()

# Initialize the database connection
try:
    if db_uri == MYSQL:
        db = configure_db(db_uri, mysql_host, mysql_user, mysql_password, mysql_db)
    else:
        db = configure_db(db_uri)
except Exception as e:
    st.error(f"An error occurred during database initialization: {str(e)}")
    st.stop()

# Toolkit setup
toolkit = SQLDatabaseToolkit(db=db, llm=llm)

agent = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Initialize message history
if "messages" not in st.session_state or st.sidebar.button("Clear Message History"):
    st.session_state['messages'] = [{"role": "assistant", "content": "How can I help you?"}]

# Display chat messages
for msg in st.session_state.messages:
    st.chat_message(msg['role']).write(msg['content'])

# User input for queries
user_query = st.chat_input(placeholder="Ask anything from databases")

if user_query:
    st.session_state.messages.append({"role": 'user', 'content': user_query})
    st.chat_message("user").write(user_query)
    
    with st.chat_message('assistant'):
        streamlit_callback = StreamlitCallbackHandler(st.container())
        response = agent.run(user_query, callbacks=[streamlit_callback])
        st.session_state.messages.append({'role': "assistant", "content": response})
        st.write(response)