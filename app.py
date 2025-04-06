import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.utilities import ArxivAPIWrapper,WikipediaAPIWrapper
from langchain_community.tools import ArxivQueryRun,WikipediaQueryRun,DuckDuckGoSearchResults
from langchain.agents import initialize_agent,AgentType
from langchain.callbacks import StreamlitCallbackHandler
import os
from dotenv import load_dotenv
load_dotenv()


arxiv_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=500)
wiki_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=500)
arxiv=ArxivQueryRun(api_wrapper=arxiv_wrapper)
wiki=WikipediaQueryRun(api_wrapper=wiki_wrapper)
search=DuckDuckGoSearchResults(name="Search")

st.title("Langchain --- Chat with Search")
st.sidebar.title("settings")
api_key=st.sidebar.text_input("Enter the api_key",type="password")

if not api_key:
    st.error("Please enter an API key.")
    st.stop()

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi,iam a ChatBot who can search the web. How can i assist you"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

if prompt:=st.chat_input(placeholder="What is Machine Learning?"):
    st.session_state.messages.append({"role":"user","content":prompt})
    st.chat_message("user").write(prompt)
   
    llm=ChatGroq(groq_api_key=api_key,model_name="Llama3-8b-8192",streaming=True)
    tools=[search,arxiv,wiki]

    search_agent=initialize_agent(tools,llm,agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION)

    with st.chat_message("assistant"):
        st_cb=StreamlitCallbackHandler(st.container(),expand_new_thoughts=True)
        response=search_agent.run(prompt,callbacks=[st_cb])
        st.session_state.messages.append({'role':'assistant',"content":response})
        st.write(response)


