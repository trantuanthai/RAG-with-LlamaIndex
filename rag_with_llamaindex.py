from llama_index.core.evaluation import generate_question_context_pairs
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.evaluation import RetrieverEvaluator
from llama_index.llms.openai import OpenAI

import streamlit as st
import openai

openai.api_key = st.secrets.openai_key

st.title("HỆ THỐNG TƯ VẤN TUYỂN SINH")

sidebar = st.sidebar

if "messages" not in st.session_state.keys(): # Initialize the chat messages history
    st.session_state.messages = [
        {"role": "assistant", "content": "Tôi có thể tư vấn gì cho bạn?"}
    ]

if "conversation_history" not in st.session_state.keys() or st.sidebar.button("Cuộc trò chuyện mới"):
  st.session_state.conversation_history = []

# New Conversation button with a unique key
new_conversation_button = sidebar.button("Cuộc trò chuyện mới", key="new_conversation")

if new_conversation_button:  # Reset conversation on button click
    st.session_state.messages = []

@st.cache_resource(show_spinner=False)
def load_data():
    with st.spinner(text="Dữ liệu đang được tải lên, vui lòng chờ trong ít phút!"):
        documents = SimpleDirectoryReader("./Data").load_data()

        # Define an LLM
        llm = OpenAI(model="gpt-3.5")

        # Build index with a chunk_size of 512
        node_parser = SimpleNodeParser.from_defaults(chunk_size=512)
        nodes = node_parser.get_nodes_from_documents(documents)
        index = VectorStoreIndex(nodes)

        return index

index = load_data()

if "chat_engine" not in st.session_state.keys(): # Initialize the chat engine
        st.session_state.chat_engine = index.as_chat_engine(chat_mode="condense_plus_context", verbose=True)

if prompt := st.chat_input("Câu hỏi"): # Prompt for user input and save to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages: # Display the prior chat messages
    with st.chat_message(message["role"]):
        st.write(message["content"])

# If last message is not from assistant, generate a new response
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("..."):
            response = st.session_state.chat_engine.chat(prompt)
            st.write(response.response)
            message = {"role": "assistant", "content": response.response}
            st.session_state.messages.append(message) # Add response to message history
            st.session_state.conversation_history.append(st.session_state.messages.copy()) # Append conversation history

with st.sidebar.expander("Conversation History"):
    for conversation in st.session_state.conversation_history:
        for message in conversation:
            if message["role"] == "user":
                st.write("Bạn: " + message["content"])
            else:
                st.write("Chatbot: " + message["content"])
        st.write("----")  # Separator between conversations
