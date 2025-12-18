import streamlit as st
import asyncio
import os
from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from dotenv import load_dotenv

# 1. Page Configuration
st.set_page_config(page_title="MCP Customer Support", page_icon="💻")
st.title("💻 Computer Hardware Support")
st.markdown("Chat with our AI agent powered by MCP tools.")

# 2. Setup Environment and LLM
load_dotenv()


# We cache the agent setup so it doesn't reload on every message toggle
@st.cache_resource
def get_agent_executor():
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    MCP_SERVER_URL = "https://vipfapwm3x.us-east-1.awsapprunner.com/mcp"

    client = MultiServerMCPClient({
        "computer_products": {
            "url": MCP_SERVER_URL,
            "transport": "streamable_http"
        }
    })

    # Using an internal loop to fetch tools since this is a cached resource
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    tools = loop.run_until_complete(client.get_tools())

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert Customer Support Agent for a computer hardware company.

         CRITICAL TOOL CALLING RULES:
         1. To use a tool, you MUST output ONLY the tool name in the 'name' field.
         2. DO NOT append JSON or arguments to the tool name string.
         3. Example: name='get_customer', args={{"customer_id": "123"}}

         If a tool returns an error, explain it politely to the customer."""),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    agent = create_tool_calling_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True)


# 3. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display message history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 4. Chat Input Logic
if prompt_input := st.chat_input("How can I help you with your order or products?"):
    # Add user message to state and display
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    with st.chat_message("user"):
        st.markdown(prompt_input)

    # Generate response
    with st.chat_message("assistant"):
        agent_executor = get_agent_executor()

        # Format history for the agent
        history = []
        for msg in st.session_state.messages[:-1]:
            if msg["role"] == "user":
                history.append(HumanMessage(content=msg["content"]))
            else:
                history.append(AIMessage(content=msg["content"]))

        # Run the agent asynchronously
        with st.spinner("Consulting product database..."):
            response = asyncio.run(agent_executor.ainvoke({
                "input": prompt_input,
                "chat_history": history
            }))

            full_response = response["output"]
            st.markdown(full_response)

    # Save assistant response to state
    st.session_state.messages.append({"role": "assistant", "content": full_response})