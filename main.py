import asyncio
import os
import logging
import sys

from langchain_groq import ChatGroq
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

# setup environment variables
load_dotenv()

# setup logging to file to keep console clean for chat
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    filename="agent.log"
)
logger = logging.getLogger(__name__)


async def run_interactive_support():
    # Setup LLM - Using llama-3.3-70b for better tool-calling reliability
    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)

    MCP_SERVER_URL = "https://vipfapwm3x.us-east-1.awsapprunner.com/mcp"

    # Create MCP client
    client = MultiServerMCPClient({
        "computer_products": {
            "url": MCP_SERVER_URL,
            "transport": "streamable_http"
        }
    })

    try:
        # Get tools from the MCP server
        tools = await client.get_tools()
        print(f"✅ Connected! Loaded {len(tools)} tools from MCP server.")

        # Define the Support Persona with escaped curly braces for tool examples
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert Customer Support Agent for a computer hardware company.

             CRITICAL TOOL CALLING RULES:
             1. To use a tool, you MUST output ONLY the tool name in the 'name' field.
             2. DO NOT append JSON or brackets to the tool name string.
             3. Example: name='get_customer', args={{"customer_id": "123"}}

             Always use your tools to provide accurate technical information."""),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        # Construct the Agent
        agent = create_tool_calling_agent(llm, tools, prompt)
        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

        print("\n--- Computer Support Chat Started (Type 'exit' or 'quit' to stop) ---")

        # Initialize an empty chat history
        chat_history = []

        while True:
            # 1. Get user input from the terminal
            user_input = input("\nUser: ")

            # 2. Check for exit command
            if user_input.lower() in ["exit", "quit", "q"]:
                print("Exiting chat. Goodbye!")
                break

            if not user_input.strip():
                continue

            try:
                # 3. Process the query with the agent
                # Note: 'chat_history' is passed if your prompt uses it
                response = await agent_executor.ainvoke({
                    "input": user_input,
                    "chat_history": chat_history
                })

                # 4. Display the response
                print(f"\nAgent: {response['output']}")

                # Update history (optional: limit history size to manage tokens)
                chat_history.append(("human", user_input))
                chat_history.append(("ai", response['output']))

            except Exception as e:
                print(f"\nError: {str(e)}")
                logger.error(f"Error during agent invocation: {e}")

    finally:
        # Cleanup if necessary (stateless clients handle most cleanup automatically)
        pass


if __name__ == "__main__":
    try:
        asyncio.run(run_interactive_support())
    except KeyboardInterrupt:
        print("\nSession ended by user.")
        sys.exit(0)