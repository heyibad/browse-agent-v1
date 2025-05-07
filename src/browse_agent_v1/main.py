import asyncio
import os
from dotenv import load_dotenv, find_dotenv
from openai import AsyncOpenAI
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_api,
    set_default_openai_client,
    set_tracing_disabled,
    OpenAIChatCompletionsModel
)
from browser_use import Agent as BrowserAgent
from langchain_google_genai import ChatGoogleGenerativeAI



# Load the .env file
load_dotenv(find_dotenv()) 

# Get the API key from the .env file
gemini_api_key=os.getenv("GEMINI_API_KEY")
model_name= os.getenv("MODEL_NAME")
base_url=os.getenv("BASE_URL")

# print(f"{gemini_api_key=}, {model_name=}, {base_url=}") # Debugging line

if not gemini_api_key or not model_name or not base_url :
    raise Exception("API Key or Credentials not found in .env file")

# Create an instance of the OpenAI client for external LLM(Gemini)
client= AsyncOpenAI(
    api_key=gemini_api_key,
    base_url=base_url
)

set_default_openai_client(client=client, use_for_tracing=False)
set_default_openai_api("chat_completions")
set_tracing_disabled(disabled=True)

LLM = ChatGoogleGenerativeAI(
    model=model_name,
    api_key=gemini_api_key,
)

@function_tool
def get_weather(city: str):
    print(f"weather for {city}")
    return f"The weather in {city} is sunny."


@function_tool(
    name_override="browser_agent",
    description_override="Use this to navigate in a real browser: visit URLs, click links, scrape content, etc."
)
async def browser_agent(query: str) -> str:
    # Create and run the BrowserUse agent
    agent = BrowserAgent(llm=LLM, task=query)
    result = await agent.run()
    # `result.output` (or `.result`) is the text the agent scraped/back-chatted
    return result.output or result.result


async def main(input_text: str):
    agent = Agent(
    name="Assistant",
    instructions=(
        "You are an AI assistant with two tools:\n"
        "1. get_weather(city) → returns weather forecast\n"
        "2. browser_agent(query) → can open a browser, visit URLs, click links, and scrape content.\n"
        "Whenever the user asks you to navigate or fetch information from a website, "
        "you MUST call browser_agent with a clear, concise query."
    ),
    model=model_name,
    tools=[get_weather, browser_agent],
)


    result = await Runner.run(agent, input_text)
    print(result.final_output)


def run():
    input_text = input("Enter your query: ")
    asyncio.run(main(input_text))