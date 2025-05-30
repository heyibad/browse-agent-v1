import asyncio
from steel import Steel
from browser_use import Agent
from langchain_google_genai import ChatGoogleGenerativeAI

from browser_use.browser.browser import Browser, BrowserConfig
from browser_use.browser.context import BrowserContext
from dotenv import load_dotenv
import os

load_dotenv()

# Initialize the Steel client with API key
# Replace "YOUR_STEEL_API_KEY" with your actual API key
client = Steel(steel_api_key=os.getenv("steel_api_key"))

# Create a Steel session
print("Creating Steel session...")
session = client.sessions.create(
    use_proxy=False,
    solve_captcha=False,
)
print(f"Session created at {session.session_viewer_url}")

# Connect browser-use to Steel
# Replace YOUR_STEEL_API_KEY with your actual API key
cdp_url = f"wss://connect.steel.dev?apiKey={os.getenv('steel_api_key')}&sessionId={session.id}"
browser = Browser(config=BrowserConfig(cdp_url=cdp_url))
browser_context = BrowserContext(browser=browser)


# Create and configure the AI agent
gemini_api_key=os.getenv("GEMINI_API_KEY")
model_name= os.getenv("MODEL_NAME")
if not gemini_api_key or not model_name :
    raise Exception("API Key or Credentials not found in .env file")
model = ChatGoogleGenerativeAI(
    model=model_name,
    api_key=gemini_api_key,  # Replace with your actual OpenAI API key
)

task = "Go to https://github.com/heyibad/tasker/, and tell me about it."

agent = Agent(
    task=task,
    llm=model,
    browser=browser,
    browser_context=browser_context,
)

async def main():
  try:
      # Run the agent
      print("Running the agent...")
      await agent.run()
      print("Task completed!")
      
  except Exception as e:
      print(f"An error occurred: {e}")
  finally:
      # Clean up resources
      if browser:
          await browser.close()
          print("Browser closed")
      if session:
          client.sessions.release(session.id)
          print("Session released")
      print("Done!")

# Run the async main function
if __name__ == '__main__':
    asyncio.run(main())