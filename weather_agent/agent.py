from google.adk.agents import Agent
from google.adk.models.lite_llm import LiteLlm # For multi-model support
from .tools import block_keyword_guardrail, block_paris_tool_guardrail, get_weather
from .subagents.greeting_agent.agent import greeting_agent
from .subagents.farewell_agent.agent import farewell_agent
# @title Define the Weather Agent
# Use one of the model constants defined earlier
AGENT_MODEL = "gemini-2.0-flash" # Starting with Gemini

root_agent = Agent(
        name="weather_agent",
        model=AGENT_MODEL,
        description="The main coordinator agent. Handles weather requests and delegates greetings/farewells to specialists.",
        instruction="You are the main Weather Agent coordinating a team. Your primary responsibility is to provide weather information. "
                    "Use the 'get_weather' tool ONLY for specific weather requests (e.g., 'weather in London'). "
                    "You have specialized sub-agents: "
                    "1. 'greeting_agent': Handles simple greetings like 'Hi', 'Hello'. Delegate to it for these. "
                    "2. 'farewell_agent': Handles simple farewells like 'Bye', 'See you'. Delegate to it for these. "
                    "Analyze the user's query. If it's a greeting, delegate to 'greeting_agent'. If it's a farewell, delegate to 'farewell_agent'. "
                    "If it's a weather request, handle it yourself using 'get_weather'. "
                    "For anything else, respond appropriately or state you cannot handle it.",
        tools=[get_weather], # Root agent still needs the weather tool for its core task
        # Key change: Link the sub-agents here!
        sub_agents=[greeting_agent, farewell_agent],
        output_key="last_weather_report", # <<< Auto-save agent's final weather response
        before_model_callback=block_keyword_guardrail,
        before_tool_callback=block_paris_tool_guardrail,
    )