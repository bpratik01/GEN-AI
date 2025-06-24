from phi.agent import Agent
from phi.model.groq import Groq
from phi.tools.yfinance import YFinanceTools
from phi.tools.googlesearch import GoogleSearch
from dotenv import load_dotenv
import os

load_dotenv()

web_search_agent = Agent(
    name="web_search_agent",
    role="web researcher",
    model=Groq(id="Llama3-groq-70b-8192-tool-use-preview"),
    tools=[GoogleSearch()],
    instructions=[
        "Search multiple sources for comprehensive information",
        "Prioritize recent results within last 24 hours",
        "Include source URLs, dates, and reliability assessment",
        "Focus on financial news, market analysis, and expert opinions",
        "Cross-reference information from multiple sources",
        "Filter out promotional or biased content"
    ],
    show_tool_calls=True,
    markdown=True
)

financial_agent = Agent(
    name="finance_agent",
    role="financial analyst",
    model=Groq(id="Llama3-groq-70b-8192-tool-use-preview"),
    tools=[YFinanceTools(
        stock_price=True,
        analyst_recommendations=True,
        stock_fundamentals=True,
        company_news=True
    )],
    instructions=[
        "Present financial data in clear, formatted tables",
        "Calculate key financial metrics and ratios",
        "Compare current values with historical averages",
        "Highlight significant changes in stock performance",
        "Include relevant market context and sector analysis",
        "Provide risk assessment based on volatility metrics",
        "Analyze trading volume and price momentum"
    ],
    show_tool_calls=False,
    markdown=True,
    debug_mode=False
)

multi_agent = Agent(
    team=[web_search_agent, financial_agent],
    instructions=[
        "Synthesize information from both financial data and web research",
        "Cross-validate data between sources for accuracy",
        "Present comprehensive analysis combining technical and news data",
        "Prioritize recent market movements and news impact",
        "Highlight discrepancies between market data and news sentiment",
        "Provide clear actionable insights based on combined analysis"
    ],
    show_tool_calls=True,
    markdown=True,
    debug_mode=False
)

response = multi_agent.print_response(
    "What is the current price of Apple stock?",
    stream=True
)

print(response)