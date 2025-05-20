from typing import Annotated
from typing_extensions import TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import START, END, MessageGraph
from langgraph.graph.state import StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage
from langchain_core.runnables import RunnableConfig

import os
from dotenv import load_dotenv
load_dotenv()

os.environ['LANGSMITH_API_KEY'] = os.getenv("LANGSMITH_API_KEY")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")



model = ChatOpenAI(
    temperature=0,
    model="gpt-4o-mini",
    max_tokens=2000,
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

def make_default_graph():
  chat_graph = StateGraph(State)

  def call_model(state):
      return {'messages': [model.invoke(state['messages'])]}

  chat_graph.add_node('agent', call_model)
  chat_graph.add_edge(START, 'agent')
  chat_graph.add_edge('agent', END)

  return chat_graph.compile()

def tool_calling_graph():

  @tool
  def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b

  tool_node = ToolNode([add_numbers])
  model_with_tool = model.bind_tools([add_numbers])

  def call_model(state):
    return {'messages': [model_with_tool.invoke(state['messages'])]}

  def should_continue(state):
    if state['messages'][-1].tool_calls:
      return 'tool'
    else:
      return END

  graph_workflow = StateGraph(State)

  graph_workflow.add_node('agent', call_model)
  graph_workflow.add_node('tool', tool_node)
  
  graph_workflow.add_edge(START, 'agent')
  graph_workflow.add_conditional_edges(
    'agent',
    should_continue,
    {
      'tool': 'tool',  # Explicitly map 'tool' return value to the tool node
      END: END  # Map END to the end state
    }
  )
  graph_workflow.add_edge('tool', 'agent')  # After tool execution, return to agent

  return graph_workflow.compile()

agent = tool_calling_graph()