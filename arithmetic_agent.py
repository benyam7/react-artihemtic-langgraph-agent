from langchain_core.messages import SystemMessage
from langchain_deepseek import ChatDeepSeek
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition, ToolNode
import os, getpass
from langgraph.checkpoint.memory import MemorySaver

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")
_set_env("DEEPSEEK_API_KEY")

def add(a: int, b: int) -> int:
    """Adds a and b.

    Args:
        a: first int
        b: second int
    """
    return a + b

def multiply(a: int, b: int) -> int:
    """Multiplies a and b.

    Args:
        a: first int
        b: second int
    """
    return a * b

def divide(a: int, b: int) -> float:
    """Divide a and b.

    Args:
        a: first int
        b: second int
    """
    return a / b

def subtract(a: int, b: int) -> int:
    """Subtracts a and b.

    Args:
        a: first int
        b: second int
    """
    return a - b

tools = [add, multiply, divide, subtract]

# Define LLM with bound tools
llm = ChatDeepSeek(model="deepseek-chat")
llm_with_tools = llm.bind_tools(tools)

# System message
sys_msg = SystemMessage(content="You are a helpful assistant tasked with writing performing arithmetic on a set of inputs.")

config = {"configurable": {"thread_id": "1"}}

# Node
def assistant(state: MessagesState):

    return {"messages": [llm_with_tools.invoke([sys_msg] + state["messages"], config=config)]}


# Build graph
builder = StateGraph(MessagesState)

memory = MemorySaver()

builder.add_node("assistant", assistant)
builder.add_node("tools", ToolNode(tools))
builder.add_edge(START, "assistant")
builder.add_conditional_edges(
    "assistant",
    # If the latest message (result) from assistant is a tool call -> tools_condition routes to tools
    # If the latest message (result) from assistant is a not a tool call -> tools_condition routes to END
    tools_condition,
)
builder.add_edge("tools", "assistant")

# Compile graph
react_graph_memory = builder.compile(checkpointer=memory)

