from langgraph.graph import StateGraph, END

from src.state import RAGCoTState
from src.nodes.planner import plan_steps
from src.nodes.retrieve import retrieve_per_step
from src.nodes.responder import generate_answer
import os
from dotenv import load_dotenv
load_dotenv()

os.environ["langsmith_api_key"] = os.getenv("langsmith_api_key")
# -------------------------------
# 4. LangGraph Graph
# -------------------------------
builder = StateGraph(RAGCoTState)
builder.add_node("planner", plan_steps)
builder.add_node("retriever", retrieve_per_step)
builder.add_node("responder", generate_answer)

builder.set_entry_point("planner")
builder.add_edge("planner", "retriever")
builder.add_edge("retriever", "responder")
builder.add_edge("responder", END)

graph = builder.compile()
