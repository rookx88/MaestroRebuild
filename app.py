"""AI-Powered Memory Management System"""
# ---------------------------
# 1. IMPORTS & ENVIRONMENT SETUP
# ---------------------------
import os
import getpass
import uuid
from datetime import datetime
from typing import Optional, Literal, TypedDict

from IPython.display import Image, display
from pydantic import BaseModel, Field
from trustcall import create_extractor
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, merge_message_runs
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, MessagesState, END, START
from langgraph.store.base import BaseStore
from langgraph.store.memory import InMemoryStore

# Environment configuration
def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("LANGCHAIN_API_KEY")
os.environ.update({
    "LANGCHAIN_TRACING_V2": "true",
    "LANGCHAIN_PROJECT": "langchain-academy"
})

# ---------------------------
# 2. DATA MODELS
# ---------------------------
class Memory(BaseModel):
    """Conversation memory structure"""
    content: str = Field(description="Main content of the memory")

class Profile(BaseModel):
    """User profile structure"""
    name: Optional[str] = Field(description="User's name", default=None)
    location: Optional[str] = Field(description="User's location", default=None)
    job: Optional[str] = Field(description="User's job", default=None)
    connections: list[str] = Field(
        description="Personal connections",
        default_factory=list
    )
    interests: list[str] = Field(
        description="User interests", 
        default_factory=list
    )

class ToDo(BaseModel):
    """Task management structure"""
    task: str = Field(description="Task description")
    time_to_complete: Optional[int] = Field(description="Estimated minutes to complete")
    deadline: Optional[datetime] = Field(description="Due date", default=None)
    solutions: list[str] = Field(
        description="Actionable solutions",
        min_items=1,
        default_factory=list
    )
    status: Literal["not started", "in progress", "done", "archived"] = Field(
        description="Task status",
        default="not started"
    )

class UpdateMemory(TypedDict):
    """Memory update decision structure"""
    update_type: Literal['user', 'todo', 'instructions']

# ---------------------------
# 3. CORE COMPONENTS
# ---------------------------
class Spy:
    """Tool call inspector"""
    def __init__(self):
        self.called_tools = []

    def __call__(self, run):
        q = [run]
        while q:
            r = q.pop()
            if r.child_runs:
                q.extend(r.child_runs)
            if r.run_type == "chat_model":
                self.called_tools.append(
                    r.outputs["generations"][0][0]["message"]["kwargs"]["tool_calls"]
                )

# Initialize AI components
model = ChatOpenAI(model="gpt-4o", temperature=0)
spy = Spy()

# Create extractors
profile_extractor = create_extractor(
    model,
    tools=[Profile],
    tool_choice="Profile",
)

trustcall_extractor = create_extractor(
    model,
    tools=[Memory],
    tool_choice="Memory",
    enable_inserts=True,
).with_listeners(on_end=spy)

todo_extractor = create_extractor(
    model,
    tools=[ToDo],
    tool_choice="ToDo",
    enable_inserts=True,
).with_listeners(on_end=spy)

# ---------------------------
# 4. HELPER FUNCTIONS
# ---------------------------
def extract_tool_info(tool_calls, schema_name="Memory"):
    """Process tool call data for user feedback"""
    changes = []
    for call_group in tool_calls:
        for call in call_group:
            if call['name'] == 'PatchDoc':
                changes.append({
                    'type': 'update',
                    'doc_id': call['args']['json_doc_id'],
                    'planned_edits': call['args']['planned_edits'],
                    'value': call['args']['patches'][0]['value']
                })
            elif call['name'] == schema_name:
                changes.append({
                    'type': 'new',
                    'value': call['args']
                })
    
    return "\n\n".join(
        f"Document {c['doc_id']} updated:\nPlan: {c['planned_edits']}\nAdded content: {c['value']}" 
        if c['type'] == 'update' else 
        f"New {schema_name} created:\nContent: {c['value']}"
        for c in changes
    )

# ---------------------------
# 5. MEMORY MANAGEMENT WORKFLOW
# ---------------------------
def task_mAIstro(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Main conversation processing logic"""
    user_id = config["configurable"]["user_id"]
    
    # Retrieve stored memories
    profile = store.search(("profile", user_id))[0].value if store.search(("profile", user_id)) else None
    
    # Format ToDos properly
    todo_entries = store.search(("todo", user_id))
    todos = "\n".join(
        f"- {ToDo(**entry.value).task} ({ToDo(**entry.value).status})"
        for entry in todo_entries
    ) if todo_entries else "No current tasks"
    
    instructions = store.search(("instructions", user_id))[0].value if store.search(("instructions", user_id)) else ""
    
    system_msg = f"""You are a helpful chatbot. Maintain and update:
    1. User Profile: {profile or 'No profile information'}
    2. ToDos: {todos}
    3. Instructions: {instructions or 'No custom instructions'}"""
    
    response = model.bind_tools([UpdateMemory]).invoke(
        [SystemMessage(content=system_msg)] + state["messages"]
    )
    return {"messages": [response]}

def update_profile(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Profile update handler"""
    user_id = config["configurable"]["user_id"]
    namespace = ("profile", user_id)
    
    result = profile_extractor.invoke({
        "messages": [SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"][:-1]
    })
    
    for resp, meta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace, meta.get("json_doc_id", str(uuid.uuid4())), resp.model_dump())
    
    return {"messages": [{
        "role": "tool",
        "content": "Profile updated",
        "tool_call_id": state['messages'][-1].tool_calls[0]['id']
    }]}

def update_todos(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """ToDo list update handler"""
    user_id = config["configurable"]["user_id"]
    namespace = ("todo", user_id)
    
    result = todo_extractor.invoke({
        "messages": [SystemMessage(content=TRUSTCALL_INSTRUCTION)] + state["messages"][:-1],
        "existing": [(str(i), "ToDo", m) for i, m in enumerate(store.search(namespace))]
    })
    
    for resp, meta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace, meta.get("json_doc_id", str(uuid.uuid4())), resp.model_dump())
    
    return {"messages": [{
        "role": "tool",
        "content": "ToDos updated",
        "tool_call_id": state['messages'][-1].tool_calls[0]['id']
    }]}

def update_instructions(state: MessagesState, config: RunnableConfig, store: BaseStore):
    """Instruction update handler"""
    user_id = config["configurable"]["user_id"]
    namespace = ("instructions", user_id)
    
    result = trustcall_extractor.invoke({
        "messages": [SystemMessage(content=CREATE_INSTRUCTIONS)] + state["messages"][:-1],
        "existing": [(str(i), "Memory", m) for i, m in enumerate(store.search(namespace))]
    })
    
    for resp, meta in zip(result["responses"], result["response_metadata"]):
        store.put(namespace, meta.get("json_doc_id", str(uuid.uuid4())), resp.model_dump())
    
    return {"messages": [{
        "role": "tool",
        "content": "Instructions updated",
        "tool_call_id": state['messages'][-1].tool_calls[0]['id']
    }]}

def route_message(state: MessagesState, config: RunnableConfig, store: BaseStore) -> Literal["END", "update_todos", "update_instructions", "update_profile"]:
    """Workflow router"""
    if not state['messages'][-1].tool_calls:
        return "END"
    
    update_type = state['messages'][-1].tool_calls[0]['args']['update_type']
    
    # Explicit mapping to match node names
    return {
        'user': "update_profile",
        'todo': "update_todos",  # Map 'todo' type to 'update_todos' node
        'instructions': "update_instructions"
    }[update_type]

# Trustcall system instructions (add to constants section)
TRUSTCALL_INSTRUCTION = """Reflect on following interaction. 
Use the provided tools to retain any necessary memories about the user. 
Use parallel tool calling to handle updates and insertions simultaneously.
System Time: {time}"""

CREATE_INSTRUCTIONS = """Reflect on the following interaction.
Based on this interaction, update your instructions for how to update ToDo list items. 
Use any feedback from the user to update how they like to have items added, etc."""

# ---------------------------
# 6. WORKFLOW SETUP
# ---------------------------
# Graph configuration
builder = StateGraph(MessagesState)

# Add main processing node
builder.add_node("task_mAIstro", task_mAIstro)

# Add update nodes
builder.add_node("update_todos", update_todos)
builder.add_node("update_profile", update_profile)
builder.add_node("update_instructions", update_instructions)

# Set up workflow edges
builder.add_edge(START, "task_mAIstro")  # Critical entry point
builder.add_conditional_edges(
    "task_mAIstro",
    route_message,
    {
        "update_todos": "update_todos",
        "update_profile": "update_profile",
        "update_instructions": "update_instructions",
        "END": END
    }
)

# Connect update nodes back to main processor
builder.add_edge("update_todos", "task_mAIstro")
builder.add_edge("update_profile", "task_mAIstro")
builder.add_edge("update_instructions", "task_mAIstro")

# Memory stores (keep original variable names)
across_thread_memory = InMemoryStore()
within_thread_memory = MemorySaver()

# Compile graph with proper references
graph = builder.compile(
    checkpointer=within_thread_memory,
    store=across_thread_memory
)

# Visualization
display(Image(graph.get_graph(xray=1).draw_mermaid_png()))

# ---------------------------
# 7. USER INTERFACE
# ---------------------------
def chat_interface():
    """Command-line chat interface for interacting with the memory system"""
    print("\n" + "="*50)
    print("AI Memory Assistant".center(50))
    print("="*50)
    print("Type your messages below. Enter 'exit' to quit.\n")
    
    # Initialize session
    session_config = {
        "configurable": {
            "thread_id": str(uuid.uuid4())[:8],  # Short session ID
            "user_id": input("Please enter your user ID to start: ")
        }
    }
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            if user_input.lower() in ('exit', 'quit'):
                print("\nAI: Goodbye! Your memories have been saved.")
                break
                
            # Process through workflow
            print("\nAI is processing...")
            for chunk in graph.stream(
                {"messages": [HumanMessage(content=user_input)]},
                session_config,
                stream_mode="values"
            ):
                if response := chunk.get("messages"):
                    latest_msg = response[-1]
                    print(f"\nAI: {latest_msg.content}")
                    
                    # Show memory updates if any
                    if "tool_calls" in latest_msg.additional_kwargs:
                        print("\n[System] Memory Updated:")
                        print(extract_tool_info(spy.called_tools))
                    
        except KeyboardInterrupt:
            print("\n\nSession ended. Memories preserved for next time.")
            break

if __name__ == "__main__":
    chat_interface()