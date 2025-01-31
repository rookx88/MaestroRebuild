from langchain_core.messages import HumanMessage

def test_encrypted_memory_flow():
    from app import chat_interface, graph, encrypted_store
    test_input = "My credit card is 4111-1111-1111-1111"
    
    # Simulate user input
    result = graph.invoke(
        {"messages": [HumanMessage(content=test_input)]},
        {"configurable": {"user_id": "test_user"}}
    )
    
    # Verify storage
    memories = encrypted_store.search(("instructions", "test_user"))
    assert any("credit card" in str(m) for m in memories)
    
    # Verify raw encryption
    raw_data = encrypted_store.base_store.search(("instructions", "test_user"))
    assert "4111" not in str(raw_data) 