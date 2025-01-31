import pytest
from cryptography.fernet import Fernet
from app import EncryptedStore, InMemoryStore

@pytest.fixture
def encrypted_store():
    key = Fernet.generate_key()
    return EncryptedStore(InMemoryStore(), key)

def test_encryption_roundtrip(encrypted_store):
    original = {"secret": "my password is 12345"}
    encrypted = encrypted_store._encrypt(original)
    decrypted = encrypted_store._decrypt(encrypted)
    assert decrypted == original
    assert encrypted != str(original)  # Ensure encryption changed data

def test_tamper_detection(encrypted_store):
    original = {"email": "user@example.com"}
    encrypted = encrypted_store._encrypt(original)
    
    # Tamper with encrypted data
    tampered = encrypted[:-1] + "x"
    
    with pytest.raises(Exception):
        encrypted_store._decrypt(tampered)

def test_store_operations(encrypted_store):
    # Test full storage workflow
    test_data = {"ssn": "123-45-6789"}
    encrypted_store.put(("user", "1"), "data", test_data)
    
    # Retrieve and verify
    retrieved = encrypted_store.get(("user", "1"), "data")
    assert retrieved == test_data
    
    # Verify raw storage is encrypted
    raw_data = encrypted_store.base_store.get(("user", "1"), "data")
    assert "ssn" not in raw_data 