import hashlib
import secrets

def generate_api_key(prefix: str, num_bytes: int = 32) -> tuple[str, str]:
    """
    Generates a new API key with a prefix. The prefix is primarily to help users recognize keys
    when we have different environments
    """
    random_part = secrets.token_urlsafe(num_bytes)
    api_key = f"{prefix}_{random_part}"

    hashed_key = hashlib.sha256(api_key.encode('utf-8')).hexdigest()

    return api_key, hashed_key


def verify_api_key(provided_key: str, stored_hash: str) -> bool:
    """
    Verifies a provided API key against a stored hash.
    """
    provided_key_hash = hashlib.sha256(provided_key.encode('utf-8')).hexdigest()
    return secrets.compare_digest(provided_key_hash, stored_hash)


