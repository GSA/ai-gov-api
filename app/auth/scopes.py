from enum import Enum

# This is mostly a placeholder to get a permission model working
# It's not clear that this should not be it's own DB table
# but for now this is is simpler

class Scope(str, Enum):
    # User management
    USERS_READ = "users:read"
    USERS_WRITE = "users:write"
    
    # Inference
    MODELS_INFERENCE = "models:inference"
    MODELS_EMBEDDING = "models:embedding"
        
    # System-level permissions
    ADMIN = "admin"
