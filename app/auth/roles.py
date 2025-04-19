from enum import Enum

# This is mostly a placeholder to get a permission model working
# It's not clear that this should not be it's own DB table
# but for now this is is simpler
# The idea is that roles will be associated with scopes which 
# they can then add to API keys. For example, an Admin would be
# able to create and api key that can manipulate user data, 
# but a regular user would not.

class Role(str, Enum):
    ADMIN = "admin"
    USER = "user"