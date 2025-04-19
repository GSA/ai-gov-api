import asyncio

from app.schema.user import UserCreate
from app.schema.api_key import APIKeyCreate
from app.repositories.users import UserRepository
from app.repositories.api_keys import APIKeyRepository
from app.auth.scopes import Scope
from app.db.session import async_session


roles = ['admin', 'api_manager', 'user']
permissions = ['model:inference', 'model:embeddings']

users = [
    {
        "email": "mark.meyer@gsa.gov",
        "name": "Mark",
        "role": "admin",
        "api_key_values":"60d3b81a-4dac-4330-a908-22c693bc9845"
    }
]

async def insert(users):

    for user in users:
        user_schema = UserCreate(**user)
        async with async_session() as session:
            new_user = await UserRepository(session).create(user_schema)
        async with async_session() as session:
            token = APIKeyCreate(
                key_value=user['api_key_values'],
                manager_id=new_user.id,
                scopes=[Scope.MODELS_CLAUDE_3_5_INFERENCE]
            )

            new_token = await APIKeyRepository(session).create(token)
        print("inserted:", new_user, new_token)


if __name__ == "__main__":
    asyncio.run(insert(users))