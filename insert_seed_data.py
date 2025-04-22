import asyncio

from app.auth.schemas import APIKeyCreate, Scope
from app.auth.api_key_services import generate_api_key
from app.users.schemas import UserCreate
from app.users.repositories import UserRepository
from app.auth.repositories import APIKeyRepository
from app.db.session import async_session


roles = ['admin', 'api_manager', 'user']
permissions = ['model:inference', 'model:embeddings']
key, key_hash = generate_api_key("testing", 10)
users = [
    {
        "email": "test4_user@gsa.gov",
        "name": "Buck",
        "role": "admin",
    }
]

async def insert(users):

    for user in users:
        user_schema = UserCreate(**user)
        async with async_session() as session:
            new_user = await UserRepository(session).create(user_schema)
        async with async_session() as session:
            token = APIKeyCreate(
                hashed_key=key_hash,
                key_prefix="testing",
                manager_id=new_user.id,
                scopes=[Scope.MODELS_INFERENCE, Scope.MODELS_EMBEDDING]
            )

            await APIKeyRepository(session).create(token)

        print("api_key: remeber this, it can't be recovered:", key)


if __name__ == "__main__":
    asyncio.run(insert(users))