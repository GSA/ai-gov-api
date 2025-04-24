#!/usr/bin/env python
# -*- coding: utf-8 -*-

import asyncio
import argparse
import sys

try:
    from app.auth.schemas import APIKeyCreate, Scope, Role
    from app.auth.utils import generate_api_key
    from app.users.schemas import UserCreate
    from app.users.repositories import UserRepository
    from app.auth.repositories import APIKeyRepository
    from app.db.session import async_session
except ImportError as e:
    print(f"Error importing application modules: {e}")
    print("Please make sure the script is run from the correct directory "
          "you probably need to have venv activated or"
          "use `uv run` (see Readme)")
    sys.exit(1)

SCOPES = [Scope.MODELS_INFERENCE, Scope.MODELS_EMBEDDING]
KEY_PREFIX = "test_adm"

async def create_admin_user(email: str, name: str, key_length: int):
    """
    Creates an admin user and an associated API key.
    """

    secret_key, key_hash = generate_api_key(KEY_PREFIX, key_length)

    try:
        # Create the user schema
        user_schema = UserCreate(
            email=email,
            name=name,
            role=Role.ADMIN, 
        )

        async with async_session() as session:
            user_repo = UserRepository(session)
            existing_user = await user_repo.get_by_email(email)
            if existing_user:
                print(f"Error: User with email '{email}' already exists.")
                return 

            new_user = await user_repo.create(user_schema)

        async with async_session() as session:
            api_key_schema = APIKeyCreate(
                hashed_key=key_hash,
                key_prefix=KEY_PREFIX,
                manager_id=new_user.id,
                scopes=SCOPES
            )
            await APIKeyRepository(session).create(api_key_schema)


        print("\n" + "=" * 50)
        print("  ADMIN USER AND API KEY CREATED SUCCESSFULLY!")
        print("  Email:", new_user.email)
        print("  Name:", new_user.name)
        print("  User ID:", new_user.id)
        print("\n  IMPORTANT: Save the following API key. It cannot be recovered.")
        print(f"  API Key: {secret_key}")
        print("=" * 50 + "\n")

    except Exception as e:
        print(f"\nError during admin user creation: {e}")
        sys.exit(1)


def main():
  
    parser = argparse.ArgumentParser(
        description="Create an admin user and their API key."
    )

    parser.add_argument(
        "--email",
        type=str,
        required=True,
        help="Email address for the new admin user."
    )
    parser.add_argument(
        "--name",
        type=str,
        required=True,
        help="Full name of the new admin user."
    )
    parser.add_argument(
        "--key-length",
        type=int,
        default=12, 
        help="Length of the secret part of the API key (default: 12)."
    )

    args = parser.parse_args()

    if args.key_length <= 0:
        print("Error: --key-length must be a positive integer.", file=sys.stderr)
        sys.exit(1)

    asyncio.run(create_admin_user(
        email=args.email,
        name=args.name,
        key_length=args.key_length
    ))


if __name__ == "__main__":
    main()
