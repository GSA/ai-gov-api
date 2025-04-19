## Running Dev:

Keep uv up-to-date:
```
uv self update
```

## First time here?

This project uses `uv`. You should be able to install it by following the [installation documentation](https://docs.astral.sh/uv/getting-started/installation/). The `curl` shell command is the recommended and fastest way to install uv.

Once installed you can sync the dependencies. This will automatically create a `.venv` folder, but you can generally ignore it while using uv. 
```
uv sync
```

Start the server:

```
uv run fastapi dev
```

Running tests:

```
uv run pytest
```


## Database Migrations
This project uses alembic to manage changes to the database scheme.

Create a migration:
```
uv run alembic revision --autogenerate -m "First Migration"
```

Update tables:
```
uv run alembic upgrade head
```