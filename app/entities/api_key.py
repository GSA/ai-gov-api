import uuid
from datetime import UTC, datetime

from sqlalchemy import Index, ForeignKey, ARRAY, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.entities.base import Base

## Open questions:
# 1. Do tokens have fine-grained permissions?
# 2. Tokens have managers, we probably shouldn't delete them when 
#    managers leave, but we also shouldn't have orphaned tokens
# 3. Should we only store hashes of tokens
# 4. Should we be able to make tokens with null expires_at (meaning forever)

class APIKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    key_value: Mapped[uuid.UUID] = mapped_column()
    manager_id = mapped_column(ForeignKey("users.id"))
    scopes: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, server_default='{}')
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now(UTC), onupdate=datetime.now(UTC)
    )
    expires_at: Mapped[datetime] = mapped_column(nullable=True)

    manager = relationship("User", back_populates="api_keys")

    __table_args__ = (
        Index("ix_api_keys_manager_id", "manager_id"),
        Index("ix_api_keys_key_value", "key_value"),
        )