from datetime import UTC, datetime

from sqlalchemy import Index, ForeignKey, ARRAY, String
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.db.models import Base

# 1. Only store hashes of apikey
# 2. For now make tokens with null expires_at (meaning forever)
# 3. Save a prefix that is part of the key to help recgnize types


class APIKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    hashed_key: Mapped[str] = mapped_column()
    key_prefix: Mapped[str] = mapped_column()
    manager_id = mapped_column(ForeignKey("users.id"))
    scopes: Mapped[list[str]] = mapped_column(ARRAY(String), nullable=False, server_default='{}')
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(default=datetime.now(UTC))
    updated_at: Mapped[datetime] = mapped_column(
        default=datetime.now(UTC), onupdate=datetime.now(UTC)
    )
    expires_at: Mapped[datetime] = mapped_column(nullable=True)
    last_used_at: Mapped[datetime] = mapped_column(nullable=True)

    manager = relationship("User", back_populates="api_keys")

    __table_args__ = (
        Index("ix_api_keys_manager_id", "manager_id"),
        Index("ix_api_keys_hashed_key", "hashed_key"),
        )