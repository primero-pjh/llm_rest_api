from sqlalchemy import Column, Integer, String, Boolean, DateTime, func
from sqlalchemy.orm import relationship
from app.core.database import Base


class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    hashed_password = Column(String(255), nullable=False)

    # 프로필 정보
    full_name = Column(String(200), nullable=True)
    phone = Column(String(20), nullable=True)
    profile_image = Column(String(500), nullable=True)

    # 계정 상태
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    is_superuser = Column(Boolean, default=False)

    # 타임스탬프
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    last_login = Column(DateTime, nullable=True)

    # 관계 설정
    calendars = relationship("Calendar", back_populates="owner", cascade="all, delete-orphan")
    calendar_events = relationship("CalendarEvent", back_populates="creator", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<User(id={self.id}, email='{self.email}', username='{self.username}')>"
