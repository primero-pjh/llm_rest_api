from sqlalchemy import Column, Integer, String, Boolean, DateTime, Date, Time, Text, ForeignKey, Enum, func
from sqlalchemy.orm import relationship
import enum
from app.core.database import Base


class EventRepeatType(enum.Enum):
    NONE = "none"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    YEARLY = "yearly"


class EventStatus(enum.Enum):
    SCHEDULED = "scheduled"
    CANCELLED = "cancelled"
    COMPLETED = "completed"


class Calendar(Base):
    __tablename__ = "calendars"

    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(200), nullable=False)
    description = Column(Text, nullable=True)
    color = Column(String(7), default="#3788d8")  # HEX 색상 코드

    # 소유자
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # 설정
    is_default = Column(Boolean, default=False)
    is_public = Column(Boolean, default=False)

    # 타임스탬프
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # 관계 설정
    owner = relationship("User", back_populates="calendars")
    events = relationship("CalendarEvent", back_populates="calendar", cascade="all, delete-orphan")

    def __repr__(self):
        return f"<Calendar(id={self.id}, name='{self.name}', owner_id={self.owner_id})>"


class CalendarEvent(Base):
    __tablename__ = "calendar_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(300), nullable=False)
    description = Column(Text, nullable=True)
    location = Column(String(500), nullable=True)

    # 일정 시간
    start_date = Column(Date, nullable=False)
    end_date = Column(Date, nullable=False)
    start_time = Column(Time, nullable=True)  # 종일 이벤트는 None
    end_time = Column(Time, nullable=True)
    is_all_day = Column(Boolean, default=False)

    # 반복 설정
    repeat_type = Column(Enum(EventRepeatType), default=EventRepeatType.NONE)
    repeat_end_date = Column(Date, nullable=True)

    # 상태
    status = Column(Enum(EventStatus), default=EventStatus.SCHEDULED)

    # 알림 설정 (분 단위, 예: 30 = 30분 전)
    reminder_minutes = Column(Integer, nullable=True)

    # 외래 키
    calendar_id = Column(Integer, ForeignKey("calendars.id", ondelete="CASCADE"), nullable=False)
    creator_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)

    # 타임스탬프
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    # 관계 설정
    calendar = relationship("Calendar", back_populates="events")
    creator = relationship("User", back_populates="calendar_events")

    def __repr__(self):
        return f"<CalendarEvent(id={self.id}, title='{self.title}', start_date={self.start_date})>"
