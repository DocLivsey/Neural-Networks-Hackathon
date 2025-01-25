from sqlalchemy import Column, String

from core.database import Base


class MuseumItem(Base):
    __tablename__ = "museum_items"

    img_name = Column(String, index=True, primary_key=True)
    object_id = Column(String)
    name = Column(String)
    description = Column(String, nullable=True)
    group_ = Column(String)