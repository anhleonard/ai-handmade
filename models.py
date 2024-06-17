from sqlalchemy import Column, ForeignKey, Integer, String, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

# Base class for defining database models
Base = declarative_base()

class Embedding(Base):
    __tablename__ = "embeddings"  # Name of the database table

    id = Column(Integer, primary_key=True)  # Unique identifier (primary key)
    vector = Column(String)  # Assuming vector representation is stored as a string
    store_id = Column(Integer, ForeignKey("stores.id"))