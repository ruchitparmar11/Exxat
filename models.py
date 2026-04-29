from sqlalchemy import Column, Integer, String, DateTime, Text, Float
from database import Base
import datetime

class TicketRecord(Base):
    __tablename__ = "tickets"

    id = Column(Integer, primary_key=True, index=True)
    ticket_id = Column(String, unique=True, index=True)
    text = Column(Text)
    timestamp = Column(String) # Storing as string ISO format for simplicity
    
    # NLP extracted fields
    category = Column(String, index=True, nullable=True)
    sentiment = Column(String, index=True, nullable=True)
    sentiment_score = Column(Float, nullable=True)
    
    # Optional fields for ground truth if uploaded from training sets
    actual_category = Column(String, nullable=True)
    actual_sentiment = Column(String, nullable=True)
