from pydantic import BaseModel
from typing import Optional, List

class TicketBase(BaseModel):
    ticket_id: str
    text: str
    timestamp: str
    actual_category: Optional[str] = None
    actual_sentiment: Optional[str] = None

class TicketCreate(TicketBase):
    pass

class Ticket(TicketBase):
    id: int
    category: Optional[str] = None
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None

    class Config:
        from_attributes = True

class TrendSummary(BaseModel):
    category: str
    count: int
    percentage: float

class TrendResponse(BaseModel):
    total_tickets: int
    sentiment_distribution: dict
    top_categories: List[TrendSummary]
    recurring_topics: List[dict] # Will hold clustering results from scikit-learn
