from pydantic import BaseModel
from typing import Optional, List

class TicketBase(BaseModel):
    ticket_id: str
    text: str
    timestamp: str
    actual_category: Optional[str] = None
    actual_sentiment: Optional[str] = None
    standard_fields: Optional[dict] = None

class TicketCreate(TicketBase):
    pass

class TicketAnalyzeRequest(BaseModel):
    text: str

class TicketAnalyzeResponse(BaseModel):
    sentiment: str
    tone: str
    sentiment_score: float
    sentiment_reason: str
    category: str
    standard_fields: dict

class Ticket(TicketBase):
    id: int
    category: Optional[str] = None
    sentiment: Optional[str] = None
    sentiment_score: Optional[float] = None
    sentiment_reason: Optional[str] = None

    class Config:
        from_attributes = True

class TrendSummary(BaseModel):
    category: str
    count: int
    percentage: float

class CategorySentiment(BaseModel):
    positive: int = 0
    neutral: int = 0
    negative: int = 0
    positive_tickets: List[str] = []
    neutral_tickets: List[str] = []
    negative_tickets: List[str] = []

class TrendResponse(BaseModel):
    total_tickets: int
    sentiment_distribution: dict
    top_categories: List[TrendSummary]
    category_sentiments: dict # Maps category name to CategorySentiment
    product_gap_trends: List[dict]
    training_need_trends: List[dict]
