import json
import csv
from io import StringIO
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
from collections import Counter

from database import engine, Base, get_db
import models
import schemas
from nlp_engine import process_ticket, extract_trends

# Initialize DB
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Insight Engine",
    description="Automatically tags tickets and extracts trends without LLMs.",
    version="1.0.0"
)

@app.post("/api/v1/tickets/analyze", response_model=schemas.TicketAnalyzeResponse)
def analyze_single_ticket(request: schemas.TicketAnalyzeRequest):
    """
    Real-time endpoint to analyze a single incoming ticket.
    Returns sentiment, tone, and category suggestions.
    """
    nlp_result = process_ticket(request.text)
    return schemas.TicketAnalyzeResponse(
        sentiment=nlp_result['sentiment'],
        tone=nlp_result['tone'],
        sentiment_score=nlp_result['sentiment_score'],
        sentiment_reason=nlp_result['sentiment_reason'],
        category=nlp_result['category'],
        standard_fields=nlp_result['standard_fields']
    )

@app.post("/api/v1/tickets/upload", response_model=dict)
async def upload_tickets(file: UploadFile = File(...), db: Session = Depends(get_db)):
    """
    Upload a JSON or CSV file containing tickets.
    Format expected: Array of objects with 'ticket_id' and 'text'.
    """
    if not file.filename.endswith(('.json', '.csv')):
        raise HTTPException(status_code=400, detail="Only JSON and CSV files are supported.")
        
    contents = await file.read()
    tickets_data = []
    
    if file.filename.endswith('.json'):
        try:
            tickets_data = json.loads(contents.decode("utf-8"))
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format.")
    elif file.filename.endswith('.csv'):
        try:
            csv_reader = csv.DictReader(StringIO(contents.decode("utf-8")))
            for row in csv_reader:
                tickets_data.append(row)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Error reading CSV: {str(e)}")

    processed_count = 0
    
    for item in tickets_data:
        # Check standard fields or Zendesk fields
        ticket_id = item.get('ticket_id') or item.get('Ticket ID')
        
        # Map text from standard field or Zendesk fields
        text = item.get('text')
        if not text:
            subject = str(item.get('Subject', ''))
            comments = str(item.get('Public Comments', ''))
            text = f"{subject}\n{comments}".strip()
            
        if not ticket_id or not text:
            continue # Skip invalid rows
            
        ticket_id = str(ticket_id)
        timestamp = str(item.get('timestamp') or item.get('Created At') or '')
        
        # Check if exists
        existing = db.query(models.TicketRecord).filter(models.TicketRecord.ticket_id == ticket_id).first()
        if existing:
            continue
            
        # Process with NLP Engine
        nlp_result = process_ticket(text)
        
        # Save to DB
        db_ticket = models.TicketRecord(
            ticket_id=ticket_id,
            text=text,
            timestamp=timestamp,
            category=nlp_result['category'],
            sentiment=nlp_result['sentiment'],
            sentiment_score=nlp_result['sentiment_score'],
            sentiment_reason=nlp_result['sentiment_reason'],
            standard_fields=nlp_result['standard_fields'],
            actual_category=item.get('actual_category'),
            actual_sentiment=item.get('actual_sentiment')
        )
        db.add(db_ticket)
        processed_count += 1
        
    db.commit()
    
    return {"message": "Upload successful", "processed": processed_count}

@app.get("/api/v1/tickets", response_model=List[schemas.Ticket])
def get_tickets(sentiment: Optional[str] = None, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieve processed tickets. Optional filter by sentiment.
    """
    query = db.query(models.TicketRecord)
    if sentiment:
        query = query.filter(models.TicketRecord.sentiment == sentiment)
    tickets = query.offset(skip).limit(limit).all()
    return tickets

@app.get("/api/v1/tickets/{ticket_id}", response_model=schemas.Ticket)
def get_ticket_by_id(ticket_id: str, db: Session = Depends(get_db)):
    """
    Retrieve a specific ticket by its Zendesk Ticket ID.
    """
    ticket = db.query(models.TicketRecord).filter(models.TicketRecord.ticket_id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found in database.")
    
    # We need to compute tone dynamically or add it to the model. Since it's dynamic based on sentiment, we can just return it.
    # Wait, the schemas.Ticket needs tone if we want to return it. Let's make sure tone is either in the schema or we return a dict.
    # I'll just return the ticket as is; the client will see sentiment, and if they need tone they know the mapping.
    return ticket

@app.get("/api/v1/insights/trends", response_model=schemas.TrendResponse)
def get_trends(db: Session = Depends(get_db)):
    """
    Aggregates trends across all processed tickets.
    """
    tickets = db.query(models.TicketRecord).all()
    if not tickets:
        raise HTTPException(status_code=404, detail="No tickets found to analyze. Please upload some first.")
        
    total_tickets = len(tickets)
    
    # 1. Aggregate Categories
    categories = [t.category for t in tickets if t.category]
    cat_counts = Counter(categories)
    top_categories = [
        schemas.TrendSummary(category=cat, count=count, percentage=round(count/total_tickets * 100, 2))
        for cat, count in cat_counts.most_common()
    ]
    
    # 2. Aggregate Sentiments Global
    sentiments = [t.sentiment for t in tickets if t.sentiment]
    sentiment_counts = Counter(sentiments)
    
    sentiment_distribution = {}
    for s, count in sentiment_counts.items():
        sentiment_distribution[s] = {"count": count, "ticket_ids": []}
        
    for t in tickets:
        if t.sentiment in sentiment_distribution:
            sentiment_distribution[t.sentiment]["ticket_ids"].append(t.ticket_id)
    
    # 3. Sentiment by Category Breakdown
    category_sentiments = {}
    for cat in cat_counts.keys():
        category_sentiments[cat] = {
            "positive": 0, "neutral": 0, "negative": 0,
            "positive_tickets": [], "neutral_tickets": [], "negative_tickets": []
        }
        
    for t in tickets:
        if t.category and t.sentiment:
            s = t.sentiment.lower()
            if s in ["positive", "neutral", "negative"]:
                category_sentiments[t.category][s] += 1
                category_sentiments[t.category][f"{s}_tickets"].append(t.ticket_id)
            
    # 4. Extract Specific Trends for Product Gaps
    product_gap_texts = [t.text for t in tickets if t.category == "Product Gap"]
    product_gap_trends = []
    if len(product_gap_texts) >= 3:
        product_gap_trends = extract_trends(product_gap_texts, n_topics=3, n_top_words=4)
        
    # 5. Extract Specific Trends for Training Needs
    training_need_texts = [t.text for t in tickets if t.category == "Training Need"]
    training_need_trends = []
    if len(training_need_texts) >= 3:
        training_need_trends = extract_trends(training_need_texts, n_topics=3, n_top_words=4)
        
    return schemas.TrendResponse(
        total_tickets=total_tickets,
        sentiment_distribution=sentiment_distribution,
        top_categories=top_categories,
        category_sentiments=category_sentiments,
        product_gap_trends=product_gap_trends,
        training_need_trends=training_need_trends
    )
