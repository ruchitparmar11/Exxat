import json
import csv
from io import StringIO
from typing import List
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
        if 'ticket_id' not in item or 'text' not in item:
            continue # Skip invalid rows
            
        ticket_id = str(item['ticket_id'])
        text = str(item['text'])
        timestamp = str(item.get('timestamp', ''))
        
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
            actual_category=item.get('actual_category'),
            actual_sentiment=item.get('actual_sentiment')
        )
        db.add(db_ticket)
        processed_count += 1
        
    db.commit()
    
    return {"message": "Upload successful", "processed": processed_count}

@app.get("/api/v1/tickets", response_model=List[schemas.Ticket])
def get_tickets(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieve processed tickets.
    """
    tickets = db.query(models.TicketRecord).offset(skip).limit(limit).all()
    return tickets

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
    
    # 2. Aggregate Sentiments
    sentiments = [t.sentiment for t in tickets if t.sentiment]
    sentiment_counts = Counter(sentiments)
    sentiment_distribution = dict(sentiment_counts)
    
    # 3. Extract Recurring Topics (Trends) using Unsupervised Topic Modeling
    # Only analyze texts from "Negative" sentiment or specific categories like "Bug" or "Product Gap"
    # to find actionable issues.
    actionable_texts = [
        t.text for t in tickets 
        if t.sentiment == "Negative" or t.category in ["Bug", "Product Gap", "Training Need", "Other"]
    ]
    
    recurring_topics = []
    if len(actionable_texts) >= 5:
        topics = extract_trends(actionable_texts, n_topics=4, n_top_words=4)
        recurring_topics = topics
        
    return schemas.TrendResponse(
        total_tickets=total_tickets,
        sentiment_distribution=sentiment_distribution,
        top_categories=top_categories,
        recurring_topics=recurring_topics
    )
