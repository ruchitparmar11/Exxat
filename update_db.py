from database import SessionLocal
from models import TicketRecord
from nlp_engine import process_ticket

def update_db():
    db = SessionLocal()
    tickets = db.query(TicketRecord).all()
    print(f"Updating {len(tickets)} tickets in the database with the new ML model...")
    
    for t in tickets:
        nlp_res = process_ticket(t.text)
        t.category = nlp_res['category']
        # Sentiment and tone might have updated too
        t.sentiment = nlp_res['sentiment']
        t.sentiment_score = nlp_res['sentiment_score']
        # t.tone is not in DB model, but category and sentiment are.
        
    db.commit()
    db.close()
    print("Database successfully updated!")

if __name__ == "__main__":
    update_db()
