import csv
import json
import random
import uuid
from datetime import datetime, timedelta

def generate_mock_data(num_tickets=100):
    tickets = []
    
    # Define some templates for typical customer issues
    templates = [
        # Product Gaps
        ("I wish you had a feature to {missing_feature}.", "Product Gap", "Negative"),
        ("Why isn't there an option to {missing_feature}?", "Product Gap", "Negative"),
        ("It would be great if the system could {missing_feature}.", "Product Gap", "Neutral"),
        
        # Training Needs
        ("How do I {task}?", "Training Need", "Neutral"),
        ("I'm confused about how to {task}.", "Training Need", "Negative"),
        ("Is there a tutorial for {task}?", "Training Need", "Neutral"),
        ("I can't figure out how to {task}. Please help.", "Training Need", "Negative"),
        
        # Bugs
        ("The {component} is completely broken and crashes.", "Bug", "Negative"),
        ("I get an error message when I try to use {component}.", "Bug", "Negative"),
        ("The {component} is not loading on my screen.", "Bug", "Negative"),
        
        # General/Inquiry
        ("When does my subscription renew?", "Inquiry", "Neutral"),
        ("Thank you for the quick support, the new {component} is great!", "Feedback", "Positive"),
        ("Can I get an invoice for last month?", "Inquiry", "Neutral")
    ]
    
    missing_features = [
        "export reports to Excel automatically",
        "integrate with Slack",
        "add a dark mode",
        "allow multi-user editing",
        "save custom views"
    ]
    
    tasks = [
        "reset my password",
        "add a new team member",
        "configure the dashboard",
        "change my billing details",
        "set up 2FA"
    ]
    
    components = [
        "login page",
        "analytics dashboard",
        "payment gateway",
        "mobile app",
        "notification system"
    ]
    
    # Generate random dates over the past 30 days
    now = datetime.now()
    
    for i in range(num_tickets):
        template, category, sentiment = random.choice(templates)
        
        if "{missing_feature}" in template:
            text = template.format(missing_feature=random.choice(missing_features))
        elif "{task}" in template:
            text = template.format(task=random.choice(tasks))
        elif "{component}" in template:
            text = template.format(component=random.choice(components))
        else:
            text = template
            
        ticket = {
            "ticket_id": str(uuid.uuid4()),
            "text": text,
            "timestamp": (now - timedelta(days=random.randint(0, 30), hours=random.randint(0, 23))).isoformat(),
            "actual_category": category, # Ground truth for testing
            "actual_sentiment": sentiment # Ground truth for testing
        }
        tickets.append(ticket)
        
    return tickets

def main():
    tickets = generate_mock_data(200)
    
    # Save to JSON
    with open('tickets_sample.json', 'w') as f:
        json.dump(tickets, f, indent=2)
        
    # Save to CSV
    with open('tickets_sample.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["ticket_id", "text", "timestamp", "actual_category", "actual_sentiment"])
        writer.writeheader()
        writer.writerows(tickets)
        
    print(f"Generated {len(tickets)} mock tickets in tickets_sample.json and tickets_sample.csv")

if __name__ == "__main__":
    main()
