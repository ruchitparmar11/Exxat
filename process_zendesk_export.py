import csv
import json
from collections import Counter
from nlp_engine import process_ticket, extract_trends

CSV_FILE = 'zendesk_tickets_export_20260501_090018.csv'

def main():
    tickets = []
    print(f"Reading {CSV_FILE}...")
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            subject = row.get('Subject', '')
            comments = row.get('Public Comments', '')
            text = f"{subject}\n{comments}".strip()
            ticket_id = row.get('Ticket ID')
            if ticket_id and text:
                tickets.append({
                    "ticket_id": ticket_id,
                    "text": text,
                    "requester": row.get('Requester Email', '')
                })

    print(f"Found {len(tickets)} valid tickets. Processing...")
    
    results = []
    categories = []
    sentiments = []
    tones = []
    
    for t in tickets:
        nlp_res = process_ticket(t["text"])
        t.update(nlp_res)
        results.append(t)
        categories.append(nlp_res['category'])
        sentiments.append(nlp_res['sentiment'])
        tones.append(nlp_res['tone'])

    print("Generating report...")
    cat_counts = Counter(categories)
    sent_counts = Counter(sentiments)
    tone_counts = Counter(tones)
    
    report = ["# Zendesk Tickets Insights Report\n"]
    report.append(f"**Total Tickets Analyzed:** {len(tickets)}\n")
    
    report.append("## Sentiment Distribution")
    for s, c in sent_counts.most_common():
        report.append(f"- **{s}:** {c} ({round(c/len(tickets)*100, 1)}%)")
        
    report.append("\n## Tone Distribution (High, Medium, Low)")
    for t, c in tone_counts.most_common():
        report.append(f"- **{t} Tone:** {c} ({round(c/len(tickets)*100, 1)}%)")
        
    report.append("\n## Category Distribution")
    for cat, c in cat_counts.most_common():
        report.append(f"- **{cat}:** {c} ({round(c/len(tickets)*100, 1)}%)")
        
    # Analyze trends for categories with enough tickets
    report.append("\n## Recurring Trends (Topic Modeling)")
    for cat, c in cat_counts.most_common():
        if c >= 3:
            cat_texts = [t["text"] for t in results if t["category"] == cat]
            trends = extract_trends(cat_texts, n_topics=2, n_top_words=4)
            report.append(f"\n### Trends in '{cat}'")
            for tr in trends:
                report.append(f"- **Theme:** {tr['theme_name']}")
                report.append(f"  - Keywords: {', '.join(tr['keywords'])}")
                
    with open('zendesk_insights.md', 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))
        
    print("Report saved to zendesk_insights.md")

if __name__ == '__main__':
    main()
