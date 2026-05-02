from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import re
import os
import joblib

# Load NLP Models
sia = SentimentIntensityAnalyzer()

# Attempt to load trained ML Classifier for categories
ML_MODEL_PATH = "category_model.pkl"
ml_classifier = None
if os.path.exists(ML_MODEL_PATH):
    try:
        ml_classifier = joblib.load(ML_MODEL_PATH)
        print("Successfully loaded trained Machine Learning model for categorization!")
    except Exception as e:
        print(f"Could not load ML model: {e}")

# Default Taxonomy for Ticket Classification (Fallback)
TAXONOMY = {
    "Training Need": ["how to", "tutorial", "documentation", "confused", "can't figure out", "help me understand", "guide"],
    "Product Gap": ["wish", "missing feature", "why isn't there", "add", "would be great if", "doesn't have"],
    "Bug": ["error", "broken", "crash", "not loading", "failed", "bug", "doesn't work"],
    "Inquiry": ["when", "billing", "invoice", "renew", "subscription", "cost", "price"],
    "Feedback": ["thank you", "great", "awesome", "love", "hate", "terrible"]
}

def analyze_sentiment(text: str):
    """
    Returns sentiment category and compound score using VADER.
    Includes a custom heuristic to catch urgency and frustration
    so that Tone is highly accurate for clients.
    """
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    # Custom Tone/Urgency Modifiers for Customer Support
    text_lower = text.lower()
    frustration_keywords = ['urgent', 'frustrated', 'unacceptable', 'asap', 'immediately', 
                           'disappointed', 'terrible', 'awful', 'cancel', 'refund', 'angry', 
                           'stuck', 'blocking', 'escalate', 'broken']
    
    found_frustration = []
    for kw in frustration_keywords:
        if re.search(r'\b' + kw + r'\b', text_lower):
            compound -= 0.5  # Push score strongly negative
            found_frustration.append(kw)
            
    # Cap compound score at -1.0
    compound = max(-1.0, compound)
    
    # Extract VADER words for reason
    words = re.findall(r'\b\w+\b', text_lower)
    pos_words = [w for w in set(words) if w in sia.lexicon and sia.lexicon[w] > 0]
    neg_words = [w for w in set(words) if w in sia.lexicon and sia.lexicon[w] < 0]

    if compound >= 0.05:
        category = "Positive"
        reason = "Language indicates positive sentiment."
        if pos_words:
            reason += f" Detected positive keywords: {', '.join(pos_words)}."
    elif compound <= -0.05:
        category = "Negative"
        reason_parts = []
        if found_frustration:
            reason_parts.append(f"Frustration/urgency keywords: {', '.join(found_frustration)}.")
        if neg_words:
            reason_parts.append(f"Negative keywords: {', '.join(neg_words)}.")
        if not reason_parts:
            reason_parts.append("Language indicates negative sentiment.")
        reason = " ".join(reason_parts)
    else:
        category = "Neutral"
        reason = "Language is neutral; no strong positive or negative sentiment detected."
        
    return category, compound, reason

def assign_category(text: str):
    """
    Assign category using trained ML model if available.
    Otherwise, fallback to heuristic rule-based tagger.
    """
    if ml_classifier is not None:
        return ml_classifier.predict([text])[0]
        
    # --- FALLBACK HEURISTIC ---
    text_lower = text.lower()
    
    # Priority matching: check which category has the most keyword hits
    category_scores = {cat: 0 for cat in TAXONOMY.keys()}
    
    for category, keywords in TAXONOMY.items():
        for kw in keywords:
            if re.search(r'\b' + re.escape(kw) + r'\b', text_lower):
                category_scores[category] += 1
                
    # Find category with max score
    best_category = max(category_scores, key=category_scores.get)
    
    if category_scores[best_category] > 0:
        return best_category
    return "Other"

def extract_trends(texts, n_topics=3, n_top_words=5):
    """
    Extracts recurring issues using TF-IDF and NMF (Topic Modeling).
    """
    if len(texts) < n_topics:
        return [{"topic_id": 0, "keywords": ["Not enough data for clustering"]}]

    # 1. TF-IDF Vectorization
    # Clean text to remove chat timestamps like (4:14:43 PM) and common names/signatures
    cleaned_texts = []
    for text in texts:
        # Remove timestamps
        text = re.search(r'(?s)(?:\[.*?\]|\(.*?\)).*?(?=\(.*?\)|\[.*?\]|$)', text) and re.sub(r'\(.*?\)|\[.*?\]|\b(AM|PM)\b|\b\d{1,2}:\d{2}:\d{2}\b', ' ', text) or text
        # Remove common email fluff
        text = re.sub(r'(?i)\b(regards|hi|hello|thank you|thanks|email|com|edu|http|https|www|subject|caution|external|mail)\b', ' ', text)
        cleaned_texts.append(text)
        
    # We use english stop words to filter out "the", "and", etc.
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(cleaned_texts)
    
    # 2. NMF Topic Modeling
    nmf_model = NMF(n_components=n_topics, random_state=1, l1_ratio=.5, init='nndsvd').fit(tfidf)
    
    # 3. Extract top words for each topic
    tfidf_feature_names = tfidf_vectorizer.get_feature_names_out()
    
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [tfidf_feature_names[i] for i in top_features_ind]
        topics.append({
            "topic_id": topic_idx + 1,
            "keywords": top_features,
            "theme_name": " / ".join(top_features[:3]) # Auto-generate a theme name based on top 3 words
        })
        
    return topics

def extract_standard_fields(text: str, category: str, sentiment: str, tone: str):
    text_lower = text.lower()
    
    # Priority Heuristic
    priority = "Normal"
    if tone == "High" or "urgent" in text_lower or "escalate" in text_lower:
        priority = "High"
    if "asap" in text_lower or "immediately" in text_lower or "blocking" in text_lower:
        priority = "Urgent"
        
    # Ticket Summary (First 15 words)
    words = text.split()
    summary = " ".join(words[:15]) + "..." if len(words) > 15 else text
    
    # Product Heuristic
    product = "Exxat One" if "exxat one" in text_lower else "Exxat Prism" if "prism" in text_lower else "Unknown"
    
    # Severity Heuristic
    severity = "High" if priority in ["High", "Urgent"] else "Low"
    
    return {
        "Form": "Standard Support Form",
        "Product": product,
        "Severity": severity,
        "Priority": priority,
        "Level": "Level 1",
        "Sub Level - Level0": "N/A",
        "Domain": "N/A",
        "Main Category": category,
        "Sub Category": "Login Issue" if "login" in text_lower else "N/A",
        "Implementation Phase": "N/A",
        "Tenant ID": "N/A",
        "Reasons for SLA Missed": "N/A",
        "Reason to reopen": "N/A",
        "Ticket summary": summary,
        "Is Summary accurate?": "Yes",
        "Ticket Raised For": "End User"
    }

def process_ticket(text: str):
    """
    Process a single ticket to get category, sentiment, and standard fields.
    """
    sentiment, score, reason = analyze_sentiment(text)
    category = assign_category(text)
    
    tone_map = {
        "Negative": "High",
        "Neutral": "Medium",
        "Positive": "Low"
    }
    tone = tone_map.get(sentiment, "Medium")
    
    standard_fields = extract_standard_fields(text, category, sentiment, tone)
    
    return {
        "sentiment": sentiment,
        "sentiment_score": score,
        "sentiment_reason": reason,
        "category": category,
        "tone": tone,
        "standard_fields": standard_fields
    }
# Model loaded correctly
