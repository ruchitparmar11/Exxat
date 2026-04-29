from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import re

# Load NLP Models
sia = SentimentIntensityAnalyzer()

# Default Taxonomy for Ticket Classification
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
    """
    scores = sia.polarity_scores(text)
    compound = scores['compound']
    
    if compound >= 0.05:
        category = "Positive"
    elif compound <= -0.05:
        category = "Negative"
    else:
        category = "Neutral"
        
    return category, compound

def assign_category(text: str):
    """
    Heuristic rule-based tagger using taxonomy.
    Converts text to lowercase and searches for keywords.
    """
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
    # We use english stop words to filter out "the", "and", etc.
    tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
    tfidf = tfidf_vectorizer.fit_transform(texts)
    
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

def process_ticket(text: str):
    """
    Process a single ticket to get category and sentiment.
    """
    sentiment, score = analyze_sentiment(text)
    category = assign_category(text)
    
    return {
        "sentiment": sentiment,
        "sentiment_score": score,
        "category": category
    }
