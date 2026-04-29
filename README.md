# Insight Engine 🚀

Insight Engine is a lightweight, high-performance FastAPI application designed to automatically process customer support tickets. It uses **Classical NLP Algorithms** (NLTK & scikit-learn)—meaning it requires **no external LLMs, API keys, or expensive hosted models**.

This engine specializes in extracting actionable insights, categorizing issues into **Product Gaps** and **Training Needs**, and correlating them with **Customer Sentiment Trends**.

## Features
- **Zero-LLM Architecture**: High privacy, zero API costs, and runs locally using optimized algorithms.
- **Sentiment Analysis**: Uses NLTK's VADER to assign sentiment polarity (Positive, Neutral, Negative).
- **Heuristic Taxonomy**: Rule-based categorization into specific buckets like Bugs, Inquiries, Product Gaps, etc.
- **Trend Topic Modeling**: Uses TF-IDF and Non-Negative Matrix Factorization (NMF) to find exact recurring themes inside Product Gaps and Training Needs.
- **Scalable Database**: Built with SQLAlchemy, currently using SQLite for zero-setup execution, but fully decoupled to swap to MongoDB.

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone <your-repo-url>
   cd "Trends engine"
   ```

2. **Install dependencies:**
   Make sure you have Python 3.9+ installed.
   ```bash
   pip install -r requirements.txt
   ```

3. **Download NLP data:**
   ```bash
   python -c "import nltk; nltk.download('vader_lexicon'); nltk.download('punkt')"
   ```

## Running the Application

1. Start the FastAPI development server:
   ```bash
   python -m uvicorn main:app --reload
   ```

2. Open your browser to access the interactive API dashboard:
   [http://localhost:8000/docs](http://localhost:8000/docs)

## API Endpoints
- `POST /api/v1/tickets/upload`: Upload your `.csv` or `.json` file containing `{ticket_id, text}` arrays.
- `GET /api/v1/insights/trends`: Get the global analytics, broken down into specific themes for Product Gaps and Training Needs, correlated with sentiment.
- `GET /api/v1/tickets`: View the raw processed tickets in the database.

## Future Architecture Roadmap
Our ultimate goal is to integrate this engine directly with Zendesk via Webhooks and build a Zendesk Sidebar App. When an agent opens a ticket, the app will query this engine's database using TF-IDF Cosine Similarity to find past resolved tickets and instantly suggest the historic solution to the agent.
