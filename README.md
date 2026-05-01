# Zendesk Insight Engine

A privacy-first, real-time Machine Learning and NLP engine designed specifically for analyzing Zendesk support tickets. This system runs 100% locally (no third-party LLM APIs) to ensure complete data privacy for sensitive client interactions.

## 🚀 Features

### 1. Real-Time Machine Learning Categorization
Instead of relying on rigid keyword rules, this engine uses a **Supervised Machine Learning Classifier** (`LinearSVC` via `scikit-learn`). 
* It is trained directly on your historical Zendesk CSV exports.
* It learns your exact internal taxonomy (e.g., `schedules`, `login`, `student_onboarding`, `exploring_internships`).
* When a new ticket arrives, it predicts the correct category instantly (in milliseconds).

### 2. Intelligent Tone & Urgency Detection
It uses VADER Sentiment Analysis upgraded with **Customer Support Urgency Heuristics**.
* **Negative Sentiment ➡️ High Tone**: Automatically flags frustrated or angry clients as High Priority.
* **Neutral Sentiment ➡️ Medium Tone**: Standard questions and updates.
* **Positive Sentiment ➡️ Low Tone**: Polite inquiries or "thank you" messages.
* **Urgency Override**: If a client uses polite words but includes urgency keywords (e.g., "urgent", "unacceptable", "escalate", "blocking"), the engine forcefully overrides the math to guarantee a **High Tone** escalation.

### 3. Clean Topic Modeling (Trend Extraction)
Uses Unsupervised NLP (TF-IDF & NMF) to group tickets and find recurring themes.
* Includes a robust regex cleaner to strip out Zendesk chat timestamps (e.g., `(4:14:43 PM)`), system metadata, and common email fluff (e.g., "Regards", ".com").
* Extracts sharp, actionable keywords for Product Gaps and Training Needs.

---

## 🛠️ System Architecture

* **Framework:** FastAPI (Python)
* **Database:** SQLite (SQLAlchemy)
* **NLP & ML Stack:** `scikit-learn`, `nltk` (VADER), `pandas`
* **Model Format:** `joblib` (`category_model.pkl`)

### Core Files
* `main.py`: The FastAPI server containing all REST endpoints.
* `nlp_engine.py`: The "Brain". Handles ML model loading, sentiment analysis, tone mapping, and topic modeling.
* `train_model.py`: The script used to train the ML model on your Zendesk CSV.
* `process_zendesk_export.py`: A standalone script to generate Markdown reports (`zendesk_insights.md`) from CSVs.
* `update_db.py`: A utility script to retroactively update old database records using the newest ML model.

---

## 🌐 API Endpoints

Once the server is running (`python -m uvicorn main:app --reload`), access the interactive dashboard at `http://127.0.0.1:8000/docs`.

### 1. Analyze New Ticket (Real-Time)
**`POST /api/v1/tickets/analyze`**
* **Purpose:** Built for Zendesk Webhooks. Send a new ticket's text, and instantly receive the predicted Category, Sentiment, and Tone via the Machine Learning model.
* **Input Payload (JSON):**
  ```json
  {
    "text": "Hello, I cannot find the MedStar location in the Explore section."
  }
  ```
* **Output Response (JSON):**
  ```json
  {
    "sentiment": "Positive",
    "tone": "Low",
    "sentiment_score": 0.4588,
    "category": "exploring_internships"
  }
  ```

### 2. Lookup Processed Ticket
**`GET /api/v1/tickets/{ticket_id}`**
* **Purpose:** Fetch the full details, history, and NLP analysis of a specific Zendesk ticket stored in the local SQLite database.
* **Input:** The `{ticket_id}` parameter in the URL (e.g., `/api/v1/tickets/348975`).
* **Output Response (JSON):**
  ```json
  {
    "ticket_id": "348975",
    "text": "Clinical Site Location...",
    "timestamp": "2026-04-29T19:27:55Z",
    "actual_category": null,
    "actual_sentiment": null,
    "id": 212,
    "category": "exploring_internships",
    "sentiment": "Positive",
    "sentiment_score": 0.9972
  }
  ```

### 3. Fetch All Processed Tickets
**`GET /api/v1/tickets`**
* **Purpose:** Retrieve an array of all tickets currently stored in the database. Supports pagination via query parameters.
* **Parameters:** `?skip=0&limit=100` (Defaults to 100 tickets).
* **Output Response:** Array of ticket JSON objects (same structure as Lookup Processed Ticket).

### 4. Bulk Upload
**`POST /api/v1/tickets/upload`**
* **Purpose:** Upload a raw Zendesk CSV export to process hundreds of tickets at once, train the model, and store them in the database for trend analysis.
* **Input:** `multipart/form-data` containing the Zendesk `.csv` file.
* **Output Response (JSON):**
  ```json
  {
    "message": "Upload successful",
    "processed": 59
  }
  ```

### 5. Fetch Global Trends
**`GET /api/v1/insights/trends`**
* **Purpose:** Returns comprehensive global distributions for Tone, Sentiment, Categories, and auto-generated Unsupervised Topic Models.
* **Output Response (JSON excerpt):**
  ```json
  {
    "total_tickets": 59,
    "sentiment_distribution": {
      "Neutral": 10,
      "Negative": 5,
      "Positive": 44
    },
    "top_categories": [
      {
        "category": "login",
        "count": 22,
        "percentage": 37.28
      }
    ],
    "product_gap_trends": [],
    "training_need_trends": []
  }
  ```

---

## 🧠 How to Train the Model

As you handle more tickets in Zendesk, you can make the AI smarter by re-training it on your latest data.

1. Export your latest tickets from Zendesk as a CSV (must include `Subject`, `Public Comments`, and `Main Category`).
2. Save it to the project folder as `zendesk_tickets_export...csv` (update the filename in `train_model.py` if needed).
3. Run the training script:
   ```bash
   python train_model.py
   ```
4. The script will generate a new `category_model.pkl`. FastAPI will automatically detect the new file and update the real-time endpoint instantly!

---

## 🔒 Privacy Notice
This engine is completely air-gapped. All Machine Learning models run locally on the host CPU. No customer data is ever sent to OpenAI, Google Cloud, or AWS.

---

## 🚀 Future Goals: Phase 2 (Zendesk Integration & Agent Assist)

To evolve this Insight Engine into a fully integrated support tool, the following steps are planned for Phase 2:
* **Webhook Integration:** Develop Zendesk Webhook integration for real-time, event-driven ticket ingestion.
* **Smart Solution Matching:** Implement TF-IDF Cosine Similarity to match incoming tickets with past resolved issues.
* **Agent Assist API:** Create an API endpoint to serve instant historic solutions directly to support agents.
* **UI Integration:** Build and deploy a Zendesk Sidebar App to seamlessly interface with the Insight Engine.
* **Database Migration:** Migrate from SQLite to a production-grade database (MongoDB or PostgreSQL).
* **Cloud Deployment:** Deploy the API to a scalable, secure cloud environment.
