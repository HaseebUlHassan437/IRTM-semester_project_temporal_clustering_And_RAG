# 🔍 Temporal Document Clustering for E-Discovery

> **AI-Powered Email Analysis System with Gemini Integration**  
> Authors: Haseeb Ul Hassan (MSCS25003)
> Course: Information Retrieval 

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Dash](https://img.shields.io/badge/Dash-2.14.2-blue.svg)](https://dash.plotly.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4.0-orange.svg)](https://scikit-learn.org/)
[![Gemini AI](https://img.shields.io/badge/Gemini-2.5%20Flash-yellow.svg)](https://ai.google.dev/)

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Architecture](#-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Data Pipeline](#-data-pipeline)
- [Technical Details](#-technical-details)
- [API Configuration](#-api-configuration)
- [Screenshots](#-screenshots)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

**Temporal Document Clustering** is an advanced e-discovery system that combines classical Information Retrieval techniques with modern Large Language Models to analyze email datasets. Built specifically for the **Enron Email Dataset**, this system provides:

- **Unsupervised document clustering** using TF-IDF and K-means
- **Temporal analysis** with interactive date range filtering
- **Ranked keyword search** using cosine similarity
- **AI-powered insights** via Google's Gemini 2.5 Flash API
- **Interactive visualizations** with Plotly and Dash

### 🎓 Academic Context

This project demonstrates the integration of:
- **Classical IR**: TF-IDF vectorization, cosine similarity, stopword removal
- **Machine Learning**: K-means clustering, PCA dimensionality reduction
- **Modern NLP**: Stemming, tokenization, n-gram analysis
- **Generative AI**: Context-aware document analysis with Gemini

---

## ✨ Key Features

### 🔬 Core Capabilities

| Feature | Description | Technology |
|---------|-------------|------------|
| **Dynamic Clustering** | Adjust cluster count (3-15) with real-time re-clustering | K-means, scikit-learn |
| **Temporal Filtering** | Filter emails by date range with visual timeline | Plotly Time Series |
| **Ranked Search** | Keyword search with TF-IDF relevance scoring | TF-IDF + Cosine Similarity |
| **AI Analysis** | Ask natural language questions about selected emails | Gemini 2.5 Flash API |
| **Interactive Selection** | Click scatter plot points to select emails for analysis | Dash callbacks |
| **Query History** | Track last 10 AI queries with timestamps | Client-side storage |

### 🎨 Visualization Components

- **Timeline Chart**: Email volume distribution over time
- **Scatter Plot**: 2D PCA projection colored by cluster with relevance-based sizing
- **Cluster Statistics**: Document count per cluster with top terms
- **Document Viewer**: Full email details with metadata
- **AI Insights Panel**: Real-time Gemini responses with formatted output

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Interface (Dash)                    │
│  ┌────────────┐  ┌──────────────┐  ┌──────────────────┐   │
│  │  Controls  │  │ Visualizations│  │  AI Insights     │   │
│  │  Panel     │  │  (Timeline +  │  │  (Gemini Chat)   │   │
│  │            │  │   Scatter)    │  │                  │   │
│  └────────────┘  └──────────────┘  └──────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Processing Layer (Python)                   │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐ │
│  │Preprocessing │  │  Clustering  │  │  Relevance       │ │
│  │(NLTK, Regex) │  │  (K-means)   │  │  Ranking         │ │
│  └──────────────┘  └──────────────┘  └──────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    External Services                         │
│  ┌──────────────────────┐  ┌───────────────────────────┐   │
│  │  Gemini 2.5 Flash    │  │  Local Data Storage       │   │
│  │  (Google AI)         │  │  (CSV files)              │   │
│  └──────────────────────┘  └───────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow

```
Raw Emails (CSV)
    ↓
Preprocessing (clean, tokenize, stem)
    ↓
TF-IDF Vectorization
    ↓
K-means Clustering + PCA Reduction
    ↓
Interactive Dashboard
    ↓
User Queries → Relevance Ranking → Gemini Analysis → Insights
```

---

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Google Gemini API key ([Get one here](https://ai.google.dev/))

### Step 1: Clone the Repository

```bash
git clone https://github.com/HaseebUlHassan437/temporal_document_clustering.git
cd temporal_document_clustering
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Required packages:**
```
pandas==2.2.0
numpy==1.26.4
nltk==3.8.1
spacy==3.7.4
scikit-learn==1.4.0
plotly==5.18.0
dash==2.14.2
dash-bootstrap-components==1.5.0
requests==2.31.0
python-dotenv==1.0.0
```

### Step 3: Download NLTK Data

```python
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords')"
```

### Step 4: Configure Gemini API

Create a `.env` file in the project root:

```bash
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

Or set the environment variable:

```bash
# Windows (PowerShell)
$env:GEMINI_API_KEY="your_api_key_here"

# Linux/Mac
export GEMINI_API_KEY="your_api_key_here"
```

---

## 💻 Usage

### Quick Start

1. **Ensure you have the data file:**
   ```bash
   # The system expects: data/clustered_emails.csv
   # or data/clustered_emails_raw.csv (with raw email bodies)
   ```

2. **Run the dashboard:**
   ```bash
   python app.py
   ```

3. **Open in browser:**
   ```
   http://127.0.0.1:8050/
   ```

4. **Wait for initialization:**
   ```
   ======================================================================
   🚀 DASHBOARD WITH GEMINI AI READY
   ======================================================================
   📊 Documents: 10,000
   🎯 Clusters: 8
   🤖 Gemini Integration: Enabled
   🌐 Open: http://127.0.0.1:8050/
   ======================================================================
   ```

### Basic Workflow

#### 1. **Explore Data**
   - Adjust the **date range slider** to filter by time period
   - Use the **cluster filter dropdown** to focus on specific topics
   - View the **timeline** to see email volume patterns

#### 2. **Search Documents**
   - Enter keywords in the **search box** (e.g., "energy power california")
   - Adjust the **relevance threshold** (0.0 - 0.5) to filter results
   - Set **max results** to control result set size
   - Click **Search** to see ranked results

#### 3. **Analyze with AI**
   - Click points on the **scatter plot** to select emails
   - Navigate to the **AI Insights (Gemini)** tab
   - Ask questions like:
     - "What are the main topics discussed?"
     - "Summarize the key decisions made"
     - "Who are the primary stakeholders?"
   - View responses with **query history**

#### 4. **Re-cluster (Optional)**
   - Adjust the **K slider** (3-15 clusters)
   - Click **Re-Cluster** to recompute groupings
   - View updated cluster terms and visualizations

---

## 📁 Project Structure

```
temporal_document_clustering/
│
├── app.py                          # Main dashboard application (1172 lines)
├── main.py                         # (Reserved for CLI interface)
├── explore_data.py                 # Data exploration utility
├── requirements.txt                # Python dependencies
│
├── src/                            # Core modules
│   ├── __init__.py
│   ├── preprocessing.py            # Email parsing, cleaning, stemming
│   └── clustering.py               # TF-IDF, K-means, PCA
│
├── data/                           # Data files
│   ├── clustered_emails.csv        # Processed emails with clusters
│   ├── clustered_emails_raw.csv    # With original raw email bodies
│   ├── processed_emails.csv        # Intermediate processed data
│   └── enron_extracted/
│       └── emails.csv              # Original Enron dataset
│
├── notebooks/                      # Jupyter notebooks
│   └── 01_extract_data.ipynb       # Data extraction workflow
│
├── scripts/                        # Utility scripts
│   └── merge_raw_into_clustered.py # Merge raw bodies into clustered data
│
├── models/                         # Model artifacts (optional)
│   └── top_terms.json              # Cached cluster terms
│
└── docs/                           # Documentation
    ├── COMPLETION_SUMMARY.md       # Project completion overview
    ├── IMPLEMENTATION_NOTES.md     # Technical implementation details
    ├── QUICK_REFERENCE.md          # Concept explanations
    ├── DEMO_SCRIPT.md              # Presentation walkthrough
    └── NEXT_STEPS.md               # Testing checklist
```

---

## 🔄 Data Pipeline

### Stage 1: Data Extraction

**Tool:** `notebooks/01_extract_data.ipynb` or `explore_data.py`

```python
# Extract emails from Enron archive
python explore_data.py
```

**Output:** `data/enron_extracted/emails.csv`

### Stage 2: Preprocessing

**Module:** `src/preprocessing.py`

**Steps:**
1. Parse email headers (Date, From, To, Subject)
2. Extract body text
3. Clean text (remove URLs, emails, special characters)
4. Tokenize and remove stopwords
5. Apply Porter Stemming

**Code example:**
```python
from src.preprocessing import EmailPreprocessor

preprocessor = EmailPreprocessor()
processed = preprocessor.process_dataset(
    csv_path='data/enron_extracted/emails.csv',
    output_path='data/processed_emails.csv',
    sample_size=10000  # Process 10k emails
)
```

**Output:** `data/processed_emails.csv`

### Stage 3: Clustering

**Module:** `src/clustering.py`

**Steps:**
1. TF-IDF vectorization (max 1000 features, bigrams)
2. K-means clustering (default 8 clusters)
3. PCA dimensionality reduction (2D for visualization)
4. Extract top terms per cluster

**Code example:**
```python
from src.clustering import DocumentClusterer

clusterer = DocumentClusterer(n_clusters=8, max_features=1000)
clusterer.fit(documents['processed_text'])
results = clusterer.get_cluster_results()
```

**Output:** `data/clustered_emails.csv`

### Stage 4: Dashboard & Analysis

**Tool:** `app.py`

- Loads clustered data
- Provides interactive filtering and search
- Integrates Gemini AI for analysis

---

## 🔧 Technical Details

### Preprocessing Techniques

| Technique | Implementation | Purpose |
|-----------|----------------|---------|
| **Email Parsing** | Regex pattern matching | Extract metadata (date, from, to, subject) |
| **Text Cleaning** | Lowercasing, regex substitution | Remove noise (URLs, emails, special chars) |
| **Tokenization** | NLTK `word_tokenize` | Split text into words |
| **Stopword Removal** | NLTK stopwords + custom list | Remove common words (the, and, enron) |
| **Stemming** | Porter Stemmer | Reduce words to root form (running → run) |

### TF-IDF Configuration

```python
TfidfVectorizer(
    max_features=1000,      # Top 1000 terms by importance
    max_df=0.8,             # Ignore terms in >80% of docs
    min_df=5,               # Ignore terms in <5 docs
    ngram_range=(1, 2)      # Unigrams + bigrams
)
```

**Why these settings?**
- `max_features=1000`: Balances vocabulary size vs computational cost
- `max_df=0.8`: Filters out overly common terms
- `min_df=5`: Removes rare/noisy terms
- `ngram_range=(1,2)`: Captures phrases like "energy market"

### K-means Clustering

```python
KMeans(
    n_clusters=8,           # Default 8 topics
    random_state=42,        # Reproducible results
    n_init=10,              # 10 initializations for stability
    max_iter=300            # Max iterations per run
)
```

**Cluster Interpretation:**
- Each cluster represents a topic/theme
- Top terms extracted from cluster centroids
- Example clusters: Legal issues, Energy trading, Internal memos

### Relevance Ranking

**Algorithm:** Cosine Similarity

```python
def compute_relevance_scores(texts, query, vectorizer):
    # Transform texts and query to TF-IDF vectors
    tfidf_matrix = vectorizer.transform(texts + [query])
    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]
    
    # Compute cosine similarity
    scores = cosine_similarity(doc_vecs, query_vec).flatten()
    
    # Normalize to 0-1 range
    return scores / scores.max()
```

**Score Interpretation:**
- `1.0`: Perfect match (all query terms present with high importance)
- `0.5`: Partial match (some terms present)
- `0.0`: No match (query terms absent)

### Gemini API Integration

**Model:** `gemini-2.5-flash`  
**Endpoint:** `https://generativelanguage.googleapis.com/v1beta/`

**Key Features:**
- **Context-aware querying**: Selected emails passed as context
- **Retry logic**: Handles rate limits and timeouts
- **Stateless API**: Full context sent with each query
- **Error handling**: Graceful degradation on API failures

**Example request:**
```python
def query_gemini(prompt, context):
    full_prompt = f"""
    You are an intelligent assistant analyzing email documents.
    
    ## Email Context:
    {context}
    
    ## User Question:
    {prompt}
    
    Answer ONLY based on the email content provided.
    """
    
    response = requests.post(
        f"{GEMINI_API_URL}?key={GEMINI_API_KEY}",
        json={"contents": [{"parts": [{"text": full_prompt}]}]},
        timeout=30
    )
    
    return response.json()['candidates'][0]['content']['parts'][0]['text']
```

---

## 🔑 API Configuration

### Getting a Gemini API Key

1. Visit [Google AI Studio](https://ai.google.dev/)
2. Sign in with your Google account
3. Navigate to **API Keys**
4. Click **Create API Key**
5. Copy the key and store it securely

### Configuration Methods

#### Option 1: Environment Variable (Recommended)

**Windows (PowerShell):**
```powershell
$env:GEMINI_API_KEY="YOUR_API_KEY_HERE"
python app.py
```

**Linux/Mac:**
```bash
export GEMINI_API_KEY="YOUR_API_KEY_HERE"
python app.py
```

#### Option 2: .env File

Create `.env` in project root:
```
GEMINI_API_KEY=YOUR_API_KEY_HERE
```

The application automatically loads this using `python-dotenv`.

#### Option 3: Hard-code (Development Only)

⚠️ **Not recommended for production**

```python
# In app.py, line 34
GEMINI_API_KEY = "YOUR_API_KEY_HERE"  # Replace with actual key
```

### API Limits

**Free Tier:**
- 60 requests per minute
- 1500 requests per day
- Rate limiting handled automatically

---

## 📊 Screenshots

### Main Dashboard
![Dashboard Overview](docs/screenshots/dashboard.png)
*Interactive 3-tab interface with temporal filtering and cluster visualization*

### Ranked Search
![Search Results](docs/screenshots/search.png)
*TF-IDF ranked search with relevance-based point sizing*

### AI Insights
![Gemini Analysis](docs/screenshots/gemini.png)
*Context-aware document analysis with query history*

---

## 🧪 Testing

### Manual Testing Checklist

Run through these steps to verify functionality:

#### ✅ Clustering Test
1. Move **K slider** from 8 to 10
2. Click **Re-Cluster**
3. Verify cluster count updates
4. Check scatter plot refreshes

#### ✅ Temporal Filter Test
1. Adjust **date slider**
2. Verify timeline chart updates
3. Check filtered document count

#### ✅ Search Test
1. Enter `"energy power"`
2. Set threshold to `0.1`
3. Click **Search**
4. Verify ranked results appear

#### ✅ Email Selection Test
1. Click 3 points on scatter plot
2. Verify selected emails list updates
3. Check document details panel

#### ✅ Gemini Test
1. Select emails
2. Go to **AI Insights** tab
3. Ask: `"What are the main topics?"`
4. Verify response appears (3-5 sec)

---

## 📚 Documentation

Additional documentation available:

| Document | Description |
|----------|-------------|
| [IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) | Technical deep dive into implementation |
| [QUICK_REFERENCE.md](QUICK_REFERENCE.md) | Concept explanations and FAQs |
| [DEMO_SCRIPT.md](DEMO_SCRIPT.md) | Step-by-step presentation guide |
| [COMPLETION_SUMMARY.md](COMPLETION_SUMMARY.md) | Project completion overview |
| [NEXT_STEPS.md](NEXT_STEPS.md) | Testing checklist and future work |

---

## 🤝 Contributing

This is an academic project, but suggestions are welcome!

### Development Setup

1. Fork the repository
2. Create a feature branch:
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make changes and test thoroughly
4. Commit with clear messages:
   ```bash
   git commit -m "Add: Feature description"
   ```
5. Push and create a pull request

### Code Style

- Follow PEP 8 guidelines
- Add docstrings to all functions
- Comment complex logic
- Keep functions under 50 lines where possible

---

## 🐛 Known Issues & Limitations

### Current Limitations

1. **Memory Usage**: PCA on large datasets requires converting sparse TF-IDF to dense matrix
   - **Solution**: Use `TruncatedSVD` for datasets >50k documents

2. **API Rate Limits**: Free Gemini tier has 60 req/min limit
   - **Solution**: Implement request queuing or upgrade to paid tier

3. **Stateless Gemini**: No multi-turn conversation memory
   - **Solution**: Append previous response to context (see QUICK_REFERENCE.md)

4. **Cluster Stability**: K-means can produce different results on re-runs
   - **Solution**: Set `random_state=42` for reproducibility

---

## 🔮 Future Enhancements

### Planned Features

- [ ] **Export functionality**: Save filtered results to CSV/PDF
- [ ] **Advanced NLP**: Integrate spaCy for entity recognition
- [ ] **Multi-turn chat**: Add conversation memory to Gemini
- [ ] **Custom clustering**: Support hierarchical clustering, DBSCAN
- [ ] **Real-time processing**: Stream new emails for live analysis
- [ ] **User authentication**: Multi-user support with saved sessions
- [ ] **Mobile responsive**: Optimize UI for tablets/phones

### Research Extensions

- Evaluate clustering quality (silhouette score, Davies-Bouldin index)
- Compare TF-IDF vs. word embeddings (Word2Vec, BERT)
- Implement topic modeling (LDA, NMF)
- Add sentiment analysis for email tone detection
- Build entity relationship graphs from email networks

---

## 📖 References

### Academic Papers

1. Salton, G., & Buckley, C. (1988). *Term-weighting approaches in automatic text retrieval*. Information Processing & Management.
2. MacQueen, J. (1967). *Some methods for classification and analysis of multivariate observations*. Berkeley Symposium on Mathematical Statistics and Probability.
3. Klimt, B., & Yang, Y. (2004). *The Enron Corpus: A New Dataset for Email Classification Research*. ECML 2004.

### Documentation

- [Dash Documentation](https://dash.plotly.com/)
- [scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [NLTK Book](https://www.nltk.org/book/)
- [Gemini API Docs](https://ai.google.dev/docs)

### Dataset

- **Enron Email Dataset**: [CMU CALD](https://www.cs.cmu.edu/~enron/)
- **Kaggle Version**: [Enron Email Dataset](https://www.kaggle.com/datasets/wcukierski/enron-email-dataset)

---

## 👥 Authors

**Haseeb Ul Hassan (MSCS25003)**  
- Implementation Lead
- Dashboard Development
- Gemini AI Integration
- Data Processing Lead
- Preprocessing Module
- Clustering Implementation
- GitHub: [@HaseebUlHassan437](https://github.com/HaseebUlHassan437)

**Course:** Information Retrieval  
**Institution:** [ITU Lahore]  
**Semester:** MSCS 1st semester

---

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2024 Haseeb Ul Hassan 
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **Enron Dataset**: Carnegie Mellon University & Kaggle
- **Google AI**: Gemini API access
- **Open Source Community**: scikit-learn, NLTK, Plotly, Dash contributors
- **Course Instructor**: [DR. Ahmad Mustafa] for guidance and feedback

---

## 📧 Contact & Support

### Questions or Issues?

- **Email**: [haseebulhassan1172003@gmail.com]
- **GitHub Issues**: [Create an issue](https://github.com/HaseebUlHassan437/temporal_document_clustering/issues)
- **Discussion**: [Start a discussion](https://github.com/HaseebUlHassan437/temporal_document_clustering/discussions)

### Quick Links

- 📖 [Full Documentation](docs/)
- 🎥 [Demo Video](docs/demo_video.mp4) *(if available)*
- 📊 [Project Report](docs/project_report.pdf) *(if available)*
- 🔗 [Live Demo](https://your-demo-url.com) *(if deployed)*

---

<div align="center">

**⭐ Star this repository if you found it helpful!**

Made with ❤️ for Information Retrieval Course

</div>
