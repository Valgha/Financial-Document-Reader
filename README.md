# Financial-Document-Reader
A Streamlit-powered web application for extracting financial entities from multiple document formats — including PDF, DOCX, TXT, and scanned images — using rule-based extraction, NER (Named Entity Recognition), and RAG (Retrieval-Augmented Generation) with OpenAI or local models.

Installation

Clone Repository
git clone https://github.com/yourusername/financial-document-reader.git
cd financial-document-reader

 Create a Virtual Environment
 python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows

Install Dependencies
pip install -r requirements.txt

requirements.txt
streamlit
pdfplumber
python-docx
Pillow
numpy
python-dotenv
pytesseract
opencv-python
sentence-transformers
faiss-cpu
transformers

Create a .env file in the project root:
OPENAI_API_KEY=your_openai_api_key_here

Usage
Run the Streamlit app:

streamlit run app.py


Technology Stack
Frontend & UI: Streamlit

Text Extraction:

pdfplumber (PDF text)

python-docx (DOCX parsing)

pytesseract + opencv (OCR)

AI / NLP:

HuggingFace Transformers (dslim/bert-base-NER, google/flan-t5-base)

Sentence Transformers (all-MiniLM-L6-v2)

FAISS (vector search)

OpenAI GPT API (optional)

Utilities:

Regex rule-based extraction

JSON output cleaning

.env config handling


