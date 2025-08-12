# Financial-Document-Reader
A Streamlit-powered web application for extracting financial entities from multiple document formats — including PDF, DOCX, TXT, and scanned images — using rule-based extraction, NER (Named Entity Recognition), and RAG (Retrieval-Augmented Generation) with OpenAI or local models.


 Features
Multi-format Support

PDF (selectable text & OCR fallback)

DOCX

TXT

Images (PNG, JPG, JPEG)

Entity Extraction

Predefined financial entities (e.g., Counterparty, ISIN, Notional, Maturity, Coupon, etc.)

Rule-based regex patterns

NER with dslim/bert-base-NER

RAG-based LLM extraction (OpenAI or local T5 model)

Optional AI Integrations

OpenAI GPT models (OPENAI_API_KEY required)

Local summarizer (google/flan-t5-base)

FAISS vector search for document retrieval

OCR Support

Uses pytesseract and pdfplumber for extracting text from scanned PDFs & images

📦 Installation
1️⃣ Clone the Repository
bash
Copy
Edit
git clone https://github.com/yourusername/financial-document-reader.git
cd financial-document-reader
2️⃣ Create a Virtual Environment
bash
Copy
Edit
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
3️⃣ Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
Example requirements.txt:

nginx
Copy
Edit
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



HuggingFace Transformers (dslim/bert-base-NER, google/flan-t5-base)

<img width="1920" height="1080" alt="Screenshot (36)" src="https://github.com/user-attachments/assets/8036afcf-7ce7-4342-ba5b-4fd6f85b4fb0" />
<img width="1920" height="1080" alt="Screenshot (34)" src="https://github.com/user-attachments/assets/90f28030-6d21-4133-95fe-12ef30dee6dc" />
<img width="1920" height="1080" alt="Screenshot (33)" src="https://github.com/user-attachments/assets/51f1ad53-5a4d-4f6e-b2c1-bb21bb04e917" />
<img width="1920" height="1080" alt="Screenshot (39)" src="https://github.com/user-attachments/assets/383de041-d5f6-4b36-ac71-fa800f304b45" />
<img width="1920" height="1080" alt="Screenshot (37)" src="https://github.com/user-attachments/assets/827ce5c2-3ec9-427a-84eb-292f829659ca" />

