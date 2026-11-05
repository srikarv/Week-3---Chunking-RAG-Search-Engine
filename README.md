# PDF Research Desk

A local RAG search engine for the six reports in `PDF Files`. It extracts text from the PDFs, splits it into overlapping word chunks, ranks passages with local TF-IDF retrieval, and optionally asks an OpenAI-compatible chat model to compose a cited answer.

## Setup

From this folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Optional model setup:

```powershell
Copy-Item .env.example .env
# Put your OPENAI_API_KEY in .env
```

Without an API key, the app still works in retrieval-only mode and displays the best evidence passages with source and page citations.

## Run

Run the indexing smoke test:

```powershell
python app.py --smoke-test
```

Start the web app:

```powershell
streamlit run app.py
```

Then open the local URL shown by Streamlit. Use the sidebar to adjust chunk size, overlap, and the number of retrieved passages. The index is cached for the selected chunk settings.

## RAG flow

1. `PdfReader` extracts page text and preserves page numbers.
2. Pages are cleaned and split into overlapping chunks.
3. A local TF-IDF matrix retrieves the most relevant chunks for a question.
4. Retrieved chunks are shown as evidence and, when configured, passed to the model in a prompt that requires source citations and prohibits unsupported claims.

The app expects PDFs in the adjacent `PDF Files` folder. Set `PDF_DIR` in `.env` to use another folder.

## Chunk-size recommendation

The default is **900 words with 150 words of overlap**. The supplied reports range from short survey reports to substantially longer research and enterprise reports. A 900-word window is large enough to keep a finding, its qualification, and nearby context together, while remaining small enough for TF-IDF retrieval to distinguish topics within a report. The 150-word overlap reduces boundary losses when a finding spans two windows.

Chunk size is measured in words because the local index is built from extracted PDF text. For this corpus, start with 900/150, then compare 600-800 words for precise fact lookup or 1,000-1,200 words for broad synthesis. The sidebar lets you test those settings without changing code.

## Corpus availability

The index skips PDF placeholders that are visible in OneDrive but not downloaded locally and reports their names in the smoke test and UI. Download or sync those files before indexing if they are needed for answers. PDFs containing scanned images without an extractable text layer require OCR before they can be retrieved.
