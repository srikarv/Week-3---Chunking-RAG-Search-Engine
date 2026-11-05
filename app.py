from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from dotenv import load_dotenv
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = Path(os.getenv("PDF_DIR", BASE_DIR / "PDF Files"))


@dataclass(frozen=True)
class Chunk:
    source: str
    page: int
    text: str


class LocalRAG:
    """PDF chunking and local lexical retrieval for the supplied report set."""

    def __init__(self, pdf_dir: Path = PDF_DIR, chunk_size: int = 900, overlap: int = 150):
        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        self.pdf_dir = Path(pdf_dir)
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.chunks: list[Chunk] = []
        self.skipped_files: list[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None

    @staticmethod
    def _clean(text: str) -> str:
        return re.sub(r"\s+", " ", text).strip()

    def _chunk_text(self, text: str) -> Iterable[str]:
        words = self._clean(text).split()
        step = self.chunk_size - self.overlap
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + self.chunk_size])
            if len(chunk) >= 80:
                yield chunk
            if start + self.chunk_size >= len(words):
                break

    def build(self) -> int:
        self.chunks = []
        self.skipped_files = []
        pdfs = sorted(self.pdf_dir.glob("*.pdf"))
        if not pdfs:
            raise FileNotFoundError(f"No PDF files found in {self.pdf_dir}")
        for pdf_path in pdfs:
            if not pdf_path.is_file():
                self.skipped_files.append(pdf_path.name)
                continue
            reader = PdfReader(str(pdf_path))
            for page_number, page in enumerate(reader.pages, start=1):
                text = page.extract_text() or ""
                for chunk in self._chunk_text(text):
                    self.chunks.append(Chunk(pdf_path.name, page_number, chunk))
        if not self.chunks:
            raise ValueError("The PDFs did not yield extractable text")
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.98)
        self.matrix = self.vectorizer.fit_transform(chunk.text for chunk in self.chunks)
        return len(self.chunks)

    def search(self, query: str, top_k: int = 5) -> list[tuple[Chunk, float]]:
        if self.vectorizer is None or self.matrix is None:
            raise RuntimeError("Call build() before search()")
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = np.argsort(scores)[::-1]
        return [(self.chunks[index], float(scores[index])) for index in ranked[:top_k] if scores[index] > 0]


def make_prompt(question: str, results: list[tuple[Chunk, float]]) -> str:
    context = "\n\n".join(
        f"[{index}] {chunk.source}, page {chunk.page}\n{chunk.text}"
        for index, (chunk, _) in enumerate(results, start=1)
    )
    return (
        "Answer the question using only the provided sources. If the sources do not contain "
        "the answer, say that clearly. Cite supporting sources inline as [1], [2], etc. "
        "Be concise and distinguish reported findings from your own synthesis.\n\n"
        f"Question: {question}\n\nSources:\n{context}"
    )


def generate_answer(question: str, results: list[tuple[Chunk, float]]) -> str:
    """Use an OpenAI-compatible endpoint when configured; otherwise provide retrieval output."""
    if not results:
        return "I could not find relevant passages in the indexed PDFs."
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=os.getenv("OPENAI_BASE_URL") or None)
        response = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=0.1,
            messages=[
                {"role": "system", "content": "You are a careful research assistant."},
                {"role": "user", "content": make_prompt(question, results)},
            ],
        )
        return response.choices[0].message.content or "The model returned no answer."

    excerpts = []
    for index, (chunk, score) in enumerate(results, start=1):
        excerpts.append(f"[{index}] {chunk.source}, page {chunk.page} (match {score:.2f})\n{chunk.text}")
    return (
        "No model API key is configured, so here are the most relevant passages to answer from:\n\n"
        + "\n\n".join(excerpts)
    )


def run_smoke_test() -> None:
    rag = LocalRAG()
    count = rag.build()
    results = rag.search("How does AI affect software developer productivity?", top_k=3)
    print(f"Indexed {count} chunks from {len(set(chunk.source for chunk in rag.chunks))} PDFs")
    if rag.skipped_files:
        print(f"Skipped unavailable PDFs: {', '.join(rag.skipped_files)}")
    print(f"Retrieved {len(results)} passages")
    for chunk, score in results:
        print(f"- {chunk.source} p.{chunk.page}: {score:.3f}")


def run_app() -> None:
    import streamlit as st

    st.set_page_config(page_title="PDF Research Desk", page_icon="🔎", layout="wide")
    st.title("PDF Research Desk")
    st.caption("Ask grounded questions across the six provided AI and software-engineering reports.")

    @st.cache_resource(show_spinner="Reading and indexing PDFs...")
    def load_index(chunk_size: int, overlap: int) -> LocalRAG:
        rag = LocalRAG(chunk_size=chunk_size, overlap=overlap)
        rag.build()
        return rag

    with st.sidebar:
        st.header("Index")
        chunk_size = st.slider("Chunk size (words)", 300, 1500, 900, 50)
        overlap = st.slider("Chunk overlap (words)", 0, 400, 150, 25)
        top_k = st.slider("Retrieved passages", 2, 10, 5)
        st.divider()
        st.write(f"PDF folder: `{PDF_DIR}`")
        if os.getenv("OPENAI_API_KEY"):
            st.success(f"Answer model: `{os.getenv('OPENAI_MODEL', 'gpt-4o-mini')}`")
        else:
            st.info("No API key detected. Retrieval-only mode is active.")

    if overlap >= chunk_size:
        st.error("Chunk overlap must be smaller than chunk size.")
        return
    try:
        rag = load_index(chunk_size, overlap)
    except Exception as error:
        st.error(str(error))
        return

    st.write(f"Indexed **{len(rag.chunks):,}** passages from **{len(set(c.source for c in rag.chunks))}** PDFs.")
    if rag.skipped_files:
        st.warning("Unavailable PDFs were skipped: " + ", ".join(rag.skipped_files))
    question = st.text_area("Question", placeholder="Compare what the reports say about AI and developer productivity.", height=90)
    ask = st.button("Search and answer", type="primary", disabled=not question.strip())
    if ask:
        results = rag.search(question, top_k=top_k)
        st.subheader("Answer")
        st.write(generate_answer(question, results))
        with st.expander("Retrieved evidence", expanded=True):
            if not results:
                st.warning("No matching passages were found.")
            for index, (chunk, score) in enumerate(results, start=1):
                st.markdown(f"**[{index}] {chunk.source}, page {chunk.page}** · match {score:.2f}")
                st.write(chunk.text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-test", action="store_true", help="Index the PDFs and run one retrieval query")
    args = parser.parse_args()
    if args.smoke_test:
        run_smoke_test()
    else:
        run_app()
