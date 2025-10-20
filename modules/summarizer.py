import nltk
nltk.download('punkt')  # Ensure NLTK tokenizer data is available

from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer
import fitz  # PyMuPDF for PDF handling
from docx import Document
import pandas as pd

def summarize_text(text, sentences_count=5):
    """
    Generate a summary of the given text using LexRank algorithm.
    """
    parser = PlaintextParser.from_string(text, Tokenizer("english"))
    summarizer = LexRankSummarizer()
    summary = summarizer(parser.document, sentences_count)
    return ' '.join(str(sentence) for sentence in summary)

def extract_text_from_pdf(file_path):
    """
    Extract and return text from a PDF file.
    """
    doc = fitz.open(file_path)
    full_text = ''
    for page in doc:
        full_text += page.get_text()
    doc.close()
    return full_text

def extract_text_from_docx(file_path):
    """
    Extract and return text from a DOCX file.
    """
    doc = Document(file_path)
    return '\n'.join(paragraph.text for paragraph in doc.paragraphs)

def extract_text_from_excel(file_path):
    """
    Extract and return all data from an Excel file as text for summarization.
    """
    df = pd.read_excel(file_path)
    return '\n'.join(df.astype(str).agg(' '.join, axis=1))
