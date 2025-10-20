from pdf2docx import Converter
import tabula
import pandas as pd
from fpdf import FPDF
import os

def pdf_to_docx(pdf_path):
    docx_path = os.path.splitext(pdf_path)[0] + '.docx'
    cv = Converter(pdf_path)
    cv.convert(docx_path, start=0, end=None)
    cv.close()
    return docx_path

def pdf_to_excel(pdf_path):
    excel_path = os.path.splitext(pdf_path)[0] + '.xlsx'
    tabula.convert_into(pdf_path, excel_path, output_format="xlsx", pages='all')
    return excel_path

def excel_to_pdf(excel_path):
    df = pd.read_excel(excel_path)
    pdf_path = os.path.splitext(excel_path)[0] + ".pdf"

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=10)

    for col_name in df.columns:
        pdf.cell(40, 10, str(col_name), border=1)
    pdf.ln()

    for i, row in df.iterrows():
        for item in row:
            pdf.cell(40, 10, str(item), border=1)
        pdf.ln()
    pdf.output(pdf_path)
    return pdf_path
