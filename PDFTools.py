#!/usr/bin/env python

import io
import fitz

class PDFGenerator:

    borders = (36, 36, -36, -36)

    meta = {
        "creationDate": fitz.get_pdf_now(),
        "modDate": fitz.get_pdf_now(),
        "creator": "PyMuPDF library: https://pypi.org/project/PyMuPDF",
        "producer": None,
        "title": None,
        "author": None,
        "subject": None,
        "keywords": None,
    }

    def __init__(self, meta: dict|None = None):
        self.out_file = io.BytesIO()
        self.writer = fitz.DocumentWriter(self.out_file)
        self.page = 0
        self.meta = meta
    
    def generatePdf(self, html: str, rootDir: str, css: str):
        story = fitz.Story(html=html, archive=rootDir, user_css=css)
        rect = fitz.paper_rect("A4")
        where = rect + self.borders

        more = 1
        while more:  # loop outputting the story
            self.page += 1
            device = self.writer.begin_page(rect)
            more, _ = story.place(where)  # layout into allowed rectangle
            # story.element_positions(self._recorder, {"toc": True, "pdfile": self})
            story.draw(device)
            self.writer.end_page()
    
    def save(self, outputPath: str):
        self.writer.close()
        doc = fitz.open("pdf", self.out_file)
        doc.set_metadata(self.meta)
        doc.save(outputPath)
        doc.close()
