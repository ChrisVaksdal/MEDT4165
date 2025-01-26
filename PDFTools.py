#!/usr/bin/env python

import io
import os

import fitz

from markdown_pdf import MarkdownPdf, Section

from markdown_it import MarkdownIt

from helpers import getExerciseDir, getOutputDir, getFileContent, createSyntaxHighlightedText

class HTMLGenerator:
    def __init__(self):
        self.mdParser = MarkdownIt("commonmark").enable("table")
    
    def generateHtml(self, md: str):
        return self.mdParser.render(md)

htmlGenerator = HTMLGenerator()

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

    def __init__(self):
        self.out_file = io.BytesIO()
        self.writer = fitz.DocumentWriter(self.out_file)
        self.page = 0
    
    def generatePdf(self, html: str, rootDir: str, css: str, meta: dict|None = None):
        if meta is not None:
            self.meta += meta
        
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


def save2Html(md: str, exercise: int):
    html = htmlGenerator.generateHtml(md)
    directory = getOutputDir(exercise)
    prevDir = os.getcwd()

    os.makedirs(directory, exist_ok=True)
    os.chdir(directory)
    with open(f"Exercise{exercise}.html", "w") as file:
        file.write(html)

    os.chdir(prevDir)

def md2Html(md: str):
    return htmlGenerator.generateHtml(md)

def savePdf(html: str, exercise: int):
    pdf = PDFGenerator()
    imageDir = getOutputDir(exercise)
    css = getFileContent("style.css")
    pdf.generatePdf(html, imageDir, css)
    pdf.save(f"{getOutputDir(exercise)}/Exercise{exercise}.pdf")

def generateMarkdown(pdf: MarkdownPdf, exercise: int, includeSource: bool = False):
    pdf.meta["title"] = f"MEDT4165 - Exercise {exercise}"
    pdf.meta["author"] = "Christoffer-Robin Vaksdal"

    exerciseDir = getExerciseDir(exercise)
    imageDir = f"{exerciseDir}/output"
    outputDir = getOutputDir(exercise)

    md = getFileContent(f"{exerciseDir}/Exercise{exercise}.md")
    css = getFileContent("style.css")
    pdf.add_section(Section(md, root=imageDir), user_css=css)
    
    if includeSource:
        code = getFileContent(f"{exerciseDir}/main.py")
        code, css = createSyntaxHighlightedText(code)
        # code = f"```python\n{code}\n```"
        with open("code.html", "w") as html:
            html.write(code)
        
        pdf.add_section(Section(f"# Source code\n\n{code}"), user_css=css)

    pdf.save(f"{outputDir}/Exercise{exercise}.pdf")
