#!/usr/bin/env python

import argparse
from markdown_pdf import MarkdownPdf, Section

from helpers import getExerciseDir, getOutputDir, getFileContent, runExercise

def generateMarkdown(pdf: MarkdownPdf, exercise: int, includeSource: bool = False):
    pdf.meta["title"] = f"MEDT4165 - Exercise {exercise}"
    pdf.meta["author"] = "Christoffer-Robin Vaksdal"

    exerciseDir = getExerciseDir(exercise)
    outputDir = getOutputDir(exercise)

    md = getFileContent(f"{exerciseDir}/Exercise{exercise}.md")
    pdf.add_section(Section(md, root=outputDir), user_css=getFileContent("style.css"))
    
    if includeSource:
        code = getFileContent(f"{exerciseDir}/main.py")
        code = f"```py\n{code}\n```"
        pdf.add_section(Section(f"# Source code\n\n{code}"))

    pdf.save(f"{outputDir}/Exercise{exercise}.pdf")


def main(args=None):
    if args is not None:
        if args.run:
            for exercise in args.exercises:
                runExercise(exercise)
    
    for exercise in args.exercises:
        generateMarkdown(MarkdownPdf(), exercise, args.include_src)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog='generatePDF.py',
        description='Converts Markdown exercises to PDF.',
        epilog='Text at the bottom of help'
    )

    parser.add_argument('--include-src', action='store_true', help='Include source code in PDF.')
    parser.add_argument('--run', action='store_true', help='Run exercise(s). Specify using --exercises.')
    parser.add_argument('--exercises', nargs='+', type=int, help='Exercises to run.', default=[])

    main(parser.parse_args())
