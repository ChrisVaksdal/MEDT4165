#!/usr/bin/env python

import argparse

from helpers import getFileContent, runExercise, getOutputDir

from PDFTools import PDFGenerator
from HTMLTools import HTMLGenerator


def main(args=None):
    if args is not None:
        if args.run:
            for exercise in args.exercises:
                print(f"Running exercise {exercise}...")
                runExercise(exercise)

    for exercise in args.exercises:
        print(f"Generating HTML: 'Exercise {exercise}'")

        md = getFileContent(f"Exercise {exercise}/Exercise{exercise}.md")

        html = HTMLGenerator()
        htmlGenerator = HTMLGenerator()

        if args.include_src:
            print(f"Include source code in HTML: 'Exercise {exercise}'")
            codeMd = f"## Source code\n\n### exercises/Exercise {exercise}/main.py\n```python\n{getFileContent(f'Exercise {exercise}/main.py')}```\n"
            codeMd += "\n\n>See full source code project on [GitHub: Chris Vaksdal / MEDT4165](https://github.com/ChrisVaksdal/MEDT4165)"
            md = f"{md}\n\n{codeMd}"

        html = htmlGenerator.generateHtml(md)

        if args.export == "html":
            with open(f"{getOutputDir(exercise)}/Exercise{exercise}.html",
                      "w") as file:
                html.head += f"<title>MEDT4165 - Exercise {exercise}</title>"
                html.head += f"<style>{getFileContent('style.css')}</style>"
                file.write(str(html))

        elif args.export == "pdf":
            print(f"Generating PDF: 'Exercise {exercise}'")
            pdf = PDFGenerator(
                meta={
                    "title": f"MEDT4165 - Exercise {exercise}",
                    "author": "Christoffer-Robin Vaksdal"
                })
            imageDir = getOutputDir(exercise)
            css = getFileContent("style.css")
            pdf.generatePdf(str(html), imageDir, css)
            pdf.save(f"{getOutputDir(exercise)}/Exercise{exercise}.pdf")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateHandIn.py",
        description="Generates hand-in documents for exercises.",
    )

    parser.add_argument("--include-src",
                        action="store_true",
                        help="Include source code in PDF.")
    parser.add_argument("--run",
                        action="store_true",
                        help="Run exercise(s). Specify using --exercises.")
    parser.add_argument("--exercises",
                        nargs="+",
                        type=int,
                        help="Exercises to run.",
                        default=[])
    parser.add_argument("--export", type=str, help="", default="pdf")

    main(parser.parse_args())
