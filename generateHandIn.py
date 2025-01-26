import argparse

from helpers import createSyntaxHighlightedText, getFileContent, runExercise
from PDFTools import md2Html, savePdf

def main(args=None):
    if args is not None:
        if args.run:
            for exercise in args.exercises:
                print(f"Running exercise {exercise}...")
                runExercise(exercise)
    
    for exercise in args.exercises:
        print(f"Generating HTML: 'Exercise {exercise}'")
        html = md2Html(getFileContent(f"Exercise {exercise}/Exercise{exercise}.md"))
        if args.include_src:
            codeHtml, codeCss = createSyntaxHighlightedText(getFileContent(f"Exercise {exercise}/main.py"))
            html += f"<style>{codeCss}</style>{codeHtml}"
        savePdf(html, exercise)
    
    print("Done.")
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="generateHandIn.py",
        description="Generates hand-in documents for exercises.",
    )

    parser.add_argument("--include-src", action="store_true", help="Include source code in PDF.")
    parser.add_argument("--run", action="store_true", help="Run exercise(s). Specify using --exercises.")
    parser.add_argument("--exercises", nargs="+", type=int, help="Exercises to run.", default=[])
    parser.add_argument("--export", type=str, help="", default="pdf")

    main(parser.parse_args())
