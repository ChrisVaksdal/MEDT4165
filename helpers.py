#!/usr/bin/env python

import os

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

def getExerciseDir(exercise: int):
    directory = f"{os.path.dirname(os.path.abspath(__file__))}/Exercise {exercise}"
    if not os.path.exists(directory):
        raise Exception(f"Exercise {exercise} does not exist.")
    return directory

def getOutputDir(exercise: int):
    directory = f"{os.path.dirname(os.path.abspath(__file__))}/output/{exercise}"
    if not os.path.exists(directory):
        os.makedirs(directory)
    return directory

def getFileContent(path: str):
    with open(path) as f:
        return f.read()

def runExercise(exercise: int):
    path = f"{getExerciseDir(exercise)}/main.py"
    with open(path) as f:
        print(f"Running exercise {exercise}...")
        os.chdir(f"./Exercise {exercise}")
        exec(open(path).read())
        os.chdir("..")
    

def createSyntaxHighlightedText(code: str):
    lexer = PythonLexer(stripall=True)
    formatter = HtmlFormatter(linenos=False, cssclass="syntax_highlighted", style="sas", wrapcode=True)
    return highlight(code, lexer, formatter), formatter.get_style_defs()
    # + f"<style>{formatter.get_style_defs()}</style"
