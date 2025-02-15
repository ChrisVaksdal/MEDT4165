#!/usr/bin/env python

from markdown_it import MarkdownIt

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter


class HTMLSimple:

    def __init__(self, body: str = "", head: str = ""):
        self.body = body
        self.head = head

    def __str__(self):
        return f"<html><head>{self.head}</head><body>{self.body}</body></html>"

    def __repr__(self):
        return str(self)

    def __add__(self, other):
        return HTMLSimple(self.body + other.body, self.head + other.head)


class HTMLGenerator:

    def __init__(self):
        self.mdParser = MarkdownIt("commonmark").enable("table")
        self.lexer = PythonLexer(stripall=True)
        self.formatter = HtmlFormatter(linenos=False,
                                       cssclass="syntax_highlighted",
                                       style="sas",
                                       wrapcode=True)
        self.mdParser.options[
            "highlight"] = lambda code, _lang, _langattrs: highlight(
                code, self.lexer, self.formatter)

    def generateHtml(self, md: str) -> HTMLSimple:
        return HTMLSimple(
            body=self.mdParser.render(md),
            head=f"<style>{self.formatter.get_style_defs()}</style>")
