#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unit tests for the DSL lexer."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from compiler.parser.lexer import Lexer, Token, TokenType, LexerError


class TestLexerBasic(unittest.TestCase):
    def test_empty_source(self):
        tokens = Lexer("").tokenize()
        self.assertEqual(tokens[-1].type, TokenType.EOF)

    def test_integer(self):
        tokens = Lexer("42").tokenize()
        ints = [t for t in tokens if t.type == TokenType.INTEGER]
        self.assertEqual(len(ints), 1)
        self.assertEqual(ints[0].value, "42")

    def test_float(self):
        tokens = Lexer("3.14").tokenize()
        floats = [t for t in tokens if t.type == TokenType.FLOAT]
        self.assertEqual(len(floats), 1)
        self.assertEqual(floats[0].value, "3.14")

    def test_string_single_quotes(self):
        tokens = Lexer("'hello'").tokenize()
        strings = [t for t in tokens if t.type == TokenType.STRING]
        self.assertEqual(len(strings), 1)
        self.assertEqual(strings[0].value, "hello")

    def test_string_double_quotes(self):
        tokens = Lexer('"world"').tokenize()
        strings = [t for t in tokens if t.type == TokenType.STRING]
        self.assertEqual(len(strings), 1)
        self.assertEqual(strings[0].value, "world")

    def test_keywords(self):
        source = "if elif else while for in def return pass break continue import from True False None and or not"
        tokens = Lexer(source).tokenize()
        expected = [
            TokenType.IF, TokenType.ELIF, TokenType.ELSE,
            TokenType.WHILE, TokenType.FOR, TokenType.IN,
            TokenType.DEF, TokenType.RETURN, TokenType.PASS,
            TokenType.BREAK, TokenType.CONTINUE,
            TokenType.IMPORT, TokenType.FROM,
            TokenType.TRUE, TokenType.FALSE, TokenType.NONE,
            TokenType.AND, TokenType.OR, TokenType.NOT,
        ]
        keyword_tokens = [t for t in tokens if t.type not in
                          (TokenType.NEWLINE, TokenType.EOF)]
        self.assertEqual([t.type for t in keyword_tokens], expected)

    def test_identifier(self):
        tokens = Lexer("my_var").tokenize()
        ids = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        self.assertEqual(len(ids), 1)
        self.assertEqual(ids[0].value, "my_var")

    def test_operators(self):
        source = "+ - * / ** % // == != < > <= >= = += -= *= /="
        tokens = Lexer(source).tokenize()
        op_tokens = [t for t in tokens if t.type not in
                     (TokenType.NEWLINE, TokenType.EOF)]
        expected = [
            TokenType.PLUS, TokenType.MINUS, TokenType.STAR, TokenType.SLASH,
            TokenType.DOUBLE_STAR, TokenType.PERCENT, TokenType.DOUBLE_SLASH,
            TokenType.EQ, TokenType.NEQ, TokenType.LT, TokenType.GT,
            TokenType.LTE, TokenType.GTE, TokenType.ASSIGN,
            TokenType.PLUS_ASSIGN, TokenType.MINUS_ASSIGN,
            TokenType.STAR_ASSIGN, TokenType.SLASH_ASSIGN,
        ]
        self.assertEqual([t.type for t in op_tokens], expected)

    def test_delimiters(self):
        source = "( ) [ ] , : ."
        tokens = Lexer(source).tokenize()
        delim_tokens = [t for t in tokens if t.type not in
                        (TokenType.NEWLINE, TokenType.EOF)]
        expected = [
            TokenType.LPAREN, TokenType.RPAREN,
            TokenType.LBRACKET, TokenType.RBRACKET,
            TokenType.COMMA, TokenType.COLON, TokenType.DOT,
        ]
        self.assertEqual([t.type for t in delim_tokens], expected)


class TestLexerIndentation(unittest.TestCase):
    def test_simple_indent_dedent(self):
        source = "if True:\n    pass"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        self.assertIn(TokenType.INDENT, types)
        self.assertIn(TokenType.DEDENT, types)

    def test_nested_indent(self):
        source = "if True:\n    if False:\n        pass"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        indent_count = types.count(TokenType.INDENT)
        dedent_count = types.count(TokenType.DEDENT)
        self.assertEqual(indent_count, 2)
        self.assertEqual(dedent_count, 2)

    def test_multiple_dedent(self):
        source = "if True:\n    if False:\n        pass\nx = 1"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens]
        indent_count = types.count(TokenType.INDENT)
        dedent_count = types.count(TokenType.DEDENT)
        self.assertEqual(indent_count, dedent_count)

    def test_tabs_rejected(self):
        source = "if True:\n\tpass"
        with self.assertRaises(LexerError):
            Lexer(source).tokenize()


class TestLexerComments(unittest.TestCase):
    def test_comment_line(self):
        source = "# This is a comment\nx = 1"
        tokens = Lexer(source).tokenize()
        comments = [t for t in tokens if t.type == TokenType.COMMENT]
        self.assertEqual(len(comments), 1)
        self.assertEqual(comments[0].value, "This is a comment")

    def test_inline_comment(self):
        source = "x = 1  # inline comment"
        tokens = Lexer(source).tokenize()
        comments = [t for t in tokens if t.type == TokenType.COMMENT]
        self.assertEqual(len(comments), 1)


class TestLexerComplexSource(unittest.TestCase):
    def test_function_call(self):
        source = "RobotContext.run_action('stand')"
        tokens = Lexer(source).tokenize()
        ids = [t for t in tokens if t.type == TokenType.IDENTIFIER]
        self.assertIn("RobotContext", [t.value for t in ids])
        self.assertIn("run_action", [t.value for t in ids])
        strings = [t for t in tokens if t.type == TokenType.STRING]
        self.assertEqual(strings[0].value, "stand")

    def test_for_range(self):
        source = "for i in range(0, 5, 1):"
        tokens = Lexer(source).tokenize()
        types = [t.type for t in tokens if t.type not in
                 (TokenType.NEWLINE, TokenType.EOF)]
        self.assertEqual(types[0], TokenType.FOR)
        self.assertEqual(types[1], TokenType.IDENTIFIER)  # i
        self.assertEqual(types[2], TokenType.IN)
        self.assertEqual(types[3], TokenType.IDENTIFIER)  # range

    def test_multiline_workflow(self):
        source = """RobotContext.run_action('stand')
time.sleep(2.0)
RobotContext.run_action('walk')"""
        tokens = Lexer(source).tokenize()
        strings = [t for t in tokens if t.type == TokenType.STRING]
        self.assertEqual(len(strings), 2)
        self.assertEqual(strings[0].value, "stand")
        self.assertEqual(strings[1].value, "walk")


if __name__ == "__main__":
    unittest.main()
