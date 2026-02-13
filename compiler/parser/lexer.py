#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lexer for the UnitPort DSL (restricted Python subset).

Produces tokens including synthetic INDENT/DEDENT tokens
based on indentation tracking with a stack.
"""

from __future__ import annotations
from enum import Enum, auto
from dataclasses import dataclass
from typing import List


class TokenType(Enum):
    # Literals
    INTEGER = auto()
    FLOAT = auto()
    STRING = auto()
    TRUE = auto()
    FALSE = auto()
    NONE = auto()

    # Identifiers and keywords
    IDENTIFIER = auto()
    IF = auto()
    ELIF = auto()
    ELSE = auto()
    WHILE = auto()
    FOR = auto()
    IN = auto()
    DEF = auto()
    RETURN = auto()
    PASS = auto()
    BREAK = auto()
    CONTINUE = auto()
    IMPORT = auto()
    FROM = auto()
    AND = auto()
    OR = auto()
    NOT = auto()

    # Operators
    PLUS = auto()        # +
    MINUS = auto()       # -
    STAR = auto()        # *
    SLASH = auto()       # /
    DOUBLE_STAR = auto() # **
    PERCENT = auto()     # %
    DOUBLE_SLASH = auto()  # //

    # Comparison
    EQ = auto()          # ==
    NEQ = auto()         # !=
    LT = auto()          # <
    GT = auto()          # >
    LTE = auto()         # <=
    GTE = auto()         # >=

    # Assignment
    ASSIGN = auto()      # =
    PLUS_ASSIGN = auto() # +=
    MINUS_ASSIGN = auto()  # -=
    STAR_ASSIGN = auto()   # *=
    SLASH_ASSIGN = auto()  # /=

    # Delimiters
    LPAREN = auto()      # (
    RPAREN = auto()      # )
    LBRACKET = auto()    # [
    RBRACKET = auto()    # ]
    COMMA = auto()       # ,
    COLON = auto()       # :
    DOT = auto()         # .
    HASH = auto()        # #

    # Indentation
    INDENT = auto()
    DEDENT = auto()
    NEWLINE = auto()

    # Special
    COMMENT = auto()
    EOF = auto()


_KEYWORDS = {
    "if": TokenType.IF,
    "elif": TokenType.ELIF,
    "else": TokenType.ELSE,
    "while": TokenType.WHILE,
    "for": TokenType.FOR,
    "in": TokenType.IN,
    "def": TokenType.DEF,
    "return": TokenType.RETURN,
    "pass": TokenType.PASS,
    "break": TokenType.BREAK,
    "continue": TokenType.CONTINUE,
    "import": TokenType.IMPORT,
    "from": TokenType.FROM,
    "True": TokenType.TRUE,
    "False": TokenType.FALSE,
    "None": TokenType.NONE,
    "and": TokenType.AND,
    "or": TokenType.OR,
    "not": TokenType.NOT,
}


@dataclass
class Token:
    """A lexer token."""
    type: TokenType
    value: str
    line: int
    col: int

    def __repr__(self):
        return f"Token({self.type.name}, {self.value!r}, L{self.line}:{self.col})"


class LexerError(Exception):
    """Lexer error with position info."""
    def __init__(self, message: str, line: int, col: int):
        super().__init__(f"Line {line}:{col}: {message}")
        self.line = line
        self.col = col


class Lexer:
    """
    Tokenize a UnitPort DSL source string.

    Produces INDENT/DEDENT tokens using a stack-based indentation tracker.
    Only spaces are allowed for indentation (tabs are rejected).
    """

    def __init__(self, source: str):
        self._source = source
        self._pos = 0
        self._line = 1
        self._col = 1
        self._tokens: List[Token] = []
        self._indent_stack: List[int] = [0]
        self._at_line_start = True

    def tokenize(self) -> List[Token]:
        """Tokenize the entire source and return token list."""
        self._tokens = []
        self._pos = 0
        self._line = 1
        self._col = 1
        self._indent_stack = [0]
        self._at_line_start = True

        lines = self._source.split("\n")

        for line_idx, line_text in enumerate(lines):
            self._line = line_idx + 1

            # Check for tabs anywhere in leading whitespace
            leading_ws = len(line_text) - len(line_text.lstrip())
            if "\t" in line_text[:leading_ws]:
                raise LexerError("Tabs are not allowed for indentation; use spaces",
                                 self._line, 1)

            # Skip completely empty lines
            stripped = line_text.lstrip(" ")
            if not stripped or stripped.startswith("#"):
                if stripped.startswith("#"):
                    indent = len(line_text) - len(stripped)
                    self._tokens.append(Token(
                        TokenType.COMMENT, stripped[1:].strip(),
                        self._line, indent + 1,
                    ))
                self._tokens.append(Token(TokenType.NEWLINE, "\\n", self._line, 1))
                continue

            # Count leading spaces
            indent_level = len(line_text) - len(stripped)

            # Legacy tab check (kept for safety)
            if "\t" in line_text[:indent_level]:
                raise LexerError("Tabs are not allowed for indentation; use spaces",
                                 self._line, 1)

            # Emit INDENT / DEDENT tokens
            current_indent = self._indent_stack[-1]
            if indent_level > current_indent:
                self._indent_stack.append(indent_level)
                self._tokens.append(Token(TokenType.INDENT, "", self._line, 1))
            elif indent_level < current_indent:
                while self._indent_stack[-1] > indent_level:
                    self._indent_stack.pop()
                    self._tokens.append(Token(TokenType.DEDENT, "", self._line, 1))

            # Tokenize the content of this line
            self._pos = indent_level
            self._col = indent_level + 1
            self._tokenize_line(line_text)
            self._tokens.append(Token(TokenType.NEWLINE, "\\n", self._line, len(line_text) + 1))

        # Emit remaining DEDENTs
        while len(self._indent_stack) > 1:
            self._indent_stack.pop()
            self._tokens.append(Token(TokenType.DEDENT, "", self._line, 1))

        self._tokens.append(Token(TokenType.EOF, "", self._line, 1))
        return self._tokens

    def _tokenize_line(self, line: str):
        """Tokenize a single line starting from self._pos."""
        length = len(line)

        while self._pos < length:
            ch = line[self._pos]

            # Skip spaces
            if ch == " ":
                self._pos += 1
                self._col += 1
                continue

            # Comment
            if ch == "#":
                comment_text = line[self._pos + 1:].strip()
                self._tokens.append(Token(TokenType.COMMENT, comment_text,
                                          self._line, self._col))
                return  # Rest of line is comment

            # String literals
            if ch in ('"', "'"):
                self._read_string(line, ch)
                continue

            # Numbers
            if ch.isdigit() or (ch == "." and self._pos + 1 < length and line[self._pos + 1].isdigit()):
                self._read_number(line)
                continue

            # Identifiers / keywords
            if ch.isalpha() or ch == "_":
                self._read_identifier(line)
                continue

            # Two-character operators
            if self._pos + 1 < length:
                two_char = line[self._pos:self._pos + 2]
                token_type = self._match_two_char(two_char)
                if token_type:
                    self._tokens.append(Token(token_type, two_char, self._line, self._col))
                    self._pos += 2
                    self._col += 2
                    continue

            # Single-character operators/delimiters
            token_type = self._match_single_char(ch)
            if token_type:
                self._tokens.append(Token(token_type, ch, self._line, self._col))
                self._pos += 1
                self._col += 1
                continue

            # Unknown character - skip with warning
            self._pos += 1
            self._col += 1

    def _read_string(self, line: str, quote: str):
        """Read a string literal."""
        start_col = self._col
        self._pos += 1  # skip opening quote
        self._col += 1

        # Check for triple quotes
        triple = False
        if self._pos + 1 < len(line) and line[self._pos:self._pos + 2] == quote * 2:
            triple = True
            self._pos += 2
            self._col += 2

        result = []
        if triple:
            # For triple quotes, just read until closing triple quote on same line
            end_marker = quote * 3
            idx = line.find(end_marker, self._pos)
            if idx >= 0:
                result.append(line[self._pos:idx])
                self._pos = idx + 3
                self._col = self._pos + 1
            else:
                # Take rest of line as the string
                result.append(line[self._pos:])
                self._pos = len(line)
                self._col = self._pos + 1
        else:
            while self._pos < len(line):
                ch = line[self._pos]
                if ch == "\\":
                    # Escape sequence
                    self._pos += 1
                    self._col += 1
                    if self._pos < len(line):
                        escape_ch = line[self._pos]
                        escape_map = {"n": "\n", "t": "\t", "\\": "\\",
                                      "'": "'", '"': '"'}
                        result.append(escape_map.get(escape_ch, escape_ch))
                elif ch == quote:
                    self._pos += 1
                    self._col += 1
                    break
                else:
                    result.append(ch)
                self._pos += 1
                self._col += 1

        self._tokens.append(Token(TokenType.STRING, "".join(result),
                                  self._line, start_col))

    def _read_number(self, line: str):
        """Read a numeric literal (int or float)."""
        start = self._pos
        start_col = self._col
        has_dot = False

        while self._pos < len(line):
            ch = line[self._pos]
            if ch.isdigit():
                self._pos += 1
                self._col += 1
            elif ch == "." and not has_dot:
                has_dot = True
                self._pos += 1
                self._col += 1
            else:
                break

        text = line[start:self._pos]
        if has_dot:
            self._tokens.append(Token(TokenType.FLOAT, text, self._line, start_col))
        else:
            self._tokens.append(Token(TokenType.INTEGER, text, self._line, start_col))

    def _read_identifier(self, line: str):
        """Read an identifier or keyword."""
        start = self._pos
        start_col = self._col

        while self._pos < len(line):
            ch = line[self._pos]
            if ch.isalnum() or ch == "_":
                self._pos += 1
                self._col += 1
            else:
                break

        text = line[start:self._pos]
        token_type = _KEYWORDS.get(text, TokenType.IDENTIFIER)
        self._tokens.append(Token(token_type, text, self._line, start_col))

    @staticmethod
    def _match_two_char(s: str):
        """Match two-character operators."""
        mapping = {
            "==": TokenType.EQ,
            "!=": TokenType.NEQ,
            "<=": TokenType.LTE,
            ">=": TokenType.GTE,
            "**": TokenType.DOUBLE_STAR,
            "//": TokenType.DOUBLE_SLASH,
            "+=": TokenType.PLUS_ASSIGN,
            "-=": TokenType.MINUS_ASSIGN,
            "*=": TokenType.STAR_ASSIGN,
            "/=": TokenType.SLASH_ASSIGN,
        }
        return mapping.get(s)

    @staticmethod
    def _match_single_char(ch: str):
        """Match single-character tokens."""
        mapping = {
            "+": TokenType.PLUS,
            "-": TokenType.MINUS,
            "*": TokenType.STAR,
            "/": TokenType.SLASH,
            "%": TokenType.PERCENT,
            "<": TokenType.LT,
            ">": TokenType.GT,
            "=": TokenType.ASSIGN,
            "(": TokenType.LPAREN,
            ")": TokenType.RPAREN,
            "[": TokenType.LBRACKET,
            "]": TokenType.RBRACKET,
            ",": TokenType.COMMA,
            ":": TokenType.COLON,
            ".": TokenType.DOT,
        }
        return mapping.get(ch)
