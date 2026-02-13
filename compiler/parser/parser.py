#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Recursive descent parser for the UnitPort DSL (restricted Python subset).

Parses tokens from the Lexer into an AST.
Unsupported constructs are captured as OpaqueBlock nodes with diagnostics.

Whitelisted functions:
- RobotContext.run_action, RobotContext.stop, RobotContext.get_sensor_data
- time.sleep, range, abs, min, max, sum, len, print
"""

from __future__ import annotations
from typing import List, Optional, Tuple

from compiler.parser.lexer import Token, TokenType, Lexer
from compiler.parser.ast_nodes import (
    ASTNode, Module, Assignment, ExpressionStatement,
    NumberLiteral, StringLiteral, BoolLiteral, Identifier, AttributeAccess,
    BinaryOp, UnaryOp, CompareOp, BooleanOp, NotOp,
    FunctionCall, IfStatement, ElifClause, WhileStatement, ForRangeStatement,
    PassStatement, ReturnStatement, BreakStatement, ContinueStatement,
    ImportStatement, CommentNode, OpaqueBlock, FunctionDef,
)
from compiler.semantic.diagnostics import Diagnostic, make_warning, make_info


class ParseError(Exception):
    """Parser error with position info."""
    def __init__(self, message: str, line: int = 0, col: int = 0):
        super().__init__(f"Line {line}:{col}: {message}")
        self.line = line
        self.col = col


class Parser:
    """
    Parse a token stream into an AST.

    Usage:
        parser = Parser(source_code)
        module, diagnostics = parser.parse()
    """

    def __init__(self, source: str):
        self._source = source
        self._source_lines = source.split("\n")
        self._tokens: List[Token] = []
        self._pos = 0
        self._diags: List[Diagnostic] = []

    def parse(self) -> Tuple[Module, List[Diagnostic]]:
        """Parse the source and return (Module AST, diagnostics)."""
        lexer = Lexer(self._source)
        try:
            self._tokens = lexer.tokenize()
        except Exception as e:
            self._diags.append(make_warning("E1001", f"Lexer error: {e}"))
            return Module(body=[OpaqueBlock(code=self._source)]), self._diags

        self._pos = 0
        self._diags = []

        body = self._parse_block(top_level=True)
        module = Module(body=body, line=1, col=1)
        return module, self._diags

    # ---------- Token stream helpers ----------

    def _peek(self) -> Token:
        """Look at current token without consuming."""
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return Token(TokenType.EOF, "", 0, 0)

    def _advance(self) -> Token:
        """Consume and return current token."""
        tok = self._peek()
        self._pos += 1
        return tok

    def _expect(self, token_type: TokenType) -> Token:
        """Consume a token of the expected type, or raise error."""
        tok = self._peek()
        if tok.type != token_type:
            raise ParseError(
                f"Expected {token_type.name}, got {tok.type.name} ({tok.value!r})",
                tok.line, tok.col,
            )
        return self._advance()

    def _match(self, *types: TokenType) -> Optional[Token]:
        """If current token matches any of the types, consume and return it."""
        if self._peek().type in types:
            return self._advance()
        return None

    def _skip_newlines(self, collect_comments: List[ASTNode] = None):
        """Skip NEWLINE tokens. Optionally collect COMMENT tokens into a list."""
        while self._peek().type in (TokenType.NEWLINE, TokenType.COMMENT):
            tok = self._peek()
            if tok.type == TokenType.COMMENT:
                self._advance()
                if collect_comments is not None:
                    collect_comments.append(
                        CommentNode(text=tok.value, line=tok.line, col=tok.col))
            else:
                self._advance()

    # ---------- Block parsing ----------

    def _parse_block(self, top_level: bool = False) -> List[ASTNode]:
        """Parse a block of statements (after INDENT or at top level)."""
        stmts: List[ASTNode] = []

        while True:
            self._skip_newlines(collect_comments=stmts)
            tok = self._peek()

            if tok.type == TokenType.EOF:
                break
            if tok.type == TokenType.DEDENT and not top_level:
                self._advance()
                break
            if top_level and tok.type in (TokenType.DEDENT, TokenType.INDENT):
                # Recover from malformed top-level indentation by consuming it.
                self._diags.append(make_warning(
                    "E1002",
                    f"Unexpected {tok.type.name} at top level",
                    node_id=None,
                ))
                self._advance()
                continue

            try:
                stmt = self._parse_statement()
                if stmt is not None:
                    stmts.append(stmt)
            except ParseError as e:
                # Recover: capture the rest of the line as opaque
                self._diags.append(make_warning(
                    "E1002", f"Parse error: {e}", node_id=None))
                opaque = self._recover_to_newline()
                if opaque:
                    stmts.append(opaque)

        return stmts

    def _recover_to_newline(self) -> Optional[OpaqueBlock]:
        """Skip tokens until next NEWLINE, producing an OpaqueBlock."""
        line = self._peek().line
        start_pos = self._pos
        parts = []
        while self._peek().type not in (TokenType.NEWLINE, TokenType.EOF,
                                         TokenType.DEDENT):
            tok = self._advance()
            parts.append(tok.value)

        # Ensure forward progress on indentation tokens that can stall recovery.
        if self._pos == start_pos and self._peek().type in (TokenType.DEDENT, TokenType.INDENT):
            self._advance()

        if self._peek().type == TokenType.NEWLINE:
            self._advance()

        if parts:
            # Try to get the original source line
            if 0 < line <= len(self._source_lines):
                return OpaqueBlock(code=self._source_lines[line - 1].strip(),
                                   line=line)
            return OpaqueBlock(code=" ".join(parts), line=line)
        return None

    # ---------- Statement parsing ----------

    def _parse_statement(self) -> Optional[ASTNode]:
        """Parse a single statement."""
        tok = self._peek()

        if tok.type == TokenType.COMMENT:
            self._advance()
            return CommentNode(text=tok.value, line=tok.line, col=tok.col)

        if tok.type == TokenType.NEWLINE:
            self._advance()
            return None

        if tok.type == TokenType.IF:
            return self._parse_if()

        if tok.type == TokenType.WHILE:
            return self._parse_while()

        if tok.type == TokenType.FOR:
            return self._parse_for()

        if tok.type == TokenType.DEF:
            return self._parse_def()

        if tok.type == TokenType.PASS:
            self._advance()
            self._match(TokenType.NEWLINE)
            return PassStatement(line=tok.line, col=tok.col)

        if tok.type == TokenType.RETURN:
            return self._parse_return()

        if tok.type == TokenType.BREAK:
            self._advance()
            self._match(TokenType.NEWLINE)
            return BreakStatement(line=tok.line, col=tok.col)

        if tok.type == TokenType.CONTINUE:
            self._advance()
            self._match(TokenType.NEWLINE)
            return ContinueStatement(line=tok.line, col=tok.col)

        if tok.type in (TokenType.IMPORT, TokenType.FROM):
            return self._parse_import()

        # Assignment or expression statement
        return self._parse_assignment_or_expr()

    def _parse_assignment_or_expr(self) -> ASTNode:
        """Parse assignment (name = expr) or expression statement."""
        tok = self._peek()
        line, col = tok.line, tok.col

        # Check for simple assignment: IDENTIFIER = expr
        if tok.type == TokenType.IDENTIFIER:
            next_pos = self._pos + 1
            if next_pos < len(self._tokens):
                next_tok = self._tokens[next_pos]
                if next_tok.type in (TokenType.ASSIGN, TokenType.PLUS_ASSIGN,
                                     TokenType.MINUS_ASSIGN, TokenType.STAR_ASSIGN,
                                     TokenType.SLASH_ASSIGN):
                    name_tok = self._advance()
                    op_tok = self._advance()

                    value = self._parse_expression()
                    self._match(TokenType.NEWLINE)

                    if op_tok.type == TokenType.ASSIGN:
                        return Assignment(target=name_tok.value, value=value,
                                          line=line, col=col)
                    else:
                        # Augmented assignment: x += 1 => x = x + 1
                        aug_ops = {
                            TokenType.PLUS_ASSIGN: "+",
                            TokenType.MINUS_ASSIGN: "-",
                            TokenType.STAR_ASSIGN: "*",
                            TokenType.SLASH_ASSIGN: "/",
                        }
                        op_str = aug_ops[op_tok.type]
                        desugar = BinaryOp(
                            left=Identifier(name=name_tok.value, line=line, col=col),
                            op=op_str, right=value, line=line, col=col,
                        )
                        return Assignment(target=name_tok.value, value=desugar,
                                          line=line, col=col)

        # Expression statement (e.g. function call)
        expr = self._parse_expression()
        self._match(TokenType.NEWLINE)
        return ExpressionStatement(expression=expr, line=line, col=col)

    # ---------- Control flow ----------

    def _parse_if(self) -> IfStatement:
        """Parse if/elif/else.

        Resilient: if the condition expression cannot be fully parsed
        (e.g. contains spaces in identifiers), we recover by collecting
        raw tokens up to the colon and wrapping them as an Identifier.
        """
        tok = self._expect(TokenType.IF)
        condition = self._parse_condition_resilient(tok.line, tok.col)
        self._match(TokenType.NEWLINE)

        body = self._parse_indented_body()

        elifs: List[ElifClause] = []
        else_body: List[ASTNode] = []

        while self._peek().type == TokenType.ELIF:
            elif_tok = self._advance()
            elif_cond = self._parse_condition_resilient(elif_tok.line, elif_tok.col)
            self._match(TokenType.NEWLINE)
            elif_body = self._parse_indented_body()
            elifs.append(ElifClause(condition=elif_cond, body=elif_body,
                                    line=elif_tok.line))

        if self._peek().type == TokenType.ELSE:
            self._advance()
            self._match(TokenType.COLON)
            self._match(TokenType.NEWLINE)
            else_body = self._parse_indented_body()

        return IfStatement(
            condition=condition, body=body, elifs=elifs,
            else_body=else_body, line=tok.line, col=tok.col,
        )

    def _parse_condition_resilient(self, line: int, col: int) -> ASTNode:
        """Parse a condition expression, recovering if it fails.

        First tries normal expression parsing + colon. If that fails,
        rewinds and collects raw tokens up to ':' as a raw identifier.
        """
        saved_pos = self._pos
        try:
            condition = self._parse_expression()
            self._expect(TokenType.COLON)
            return condition
        except ParseError:
            # Rewind to before the expression
            self._pos = saved_pos
            # Collect raw tokens until colon
            parts = []
            while self._peek().type not in (TokenType.COLON, TokenType.NEWLINE,
                                             TokenType.EOF):
                parts.append(self._advance().value)
            raw_text = " ".join(parts).strip() or "condition"
            self._diags.append(make_warning(
                "W1003",
                f"Condition expression '{raw_text}' could not be fully parsed; "
                f"preserved as raw text",
                node_id=None,
            ))
            self._match(TokenType.COLON)
            return Identifier(name=raw_text, line=line, col=col)

    def _parse_while(self) -> WhileStatement:
        """Parse while loop."""
        tok = self._expect(TokenType.WHILE)
        condition = self._parse_condition_resilient(tok.line, tok.col)
        self._match(TokenType.NEWLINE)

        body = self._parse_indented_body()

        return WhileStatement(condition=condition, body=body,
                              line=tok.line, col=tok.col)

    def _parse_for(self) -> ForRangeStatement:
        """Parse for i in range(...) loop."""
        tok = self._expect(TokenType.FOR)
        var_tok = self._expect(TokenType.IDENTIFIER)
        self._expect(TokenType.IN)

        # Expect range(...)
        range_tok = self._peek()
        if range_tok.type == TokenType.IDENTIFIER and range_tok.value == "range":
            self._advance()
            self._expect(TokenType.LPAREN)
            args = self._parse_call_args()

            start = NumberLiteral(value=0, line=tok.line)
            end = NumberLiteral(value=10, line=tok.line)
            step = NumberLiteral(value=1, line=tok.line)

            if len(args) == 1:
                end = args[0]
            elif len(args) == 2:
                start = args[0]
                end = args[1]
            elif len(args) >= 3:
                start = args[0]
                end = args[1]
                step = args[2]
        else:
            # Not a range() call - parse as opaque
            self._diags.append(make_warning(
                "E1003",
                f"Only 'for x in range(...)' is supported; "
                f"found 'for {var_tok.value} in {range_tok.value}...'",
            ))
            # Recover
            start = NumberLiteral(value=0, line=tok.line)
            end = NumberLiteral(value=10, line=tok.line)
            step = NumberLiteral(value=1, line=tok.line)
            # Skip to colon
            while self._peek().type not in (TokenType.COLON, TokenType.NEWLINE,
                                             TokenType.EOF):
                self._advance()

        self._expect(TokenType.COLON)
        self._match(TokenType.NEWLINE)
        body = self._parse_indented_body()

        return ForRangeStatement(
            variable=var_tok.value, start=start, end=end, step=step,
            body=body, line=tok.line, col=tok.col,
        )

    def _parse_def(self) -> ASTNode:
        """Parse function definition (treated as opaque for IR purposes)."""
        tok = self._expect(TokenType.DEF)
        name_tok = self._expect(TokenType.IDENTIFIER)

        # Skip to colon (skip params)
        while self._peek().type not in (TokenType.COLON, TokenType.NEWLINE,
                                         TokenType.EOF):
            self._advance()
        self._match(TokenType.COLON)
        self._match(TokenType.NEWLINE)

        body = self._parse_indented_body()

        self._diags.append(make_info(
            "I1001",
            f"Function definition '{name_tok.value}' captured (may contain workflow entry)",
        ))

        return FunctionDef(name=name_tok.value, body=body,
                           line=tok.line, col=tok.col)

    def _parse_return(self) -> ReturnStatement:
        """Parse return statement."""
        tok = self._advance()  # consume 'return'
        value = None
        if self._peek().type not in (TokenType.NEWLINE, TokenType.EOF,
                                      TokenType.DEDENT):
            value = self._parse_expression()
        self._match(TokenType.NEWLINE)
        return ReturnStatement(value=value, line=tok.line, col=tok.col)

    def _parse_import(self) -> ImportStatement:
        """Parse import / from...import statement."""
        tok = self._peek()
        is_from = tok.type == TokenType.FROM

        parts = []
        # Consume all tokens until NEWLINE
        while self._peek().type not in (TokenType.NEWLINE, TokenType.EOF):
            parts.append(self._advance().value)

        self._match(TokenType.NEWLINE)

        # Parse the import text
        text = " ".join(parts)
        if is_from:
            # from X import Y, Z
            # parts: ['from', 'X', 'import', 'Y', ',', 'Z']
            module = ""
            names = []
            state = "from"
            for p in parts:
                if p == "from":
                    state = "module"
                elif p == "import":
                    state = "names"
                elif state == "module":
                    module = p if not module else module + "." + p
                elif state == "names" and p != ",":
                    names.append(p)
            return ImportStatement(module=module, names=names, is_from=True,
                                   line=tok.line, col=tok.col)
        else:
            # import X
            module = parts[1] if len(parts) > 1 else ""
            return ImportStatement(module=module, names=[], is_from=False,
                                   line=tok.line, col=tok.col)

    # ---------- Indented body ----------

    def _parse_indented_body(self) -> List[ASTNode]:
        """Parse an indented block (expects INDENT ... DEDENT)."""
        self._skip_newlines()

        if self._peek().type == TokenType.INDENT:
            self._advance()
            return self._parse_block(top_level=False)

        # Single-line body (e.g. "if True: pass")
        stmts = []
        stmt = self._parse_statement()
        if stmt:
            stmts.append(stmt)
        return stmts

    # ---------- Expression parsing (Pratt-style precedence climbing) ----------

    def _parse_expression(self) -> ASTNode:
        """Parse an expression (entry point)."""
        return self._parse_or_expr()

    def _parse_or_expr(self) -> ASTNode:
        """Parse: expr ('or' expr)*"""
        left = self._parse_and_expr()
        while self._peek().type == TokenType.OR:
            op_tok = self._advance()
            right = self._parse_and_expr()
            left = BooleanOp(left=left, op="or", right=right,
                             line=op_tok.line, col=op_tok.col)
        return left

    def _parse_and_expr(self) -> ASTNode:
        """Parse: expr ('and' expr)*"""
        left = self._parse_not_expr()
        while self._peek().type == TokenType.AND:
            op_tok = self._advance()
            right = self._parse_not_expr()
            left = BooleanOp(left=left, op="and", right=right,
                             line=op_tok.line, col=op_tok.col)
        return left

    def _parse_not_expr(self) -> ASTNode:
        """Parse: 'not' expr | comparison"""
        if self._peek().type == TokenType.NOT:
            op_tok = self._advance()
            operand = self._parse_not_expr()
            return NotOp(operand=operand, line=op_tok.line, col=op_tok.col)
        return self._parse_comparison()

    def _parse_comparison(self) -> ASTNode:
        """Parse: expr (comp_op expr)*"""
        left = self._parse_addition()
        comp_ops = {TokenType.EQ, TokenType.NEQ, TokenType.LT,
                    TokenType.GT, TokenType.LTE, TokenType.GTE}

        while self._peek().type in comp_ops:
            op_tok = self._advance()
            right = self._parse_addition()
            left = CompareOp(left=left, op=op_tok.value, right=right,
                             line=op_tok.line, col=op_tok.col)
        return left

    def _parse_addition(self) -> ASTNode:
        """Parse: term (('+' | '-') term)*"""
        left = self._parse_multiplication()
        while self._peek().type in (TokenType.PLUS, TokenType.MINUS):
            op_tok = self._advance()
            right = self._parse_multiplication()
            left = BinaryOp(left=left, op=op_tok.value, right=right,
                            line=op_tok.line, col=op_tok.col)
        return left

    def _parse_multiplication(self) -> ASTNode:
        """Parse: unary (('*' | '/' | '//' | '%') unary)*"""
        left = self._parse_power()
        while self._peek().type in (TokenType.STAR, TokenType.SLASH,
                                     TokenType.DOUBLE_SLASH, TokenType.PERCENT):
            op_tok = self._advance()
            right = self._parse_power()
            left = BinaryOp(left=left, op=op_tok.value, right=right,
                            line=op_tok.line, col=op_tok.col)
        return left

    def _parse_power(self) -> ASTNode:
        """Parse: unary ('**' unary)*"""
        left = self._parse_unary()
        while self._peek().type == TokenType.DOUBLE_STAR:
            op_tok = self._advance()
            right = self._parse_unary()
            left = BinaryOp(left=left, op="**", right=right,
                            line=op_tok.line, col=op_tok.col)
        return left

    def _parse_unary(self) -> ASTNode:
        """Parse: ('-' | '+') unary | primary"""
        if self._peek().type in (TokenType.MINUS, TokenType.PLUS):
            op_tok = self._advance()
            operand = self._parse_unary()
            return UnaryOp(op=op_tok.value, operand=operand,
                           line=op_tok.line, col=op_tok.col)
        return self._parse_primary()

    def _parse_primary(self) -> ASTNode:
        """Parse primary expression (literals, identifiers, calls, parens)."""
        tok = self._peek()

        # Parenthesized expression
        if tok.type == TokenType.LPAREN:
            self._advance()
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN)
            return expr

        # Integer literal
        if tok.type == TokenType.INTEGER:
            self._advance()
            return NumberLiteral(value=int(tok.value), line=tok.line, col=tok.col)

        # Float literal
        if tok.type == TokenType.FLOAT:
            self._advance()
            return NumberLiteral(value=float(tok.value), line=tok.line, col=tok.col)

        # String literal
        if tok.type == TokenType.STRING:
            self._advance()
            return StringLiteral(value=tok.value, line=tok.line, col=tok.col)

        # Boolean literals
        if tok.type == TokenType.TRUE:
            self._advance()
            return BoolLiteral(value=True, line=tok.line, col=tok.col)
        if tok.type == TokenType.FALSE:
            self._advance()
            return BoolLiteral(value=False, line=tok.line, col=tok.col)

        # None literal (treat as 0 for simplicity)
        if tok.type == TokenType.NONE:
            self._advance()
            return Identifier(name="None", line=tok.line, col=tok.col)

        # Identifier (may be followed by dot access or function call)
        if tok.type == TokenType.IDENTIFIER:
            return self._parse_identifier_or_call()

        raise ParseError(f"Unexpected token: {tok.type.name} ({tok.value!r})",
                         tok.line, tok.col)

    def _parse_identifier_or_call(self) -> ASTNode:
        """Parse identifier, attribute access, or function call."""
        tok = self._advance()
        node: ASTNode = Identifier(name=tok.value, line=tok.line, col=tok.col)

        # Dot access chain
        while self._peek().type == TokenType.DOT:
            self._advance()
            attr_tok = self._expect(TokenType.IDENTIFIER)
            node = AttributeAccess(object=node, attribute=attr_tok.value,
                                   line=tok.line, col=tok.col)

        # Function call
        if self._peek().type == TokenType.LPAREN:
            self._advance()
            args = self._parse_call_args()
            node = FunctionCall(func=node, args=args,
                                line=tok.line, col=tok.col)

        # Index access (treat as opaque for now)
        if self._peek().type == TokenType.LBRACKET:
            self._advance()
            while self._peek().type not in (TokenType.RBRACKET, TokenType.EOF,
                                             TokenType.NEWLINE):
                self._advance()
            self._match(TokenType.RBRACKET)

        return node

    def _parse_call_args(self) -> List[ASTNode]:
        """Parse function call arguments (between parens, closing paren consumed)."""
        args: List[ASTNode] = []
        if self._peek().type == TokenType.RPAREN:
            self._advance()
            return args

        args.append(self._parse_expression())
        while self._peek().type == TokenType.COMMA:
            self._advance()
            if self._peek().type == TokenType.RPAREN:
                break
            args.append(self._parse_expression())

        self._expect(TokenType.RPAREN)
        return args
