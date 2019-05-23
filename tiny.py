#!/usr/bin/env python3
"""
A tiny interpreter for a simple programming language. Includes a parser, bytecode
compiler, and virtual machine, as well as a full test suite, with no dependencies
other than Python 3 and the standard library.

Example code:
    >>> let x = 32
    >>> x
    32
    >>> x + 10
    42
    >>> fn add(x, y) = x + y
    >>> add(x, 10)
    42
    >>> if true then 42 else 666 end
    42

Author:  Ian Fisher (iafisher@protonmail.com)
Version: May 2019
"""
import argparse
import re
import readline
import sys
import unittest
from collections import namedtuple, ChainMap
from typing import Union, Dict, Optional, List, Tuple, cast, Any


###################
#  PARSING STAGE  #
###################


def iparse(text):
    """
    Parse the expression string according to this grammar:

    start     := expr | define
    define    := LET IDENT EQ expr
    function  := FN IDENT LPAREN params RPAREN EQ expr
    if        := IF expr THEN expr ELSE expr END
    expr      := expr OP expr  | call | NUMBER | TRUE | FALSE | IDENT

    call      := IDENT LPAREN args RPAREN
    args      := expr | expr COMMA args
    params    := IDENT | IDENT COMMA params

    Operators have the following precedence:
      1. *, /
      2. +, -
    """
    return TinyParser(TinyLexer(text)).parse()


class TinyParser:
    """
    The parser for the tiny language.

    All match_* functions assume that self.lexer.tkn is set at the first token of the
    expression to be matched, and they all leave self.lexer.tkn at one token past the
    end of the matched expression.
    """

    def __init__(self, lexer):
        self.lexer = lexer

    def parse(self):
        if self.lexer.tkn.type == "TOKEN_LET":
            tree = self.match_let()
        elif self.lexer.tkn.type == "TOKEN_FN":
            tree = self.match_fn()
        else:
            tree = self.match_expr(PREC_LOWEST)

        self.expect("TOKEN_EOF")
        return tree

    def match_let(self):
        """Match a let statement."""
        self.expect("TOKEN_LET")
        self.lexer.next_token()

        self.expect("TOKEN_SYMBOL")
        sym = self.lexer.tkn.value
        self.lexer.next_token()

        self.expect("TOKEN_ASSIGN")
        self.lexer.next_token()

        rhs = self.match_expr(PREC_LOWEST)
        return LetNode(sym, rhs)

    def match_fn(self):
        """Match a function declaration."""
        self.expect("TOKEN_FN")
        self.lexer.next_token()

        self.expect("TOKEN_SYMBOL")
        sym = self.lexer.tkn.value
        self.lexer.next_token()

        self.expect("TOKEN_LPAREN")
        self.lexer.next_token()
        params = self.match_params()
        self.expect("TOKEN_RPAREN")
        self.lexer.next_token()

        self.expect("TOKEN_ASSIGN")
        self.lexer.next_token()

        body = self.match_expr(PREC_LOWEST)
        return FnNode(sym, params, body)

    def match_expr(self, prec):
        """
        Match an expression, assuming that self.lexer.tkn is set at the first token of
        the expression.

        On exit, self.lexer.tkn will be set to the first token of the next expression.
        """
        left = self.match_prefix()

        tkn = self.lexer.tkn
        while tkn.type in PREC_MAP and prec < PREC_MAP[tkn.type]:
            left = self.match_infix(left, PREC_MAP[tkn.type])
            tkn = self.lexer.tkn
        return left

    def match_prefix(self):
        """Match a non-infix expression."""
        tkn = self.lexer.tkn
        if tkn.type == "TOKEN_INT":
            left = int(tkn.value)
            self.lexer.next_token()
        elif tkn.type == "TOKEN_TRUE":
            left = True
            self.lexer.next_token()
        elif tkn.type == "TOKEN_FALSE":
            left = False
            self.lexer.next_token()
        elif tkn.type == "TOKEN_SYMBOL":
            left = tkn.value
            self.lexer.next_token()
        elif tkn.type == "TOKEN_LPAREN":
            self.lexer.next_token()
            left = self.match_expr(PREC_LOWEST)
            self.expect("TOKEN_RPAREN")
            self.lexer.next_token()
        elif tkn.type == "TOKEN_MINUS":
            self.lexer.next_token()
            left = PrefixNode("-", self.match_expr(PREC_PREFIX))
        elif tkn.type == "TOKEN_IF":
            left = self.match_if()
        else:
            raise IParseError(
                "unexpected token '{0.value}', line {0.line} col {0.column}".format(tkn)
            )

        return left

    def match_infix(self, left, prec):
        """Match the right half of an infix expression."""
        tkn = self.lexer.tkn
        self.lexer.next_token()

        if tkn.type == "TOKEN_LPAREN":
            args = self.match_args()
            self.expect("TOKEN_RPAREN")
            self.lexer.next_token()
            return CallNode(left, args)
        else:
            right = self.match_expr(prec)
            return InfixNode(tkn.value, left, right)

    def match_if(self):
        self.expect("TOKEN_IF")
        self.lexer.next_token()

        condition = self.match_expr(PREC_LOWEST)
        self.expect("TOKEN_THEN")
        self.lexer.next_token()

        true_clause = self.match_expr(PREC_LOWEST)
        self.expect("TOKEN_ELSE")
        self.lexer.next_token()

        false_clause = self.match_expr(PREC_LOWEST)
        self.expect("TOKEN_END")
        self.lexer.next_token()

        return IfNode(condition, true_clause, false_clause)

    def match_args(self):
        """Match the argument list of a call expression."""
        args = []
        while True:
            arg = self.match_expr(PREC_LOWEST)
            args.append(arg)

            if self.lexer.tkn.type == "TOKEN_COMMA":
                self.lexer.next_token()
            else:
                break
        return args

    def match_params(self):
        """Match the parameter list of a function declaration."""
        params = []
        while True:
            self.expect("TOKEN_SYMBOL")
            params.append(self.lexer.tkn.value)

            self.lexer.next_token()
            if self.lexer.tkn.type == "TOKEN_COMMA":
                self.lexer.next_token()
            else:
                break
        return params

    def expect(self, typ):
        """Raise an error if the lexer's current token is not of the given type."""
        if self.lexer.tkn.type != typ:
            if typ == "TOKEN_EOF":
                raise IParseError("trailing input")
            elif self.lexer.tkn.type == "TOKEN_EOF":
                raise IParseError("premature end of input")
            else:
                raise IParseError(
                    "unexpected token '{0.value}', line {0.line} col {0.column}".format(
                        self.lexer.tkn
                    )
                )


CallNode = namedtuple("CallNode", ["name", "args"])
InfixNode = namedtuple("InfixNode", ["op", "left", "right"])
PrefixNode = namedtuple("PrefixNode", ["op", "operand"])
LetNode = namedtuple("LetNode", ["symbol", "expr"])
FnNode = namedtuple("FnNode", ["symbol", "params", "body"])
IfNode = namedtuple("IfNode", ["condition", "true_clause", "false_clause"])


class TinyLexer:
    """
    The lexer for the tiny language.

    The parser drives the lexical analysis by calling the next_token method.
    """

    keywords = frozenset(["fn", "let", "if", "then", "else", "end", "true", "false"])

    def __init__(self, text):
        self.text = text
        self.position = 0
        self.column = 1
        self.line = 1
        # Set the current token.
        self.next_token()

    def next_token(self):
        self.skip_whitespace()

        if self.position >= len(self.text):
            self.set_token("TOKEN_EOF", 0)
        else:
            ch = self.text[self.position]
            if ch.isalpha() or ch == "_":
                length = self.read_symbol()
                value = self.text[self.position : self.position + length]
                if value in self.keywords:
                    self.set_token("TOKEN_" + value.upper(), length)
                else:
                    self.set_token("TOKEN_SYMBOL", length)
            elif ch.isdigit():
                length = self.read_int()
                self.set_token("TOKEN_INT", length)
            elif ch == "(":
                self.set_token("TOKEN_LPAREN", 1)
            elif ch == ")":
                self.set_token("TOKEN_RPAREN", 1)
            elif ch == ",":
                self.set_token("TOKEN_COMMA", 1)
            elif ch == "+":
                self.set_token("TOKEN_PLUS", 1)
            elif ch == "*":
                self.set_token("TOKEN_ASTERISK", 1)
            elif ch == "-":
                self.set_token("TOKEN_MINUS", 1)
            elif ch == "/":
                self.set_token("TOKEN_SLASH", 1)
            elif ch == "=":
                self.set_token("TOKEN_ASSIGN", 1)
            else:
                self.set_token("TOKEN_UNKNOWN", 1)

        return self.tkn

    def skip_whitespace(self):
        while self.position < len(self.text) and self.text[self.position].isspace():
            self.next_char()

    def read_symbol(self):
        end = self.position + 1
        while end < len(self.text) and is_symbol_char(self.text[end]):
            end += 1
        return end - self.position

    def read_int(self):
        end = self.position + 1
        while end < len(self.text) and self.text[end].isdigit():
            end += 1
        return end - self.position

    def next_char(self):
        if self.text[self.position] == "\n":
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        self.position += 1

    def set_token(self, typ, length):
        value = self.text[self.position : self.position + length]
        self.tkn = Token(typ, value, self.line, self.column)

        # We can't just do self.position += length because self.line and self.column
        # would no longer be accurate.
        for _ in range(length):
            self.next_char()


Token = namedtuple("Token", ["type", "value", "line", "column"])


PREC_LOWEST = 0
PREC_ADD_SUB = 1
PREC_MUL_DIV = 2
PREC_PREFIX = 3
PREC_CALL = 4

PREC_MAP = {
    "TOKEN_PLUS": PREC_ADD_SUB,
    "TOKEN_MINUS": PREC_ADD_SUB,
    "TOKEN_ASTERISK": PREC_MUL_DIV,
    "TOKEN_SLASH": PREC_MUL_DIV,
    # The left parenthesis is the "infix operator" for function-call expressions.
    "TOKEN_LPAREN": PREC_CALL,
}


def wrap(node):
    """Stringify the parse tree node and wrap it in parentheses if it might be
    ambiguous.
    """
    if isinstance(node, (IntNode, CallNode, SymbolNode)):
        return str(node)
    else:
        return "(" + str(node) + ")"


def is_symbol_char(c):
    return c.isdigit() or c.isalpha() or c == "_"


class Function(namedtuple("Function", ["name", "parameters", "code"])):
    """
    Internal representation of functions. The `name` field is only used for error
    messages.
    """

    def __str__(self):
        return '<function "{0.name}">'.format(self)


# Type declarations for mypy.
EnvType = Union[Dict[str, int], ChainMap]
BytecodeType = Tuple[str, Union[Function, int, str]]


#######################
#  COMPILATION STAGE  #
#######################

# Definition of bytecode instruction names.
BINARY_ADD = "BINARY_ADD"
BINARY_SUB = "BINARY_SUB"
BINARY_MUL = "BINARY_MUL"
BINARY_DIV = "BINARY_DIV"
LOAD_CONST = "LOAD_CONST"
STORE_NAME = "STORE_NAME"
LOAD_NAME = "LOAD_NAME"
CALL_FUNCTION = "CALL_FUNCTION"
POP_JUMP_IF_FALSE = "POP_JUMP_IF_FALSE"
JUMP_FORWARD = "JUMP_FORWARD"


# Placeholder value for instructions without arguments.
NO_ARG = 0


def icompile(ast) -> List[BytecodeType]:
    """
    Compile the AST into a list of bytecode instructions of the form (instruction, arg).
    """
    if isinstance(ast, InfixNode):
        ret = icompile(ast.left) + icompile(ast.right)
        if ast.op == "+":
            ret.append((BINARY_ADD, NO_ARG))
        elif ast.op == "-":
            ret.append((BINARY_SUB, NO_ARG))
        elif ast.op == "*":
            ret.append((BINARY_MUL, NO_ARG))
        elif ast.op == "/":
            ret.append((BINARY_DIV, NO_ARG))
        else:
            raise RuntimeError("unknown operator '{}'".format(ast.op))
        return ret
    elif isinstance(ast, IfNode):
        condition_code = icompile(ast.condition)
        true_code = icompile(ast.true_clause)
        false_code = icompile(ast.false_clause)

        true_code.append((JUMP_FORWARD, len(false_code) + 1))
        condition_code.append((POP_JUMP_IF_FALSE, len(true_code) + 1))
        return condition_code + true_code + false_code
    elif isinstance(ast, CallNode):
        ret = []
        for arg in reversed(ast.args):
            ret += icompile(arg)
        ret.append((LOAD_NAME, ast.name))
        ret.append((CALL_FUNCTION, len(ast.args)))
        return ret
    elif isinstance(ast, LetNode):
        ret = icompile(ast.expr)
        ret.append((STORE_NAME, ast.symbol))
        return ret
    elif isinstance(ast, FnNode):
        f = Function(ast.symbol, ast.params, icompile(ast.body))
        return [(LOAD_CONST, f), (STORE_NAME, ast.symbol)]
    elif isinstance(ast, str):
        return [(LOAD_NAME, ast)]
    elif isinstance(ast, int):
        return [(LOAD_CONST, ast)]
    elif isinstance(ast, Function):
        return [(LOAD_CONST, ast)]
    else:
        raise ValueError(
            'don\'t know how to compile object of type "{}"'.format(type(ast))
        )


#####################
#  EXECUTION STAGE  #
#####################


def iexec(codeobj: List[BytecodeType], env: EnvType) -> Optional[int]:
    """Execute a code object in the given environment."""
    stack = []  # type: List[Any]
    pc = 0
    while pc < len(codeobj):
        inst, arg = codeobj[pc]
        if inst == LOAD_CONST:
            stack.append(arg)
            pc += 1
        elif inst == BINARY_ADD:
            right = stack.pop()
            left = stack.pop()
            stack.append(left + right)
            pc += 1
        elif inst == BINARY_SUB:
            right = stack.pop()
            left = stack.pop()
            stack.append(left - right)
            pc += 1
        elif inst == BINARY_MUL:
            right = stack.pop()
            left = stack.pop()
            stack.append(left * right)
            pc += 1
        elif inst == BINARY_DIV:
            right = stack.pop()
            left = stack.pop()
            stack.append(left / right)
            pc += 1
        elif inst == STORE_NAME:
            env[cast(str, arg)] = stack.pop()
            pc += 1
        elif inst == LOAD_NAME:
            try:
                stack.append(env[cast(str, arg)])
            except KeyError:
                raise IExecutionError('unbound identifier "{}"'.format(arg))
            pc += 1
        elif inst == POP_JUMP_IF_FALSE:
            top = stack.pop()
            if top:
                pc += 1
            else:
                pc += arg
        elif inst == JUMP_FORWARD:
            pc += arg
        elif inst == CALL_FUNCTION:
            new_env = ChainMap({}, env)
            function = stack.pop()
            if isinstance(function, Function):
                # Make sure enough arguments were passed to the function originally.
                if len(function.parameters) != arg:
                    msg = 'wrong number of arguments to function "{}"'.format(
                        function.name
                    )
                    raise IExecutionError(msg)
                for param in function.parameters:
                    val = stack.pop()
                    new_env[param] = val
                res = iexec(function.code, new_env)
                if res is not None:
                    stack.append(res)
            else:
                raise IExecutionError("first value of expression must be function")
            pc += 1
        else:
            raise ValueError('unrecognized bytecode instruction "{}"'.format(inst))
    if stack:
        return stack.pop()
    else:
        return None


# This error hierarchy allows me to distinguish between errors in my code, signalled by
# regular Python exceptions, and errors in the code being interpreted, signalled by
# IError exceptions.


class IError(Exception):
    pass


class IParseError(IError):
    pass


class IExecutionError(IError):
    pass


def ieval(expr: str, env: EnvType) -> Optional[int]:
    """A shortcut function to parse, compile and execute an expression."""
    return iexec(icompile(iparse(expr)), env)


################
#  TEST SUITE  #
################


class ParseTests(unittest.TestCase):
    def test_expr(self):
        self.assertEqual(iparse("1"), 1)
        self.assertEqual(iparse("x"), "x")
        self.assertEqual(iparse("8 + 2"), InfixNode("+", 8, 2))
        self.assertEqual(
            iparse("((4 * 2) + (3 - 1))"),
            InfixNode("+", InfixNode("*", 4, 2), InfixNode("-", 3, 1)),
        )
        self.assertEqual(iparse("f(1, 2, 3, 4)"), CallNode("f", [1, 2, 3, 4]))
        self.assertEqual(
            iparse("f(1, (1 + 1), 3, 4)"),
            CallNode("f", [1, InfixNode("+", 1, 1), 3, 4]),
        )

    def test_define(self):
        self.assertEqual(iparse("let x = 10"), LetNode("x", 10))
        self.assertEqual(iparse("let x = 5 * 2"), LetNode("x", InfixNode("*", 5, 2)))
        self.assertEqual(
            iparse("let x = 5 * (1 + 1)"),
            LetNode("x", InfixNode("*", 5, InfixNode("+", 1, 1))),
        )

    def test_if_else(self):
        self.assertEqual(iparse("if x then 42 else 666 end"), IfNode("x", 42, 666))
        self.assertEqual(
            iparse("if x then if y then 42 else 0 end else 666 end"),
            IfNode("x", IfNode("y", 42, 0), 666),
        )

    def test_function(self):
        self.assertEqual(iparse("fn f(x) = 10"), FnNode("f", ["x"], 10))

    def test_errors(self):
        # Trailing input is not allowed.
        with self.assertRaises(IParseError):
            iparse("1 1")
        with self.assertRaises(IParseError):
            iparse("1 + 2   1")
        with self.assertRaises(IParseError):
            iparse("let x = 10   11")
        with self.assertRaises(IParseError):
            iparse("let x = 2 * 10    11")
        with self.assertRaises(IParseError):
            iparse("fn f(x) = 10   11")
        with self.assertRaises(IParseError):
            iparse("fn f(x) = 2 * 10    11")


class ExecTests(unittest.TestCase):
    def test_atoms(self):
        env = {}
        self.assertEqual(ieval("42", env), 42)
        self.assertEqual(ieval("true", env), True)
        self.assertEqual(ieval("false", env), False)

    def test_arithmetic(self):
        env = {}
        self.assertEqual(ieval("1 + 1", env), 2)
        self.assertEqual(ieval("31 + 11", env), 42)
        self.assertEqual(ieval("(33 - 2) + 11", env), 42)
        self.assertEqual(ieval("(33 - 2) + 10 * 2 - 9", env), 42)
        self.assertEqual(ieval("42 / 9", env), 42 / 9)

    def test_binding(self):
        env = {}
        self.assertEqual(ieval("let x = 10", env), None)
        self.assertEqual(ieval("x", env), 10)
        self.assertEqual(ieval("let y = 5 * x", env), None)
        self.assertEqual(ieval("y", env), 50)
        self.assertEqual(ieval("1 + y", env), 51)

    def test_if_else(self):
        env = {}
        self.assertEqual(ieval("if true then 42 else 666 end", env), 42)
        self.assertEqual(ieval("if false then 666 else 42 end", env), 42)
        self.assertEqual(
            ieval("if if true then false else true end then 666 else 42 end", env), 42
        )

    def test_function(self):
        env = {}
        self.assertEqual(ieval("fn f(x, y) = (x + x) * (y + y)", env), None)
        self.assertEqual(ieval("f(5, 3)", env), 60)
        self.assertEqual(ieval("f(f(3 + 2, 3), 3)", env), 720)
        # Make sure that function parameters do not have global scope.
        with self.assertRaises(IExecutionError):
            ieval("x", env)
        with self.assertRaises(IExecutionError):
            ieval("y", env)
        ieval("let x = 42", env)
        ieval("f(1, 1)", env)
        # Make sure function calls don't overwrite local variables with their parameters.
        self.assertEqual(ieval("x", env), 42)
        # Test wrong number of arguments.
        with self.assertRaises(IExecutionError):
            ieval("f(5)", env)
        with self.assertRaises(IExecutionError):
            ieval("f(f(5), 3, 5)", env)


####################
#  USER INTERFACE  #
####################


def repl():
    """Run the read-eval-print loop."""
    env = {}  # type: EnvType
    try:
        while True:
            expr = input(">>> ").strip()
            if expr.startswith("!dis"):
                try:
                    code = icompile(iparse(expr[4:]))
                except IError as e:
                    print("Error:", e)
                else:
                    for inst, arg in code:
                        print(inst, repr(arg))
            else:
                try:
                    res = ieval(expr, env)
                except IError as e:
                    print("Error:", e)
                else:
                    if res is not None:
                        print(res)
    except (KeyboardInterrupt, EOFError):
        print()
        pass


if __name__ == "__main__":
    aparser = argparse.ArgumentParser()
    aparser.add_argument(
        "--test", default=False, action="store_true", help="Run the test suite"
    )
    args = aparser.parse_args()
    if args.test:
        unittest.main(argv=sys.argv[:1])
    else:
        repl()
