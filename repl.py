#!/usr/bin/env python3
"""
A tiny interpreter for a simple programming language. Includes a parser, bytecode
compiler, and virtual machine, as well as a full test suite, with no dependencies
other than Python 3 and the standard library.

Example code:

    >>> let x = 32
    >>> x
    32
    >>> (+ x 10)
    42
    >>> fn add x y = (+ x y)
    >>> (add x 10)
    42
    >>> if true then 42 else 666 end
    42

Author:  Ian Fisher (iafisher@protonmail.com)
Version: May 2019
"""
import readline
import re
import unittest
import argparse
import sys
from collections import namedtuple, ChainMap
from typing import Union, Dict, Optional, List, Tuple, cast, Any


###################
#  PARSING STAGE  #
###################

# Node types returned by the parser.
OpNode = namedtuple("OpNode", ["value", "left", "right"])
CallNode = namedtuple("CallNode", ["name", "args"])
DefineNode = namedtuple("DefineNode", ["symbol", "expr"])
IfNode = namedtuple("IfNode", ["condition", "true_clause", "false_clause"])


class Function(namedtuple("Function", ["name", "parameters", "code"])):
    """
    Internal representation of functions. The `name` field is only used for nicer error
    messages.
    """

    def __str__(self):
        return '<function "{0.name}">'.format(self)


# Type declarations for mypy.
ASTType = Union[OpNode, CallNode, DefineNode, Function, int, str]
EnvType = Union[Dict[str, int], ChainMap]
BytecodeType = Tuple[str, Union[Function, int, str]]


def parse_expr(expr: str) -> ASTType:
    """Parse the expression string according to this grammar:

    start     := expr | define
    define    := LET IDENT EQ expr
    function  := FN IDENT+ EQ expr
    if        := IF expr THEN expr ELSE expr END
    expr      := ( OP expr expr )  |  ( IDENT expr* )  | NUMBER | TRUE | FALSE | IDENT
    """
    tz = Tokenizer(expr)
    token = tz.advance()
    if token.kind in EXPR_FIRST:
        ret = match_expr(tz)
    elif token.kind == "LET":
        ret = match_define(tz)
    elif token.kind == "FN":
        ret = match_function_decl(tz)
    else:
        raise MyParseError(
            'expected "(", number or identifier, got "{}"'.format(token.value)
        )

    try:
        tz.advance()
    except MyParseError:
        return ret
    else:
        raise MyParseError("trailing input")


# The FIRST set of the expr rule: the set of all tokens that could start an expr
# production.
EXPR_FIRST = frozenset(["LEFT_PAREN", "NUMBER", "IDENT", "IF", "TRUE", "FALSE"])


def match_expr(tz: "Tokenizer") -> ASTType:
    """Match an expression."""
    if tz.token.kind == "LEFT_PAREN":
        op = tz.advance()
        if op.kind == "OP":
            tz.advance()
            left = match_expr(tz)
            tz.advance()
            right = match_expr(tz)
            tz.advance()
            if tz.token.kind != "RIGHT_PAREN":
                raise MyParseError('expected ")", got "{}"'.format(tz.token.value))
            return OpNode(op.value, left, right)
        elif op.kind == "IDENT":
            tz.advance()
            args = match_expr_star(tz)
            if tz.token.kind != "RIGHT_PAREN":
                raise MyParseError('expected ")", got "{}"'.format(tz.token.value))
            return CallNode(op.value, args)
        else:
            raise MyParseError('expected operator, got "{}"'.format(op.value))
    elif tz.token.kind == "IF":
        tz.advance()
        cond = match_expr(tz)
        tz.advance()
        if tz.token.kind != "THEN":
            raise MyParseError('expected "then", got "{}"'.format(tz.token.value))
        tz.advance()
        true_clause = match_expr(tz)
        tz.advance()
        if tz.token.kind != "ELSE":
            raise MyParseError('expected "else", got "{}"'.format(tz.token.value))
        tz.advance()
        false_clause = match_expr(tz)
        tz.advance()
        if tz.token.kind != "END":
            raise MyParseError('expected "end", got "{}"'.format(tz.token.value))
        return IfNode(cond, true_clause, false_clause)
    elif tz.token.kind == "NUMBER":
        ret = tz.token.value
        return int(ret)
    elif tz.token.kind == "TRUE":
        return True
    elif tz.token.kind == "FALSE":
        return False
    elif tz.token.kind == "IDENT":
        ret = tz.token.value
        return ret
    else:
        raise MyParseError('expected "(" or number, got "{}"'.format(tz.token.value))


def match_expr_star(tz: "Tokenizer") -> List[ASTType]:
    """Match zero or more expressions."""
    ret = []
    while tz.token.kind in EXPR_FIRST:
        ret.append(match_expr(tz))
        tz.advance()
    return ret


def match_define(tz: "Tokenizer") -> ASTType:
    """Match a define statement."""
    ident = tz.advance()
    if ident.kind != "IDENT":
        raise MyParseError('expected identifier, got "{}"'.format(ident.value))
    eq = tz.advance()
    if eq.kind != "EQ":
        raise MyParseError('expected "=", got "{}"'.format(eq.value))
    # So that the tokenizer will be correctly positioned for match_expr.
    tz.advance()
    expr = match_expr(tz)
    return DefineNode(ident.value, expr)


def match_function_decl(tz: "Tokenizer") -> ASTType:
    """Match a function declaration."""
    name = tz.advance()
    if name.kind != "IDENT":
        raise MyParseError('expected identifier, got "{}"'.format(name.value))
    parameters = []
    param = tz.advance()
    while param.kind == "IDENT":
        parameters.append(param.value)
        param = tz.advance()
    if param.kind != "EQ":
        raise MyParseError('expected "=", got "{}"'.format(param.value))
    # So that the tokenizer will be correctly positioned for match_expr.
    tz.advance()
    expr = match_expr(tz)
    code = compile_ast(expr)
    return DefineNode(name.value, Function(name.value, parameters, code))


Token = namedtuple("Token", ["kind", "value"])


class Tokenizer:
    """A class to tokenize strings in the toy language."""

    tokens = (
        ("LEFT_PAREN", r"\("),
        ("RIGHT_PAREN", r"\)"),
        ("LEFT_BRACKET", r"\["),
        ("RIGHT_BRACKET", r"\]"),
        ("IDENT", r"[A-Za-z_]+"),
        ("EQ", r"="),
        ("OP", r"\+|-|\*"),
        ("NUMBER", r"[0-9]+"),
        ("SKIP", r"\s"),
        ("MISMATCH", r"."),
    )
    keywords = frozenset(["let", "fn", "if", "then", "else", "end", "true", "false"])
    regex = re.compile("|".join("(?P<%s>%s)" % tok for tok in tokens))

    def __init__(self, expr: str) -> None:
        self.it = self.regex.finditer(expr)
        self.token = None  # type: Optional[Token]

    def advance(self) -> Token:
        try:
            while True:
                mo = next(self.it)
                kind = mo.lastgroup
                val = mo.group(kind)
                if kind == "IDENT" and val in self.keywords:
                    kind = val.upper()

                if kind == "MISMATCH":
                    raise MyParseError('unexpected character "{}"'.format(val))
                elif kind != "SKIP":
                    break
        except StopIteration:
            raise MyParseError("unexpected end of input")
        else:
            self.token = Token(kind, val)
            return self.token


#######################
#  COMPILATION STAGE  #
#######################

# Definition of bytecode instruction names.
BINARY_ADD = "BINARY_ADD"
BINARY_SUB = "BINARY_SUB"
BINARY_MUL = "BINARY_MUL"
LOAD_CONST = "LOAD_CONST"
STORE_NAME = "STORE_NAME"
LOAD_NAME = "LOAD_NAME"
CALL_FUNCTION = "CALL_FUNCTION"
POP_JUMP_IF_FALSE = "POP_JUMP_IF_FALSE"
JUMP_FORWARD = "JUMP_FORWARD"


# Placeholder value for instructions without arguments.
NO_ARG = 0


def compile_ast(ast: ASTType) -> List[BytecodeType]:
    """
    Compile the AST into a list of bytecode instructions of the form (instruction, arg).
    """
    if isinstance(ast, OpNode):
        ret = compile_ast(ast.left) + compile_ast(ast.right)
        if ast.value == "+":
            ret.append((BINARY_ADD, NO_ARG))
        elif ast.value == "-":
            ret.append((BINARY_SUB, NO_ARG))
        elif ast.value == "*":
            ret.append((BINARY_MUL, NO_ARG))
        else:
            raise ValueError('unknown AST value "{}"'.format(ast.value))
        return ret
    elif isinstance(ast, IfNode):
        condition_code = compile_ast(ast.condition)
        true_code = compile_ast(ast.true_clause)
        false_code = compile_ast(ast.false_clause)

        true_code.append((JUMP_FORWARD, len(false_code) + 1))
        condition_code.append((POP_JUMP_IF_FALSE, len(true_code) + 1))
        return condition_code + true_code + false_code
    elif isinstance(ast, CallNode):
        ret = []
        for arg in reversed(ast.args):
            ret += compile_ast(arg)
        ret.append((LOAD_NAME, ast.name))
        ret.append((CALL_FUNCTION, len(ast.args)))
        return ret
    elif isinstance(ast, DefineNode):
        ret = compile_ast(ast.expr)
        ret.append((STORE_NAME, ast.symbol))
        return ret
    elif isinstance(ast, str):
        return [(LOAD_NAME, ast)]
    elif isinstance(ast, (Function, int)):
        return [(LOAD_CONST, ast)]
    else:
        raise ValueError(
            'don\'t know how to compile object of type "{}"'.format(type(ast))
        )


#####################
#  EXECUTION STAGE  #
#####################


def execute_code(codeobj: List[BytecodeType], env: EnvType) -> Optional[int]:
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
        elif inst == STORE_NAME:
            env[cast(str, arg)] = stack.pop()
            pc += 1
        elif inst == LOAD_NAME:
            try:
                stack.append(env[cast(str, arg)])
            except KeyError:
                raise MyExecutionError('unbound identifier "{}"'.format(arg))
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
                    raise MyExecutionError(msg)
                for param in function.parameters:
                    val = stack.pop()
                    new_env[param] = val
                res = execute_code(function.code, new_env)
                if res is not None:
                    stack.append(res)
            else:
                raise MyExecutionError("first value of expression must be function")
            pc += 1
        else:
            raise ValueError('unrecognized bytecode instruction "{}"'.format(inst))
    if stack:
        return stack.pop()
    else:
        return None


# This error hierarchy allows me to distinguish between errors in my code, signalled by
# regular Python exceptions, and errors in the code being interpreted, signalled by
# MyError exceptions.


class MyError(Exception):
    pass


class MyParseError(MyError):
    pass


class MyExecutionError(MyError):
    pass


def execute_expr(expr: str, env: EnvType) -> Optional[int]:
    """A shortcut function to parse, compile and execute an expression."""
    return execute_code(compile_ast(parse_expr(expr)), env)


################
#  TEST SUITE  #
################


class ParseTests(unittest.TestCase):
    def test_expr(self):
        self.assertEqual(parse_expr("1"), 1)
        self.assertEqual(parse_expr("x"), "x")
        self.assertEqual(parse_expr("(+ 8 2)"), OpNode("+", 8, 2))
        self.assertEqual(
            parse_expr("(+ (* 4 2) (- 3 1))"),
            OpNode("+", OpNode("*", 4, 2), OpNode("-", 3, 1)),
        )
        self.assertEqual(parse_expr("(f 1 2 3 4)"), CallNode("f", [1, 2, 3, 4]))
        self.assertEqual(
            parse_expr("(f 1 (+ 1 1) 3 4)"), CallNode("f", [1, OpNode("+", 1, 1), 3, 4])
        )

    def test_define(self):
        self.assertEqual(parse_expr("let x = 10"), DefineNode("x", 10))
        self.assertEqual(
            parse_expr("let x = (* 5 2)"), DefineNode("x", OpNode("*", 5, 2))
        )
        self.assertEqual(
            parse_expr("let x = (* 5 (+ 1 1))"),
            DefineNode("x", OpNode("*", 5, OpNode("+", 1, 1))),
        )

    def test_if_else(self):
        self.assertEqual(parse_expr("if x then 42 else 666 end"), IfNode("x", 42, 666))
        self.assertEqual(
            parse_expr("if x then if y then 42 else 0 end else 666 end"),
            IfNode("x", IfNode("y", 42, 0), 666),
        )

    def test_function(self):
        self.assertEqual(
            parse_expr("fn f x = 10"),
            DefineNode("f", Function("f", ["x"], [(LOAD_CONST, 10)])),
        )

    def test_errors(self):
        # Trailing input is not allowed.
        with self.assertRaises(MyParseError):
            parse_expr("1 1")
        with self.assertRaises(MyParseError):
            parse_expr("(+ 1 2) 1")
        with self.assertRaises(MyParseError):
            parse_expr("let x = 10 11")
        with self.assertRaises(MyParseError):
            parse_expr("let x = (* 2 10) 11")
        with self.assertRaises(MyParseError):
            parse_expr("fn f x = 10 11")
        with self.assertRaises(MyParseError):
            parse_expr("fn f x = (* 2 10) 11")


class ExecTests(unittest.TestCase):
    def test_atoms(self):
        env = {}
        self.assertEqual(execute_expr("42", env), 42)
        self.assertEqual(execute_expr("true", env), True)
        self.assertEqual(execute_expr("false", env), False)

    def test_arithmetic(self):
        env = {}
        self.assertEqual(execute_expr("(+ 1 1)", env), 2)
        self.assertEqual(execute_expr("(+ 31 11)", env), 42)
        self.assertEqual(execute_expr("(+ (- 33 2) 11)", env), 42)
        self.assertEqual(execute_expr("(+ (- 33 2) (- (* 10 2) 9))", env), 42)

    def test_binding(self):
        env = {}
        self.assertEqual(execute_expr("let x = 10", env), None)
        self.assertEqual(execute_expr("x", env), 10)
        self.assertEqual(execute_expr("let y = (* 5 x)", env), None)
        self.assertEqual(execute_expr("y", env), 50)
        self.assertEqual(execute_expr("(+ 1 y)", env), 51)

    def test_if_else(self):
        env = {}
        self.assertEqual(execute_expr("if true then 42 else 666 end", env), 42)
        self.assertEqual(execute_expr("if false then 666 else 42 end", env), 42)
        self.assertEqual(
            execute_expr(
                "if if true then false else true end then 666 else 42 end", env
            ),
            42,
        )

    def test_function(self):
        env = {}
        self.assertEqual(execute_expr("fn f x y = (* (+ x x) (+ y y))", env), None)
        self.assertEqual(execute_expr("(f 5 3)", env), 60)
        self.assertEqual(execute_expr("(f (f (+ 3 2) 3) 3)", env), 720)
        # Make sure that function parameters do not have global scope.
        with self.assertRaises(MyExecutionError):
            execute_expr("x", env)
        with self.assertRaises(MyExecutionError):
            execute_expr("y", env)
        execute_expr("let x = 42", env)
        execute_expr("(f 1 1)", env)
        # Make sure function calls don't overwrite local variables with their parameters.
        self.assertEqual(execute_expr("x", env), 42)
        # Test wrong number of arguments.
        with self.assertRaises(MyExecutionError):
            execute_expr("(f 5)", env)
        with self.assertRaises(MyExecutionError):
            execute_expr("(f (f 5) 3 5)", env)


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
                    code = compile_ast(parse_expr(expr[4:]))
                except MyError as e:
                    print("Error:", e)
                else:
                    for inst, arg in code:
                        print(inst, repr(arg))
            else:
                try:
                    res = execute_expr(expr, env)
                except MyError as e:
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
