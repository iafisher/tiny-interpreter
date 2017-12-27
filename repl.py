#!/usr/bin/env python3
"""A tiny interpreter for a stupid arithmetic language."""
import readline
import re
from collections import namedtuple
from typing import Union, Dict, Optional, List, Tuple, cast


ExprNode = namedtuple('ExprNode', ['value', 'left', 'right'])
DefineNode = namedtuple('DefineNode', ['symbol', 'expr'])

ASTType = Union[ExprNode, DefineNode, int, str]
EnvType = Dict[str, int]
BytecodeType = Tuple[int, Union[int, str, None]]


def parse_expr(expr: str) -> ASTType:
    """Parse the expression string according to this grammar:

    start -> expr | define
    define -> [ IDENT expr ]
    expr -> ( OP expr expr )  |  NUMBER  |  IDENT
    """
    tz = Tokenizer(expr)
    token = tz.require_next()
    if token.kind in ('LEFT_PAREN', 'NUMBER', 'IDENT'):
        return match_expr(tz)
    elif token.kind == 'LEFT_BRACKET':
        return match_define(tz)
    else:
        raise MyParseError('expected "[", "(", number or identifier, got "{}"'.format(token.value))


def match_expr(tz: 'Tokenizer') -> ASTType:
    """Match an expression from the tokenizer, taking the tokenizer on the second token of the
    expression and leaving it one past the last token of the expression.
    """
    if tz.token.kind == 'LEFT_PAREN':
        op = tz.require_next()
        if op.kind != 'OP':
            raise MyParseError('expected operator, got "{}"'.format(op.value))
        tz.require_next()
        left = match_expr(tz)
        tz.require_next()
        right = match_expr(tz)
        right_paren = tz.require_next()
        if right_paren.kind != 'RIGHT_PAREN':
            raise MyParseError('expected ")", got "{}"'.format(right_paren.value))
        return ExprNode(op.value, left, right)
    elif tz.token.kind == 'NUMBER':
        return int(tz.token.value)
    elif tz.token.kind == 'IDENT':
        return tz.token.value
    else:
        raise MyParseError('expected "(" or number, got "{}"'.format(tz.token.value))


def match_define(tz: 'Tokenizer') -> ASTType:
    if tz.token.kind == 'LEFT_BRACKET':
        ident = tz.require_next()
        if ident.kind != 'IDENT':
            raise MyParseError('expected identifier, got "{}"'.format(ident.value))
        # So that the tokenizer will be correctly positioned for match_expr.
        tz.require_next()
        expr = match_expr(tz)
        right_bracket = tz.require_next()
        if right_bracket.kind != 'RIGHT_BRACKET':
            raise MyParseError('expected "]", got "{}"'.format(right_bracket.value))
        return DefineNode(ident.value, expr)
    else:
        raise MyParseError('expected "[" , got "{}"'.format(tz.token.value))


Token = namedtuple('Token', ['kind', 'value'])


class Tokenizer:
    """An iterator over the tokens of an expression string. The current token is available as
    `self.token`. The tokenizer is said to be "on" a particular token if `next(self)` will return
    that token. See the docstring of `match_expr` above.
    """

    tokens = (
        ('LEFT_PAREN', r'\('),
        ('RIGHT_PAREN', r'\)'),
        ('LEFT_BRACKET', r'\['),
        ('RIGHT_BRACKET', r'\]'),
        ('IDENT', r'[A-Za-z_]+'),
        ('OP', r'\+|-|\*'),
        ('NUMBER', r'[0-9]+'),
        ('SKIP', r'\s'),
        ('MISMATCH', r'.'),
    )
    regex = re.compile('|'.join('(?P<%s>%s)' % tok for tok in tokens))

    def __init__(self, expr: str) -> None:
        self.it = self.regex.finditer(expr)
        self.token = None # type: Optional[Token]

    def __iter__(self) -> 'Tokenizer':
        return self

    def __next__(self) -> Token:
        while True:
            mo = next(self.it)
            kind = mo.lastgroup
            val = mo.group(kind)
            if kind == 'MISMATCH':
                raise MyParseError('unexpected character "{}"'.format(val))
            elif kind != 'SKIP':
                break
        self.token = Token(kind, val)
        return self.token

    def require_next(self) -> Token:
        """Same as __next__ except raises a useful exception when the iterator is exhausted."""
        try:
            return next(self) # type: ignore
        except StopIteration:
            raise MyParseError('unexpected end of input') from None

    def __bool__(self) -> bool:
        return bool(self.it)


# Definition of bytecode instruction names.
BINARY_ADD =  0
BINARY_SUB =  1
BINARY_MUL =  2
LOAD_CONST =  3
STORE_NAME =  4
LOAD_NAME  =  5


def compile_ast(ast: ASTType) -> List[BytecodeType]:
    """Compile the AST into a list of bytecode instructions of the form (instruction, arg). arg is
    None if the instruction does not take an argument.
    """
    if isinstance(ast, ExprNode):
        ret = compile_ast(ast.left) + compile_ast(ast.right)
        if ast.value == '+':
            ret.append( (BINARY_ADD, None) )
        elif ast.value == '-':
            ret.append( (BINARY_SUB, None) )
        elif ast.value == '*':
            ret.append( (BINARY_MUL, None) )
        else:
            raise ValueError('unknown AST value "{}"'.format(ast.value))
        return ret
    elif isinstance(ast, DefineNode):
        ret = compile_ast(ast.expr)
        ret.append( (STORE_NAME, ast.symbol) )
        return ret
    elif isinstance(ast, str):
        return [(LOAD_NAME, ast)]
    else:
        return [(LOAD_CONST, ast)]


def execute_code(codeobj: List[BytecodeType], env: EnvType) -> Optional[int]:
    """Execute a code object in the given environment using a virtual stack machine."""
    stack = ExecutionStack()
    for inst, arg in codeobj:
        if inst == LOAD_CONST:
            stack.append(arg)
        elif inst == BINARY_ADD:
            right = stack.pop()
            left = stack.pop()
            stack.append(left + right)
        elif inst == BINARY_SUB:
            right = stack.pop()
            left = stack.pop()
            stack.append(left - right)
        elif inst == BINARY_MUL:
            right = stack.pop()
            left = stack.pop()
            stack.append(left * right)
        elif inst == STORE_NAME:
            env[cast(str, arg)] = stack.pop()
        elif inst == LOAD_NAME:
            try:
                stack.append(env[cast(str, arg)])
            except KeyError:
                raise MyExecutionError('unbound identifier "{}"'.format(arg))
        else:
            raise ValueError('unrecognized bytecode instruction "{}"'.format(inst))
    if stack:
        return stack.pop()
    else:
        return None


class ExecutionStack(list):
    """A stack class identical to Python lists except ExecutionStack.pop raises a useful exception
    when the list is empty instead of the generic Python exception.
    """

    def pop(self, *args, **kwargs):
        try:
            return super().pop(*args, **kwargs)
        except IndexError:
            raise MyExecutionError('pop from empty stack') from None


class MyError(Exception):
    pass

class MyParseError(MyError):
    pass

class MyExecutionError(MyError):
    pass


if __name__ == '__main__':
    # The read-eval-print loop (REPL).
    env = {} # type: EnvType
    try:
        while True:
            expr = input('>>> ')
            try:
                res = execute_code(compile_ast(parse_expr(expr)), env)
            except MyError as e:
                print('Error:', e)
            else:
                if res is not None:
                    print(res)
    except (KeyboardInterrupt, EOFError):
        print()
        pass
