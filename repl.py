#!/usr/bin/env python3
"""A tiny interpreter for a stupid arithmetic language."""
import readline
import re
from collections import namedtuple
from typing import Dict, Union, List, Tuple


ParseTree = namedtuple('ParseTree', ['value', 'left', 'right'])


def parse_expr(expr: str) -> Union[int, ParseTree]:
    """Parse the expression string according to this grammar:

    start -> expr
    expr -> ( OP expr expr )
    expr -> NUMBER
    """
    tz = Tokenizer(expr)
    return match_expr(tz)


def match_expr(tz: 'Tokenizer') -> Union[int, ParseTree]:
    """Match an expression from the tokenizer, taking the tokenizer on the first token of the
    expression and leaving it one past the last token of the expression.
    """
    token = tz.require_next()
    if token.kind == 'LEFT_PAREN':
        op = tz.require_next()
        if op.kind != 'OP':
            raise MyParseError('expected operator, got "{}"'.format(op.value))
        left = match_expr(tz)
        right = match_expr(tz)
        right_paren = tz.require_next()
        if right_paren.kind != 'RIGHT_PAREN':
            raise MyParseError('expected right parenthesis, got "{}"'.format(op.value))
        return ParseTree(op.value, left, right)
    elif token.kind == 'NUMBER':
        return int(token.value)
    else:
        raise MyParseError('expected left parenthesis or number, got "{}"'.format(token.value))


Token = namedtuple('Token', ['kind', 'value'])


class Tokenizer:
    """An iterator over the tokens of an expression string. The current token is available as
    `self.token`. The tokenizer is said to be "on" a particular token if `next(self)` will return
    that token. See the docstring of `match_expr` above.
    """

    tokens = (
        ('LEFT_PAREN', r'\('),
        ('RIGHT_PAREN', r'\)'),
        ('OP', r'\+|-|\*'),
        ('NUMBER', r'[0-9]+'),
    )
    regex = re.compile('|'.join('(?P<%s>%s)' % tok for tok in tokens))

    def __init__(self, expr):
        self.it = self.regex.finditer(expr)
        self.token = None

    def __iter__(self):
        return self

    def __next__(self):
        mo = next(self.it)
        kind = mo.lastgroup
        val = mo.group(kind)
        self.token = Token(kind, val)
        return self.token

    def require_next(self):
        """Same as __next__ except raises a useful exception when the iterator is exhausted."""
        try:
            return next(self)
        except StopIteration:
            raise MyParseError('unexpected end of input') from None

    def __bool__(self):
        return bool(self.it)


# Definition of bytecode instruction names.
BINARY_ADD = 'BINARY_ADD'
BINARY_SUB = 'BINARY_SUB'
BINARY_MUL = 'BINARY_MUL'
LOAD_CONST = 'LOAD_CONST'


def compile_ast(ast: Union[int, ParseTree]) -> List[Tuple[str, int]]:
    """Compile the AST into a list of bytecode instructions of the form (instruction name, arg). arg
    is None if the instruction does not take an argument.
    """
    if isinstance(ast, ParseTree):
        rest = compile_ast(ast.left) + compile_ast(ast.right)
        if ast.value == '+':
            rest.append( (BINARY_ADD, None) )
        elif ast.value == '-':
            rest.append( (BINARY_SUB, None) )
        elif ast.value == '*':
            rest.append( (BINARY_MUL, None) )
        else:
            raise MyCompileError('unknown AST value "{}"'.format(ast.value))
        return rest
    else:
        return [(LOAD_CONST, ast)]


def execute_code(codeobj: List[Tuple[str, int]], env: Dict[str, int]) -> int:
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
        else:
            raise MyExecutionError('unrecognized bytecode instruction "{}"'.format(inst))
    return stack.pop()


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

class MyCompileError(MyError):
    pass

class MyExecutionError(MyError):
    pass


if __name__ == '__main__':
    # The read-eval-print loop (REPL).
    env = {} # type: Dict[str, int]
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
