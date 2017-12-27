#!/usr/bin/env python3
"""A tiny Lisp dialect in Python with mypy static types."""
import readline
import re
from collections import namedtuple
from typing import Dict, Union, List, Tuple


ParseTree = namedtuple('ParseTree', ['value', 'left', 'right'])


def parse_lisp(expr: str) -> Union['LispVal', ParseTree]:
    """Parse the expression string according to this grammar:

    start -> expr
    expr -> ( OP expr expr )
    expr -> NUMBER
    """
    tz = Tokenizer(expr)
    return match_expr(tz)


def match_expr(tz: 'Tokenizer') -> Union['LispVal', ParseTree]:
    """Match an expression from the tokenizer, taking the tokenizer on the first token of the
    expression and leaving it one past the last token of the expression.
    """
    token = tz.require_next()
    if token.kind == 'LEFT_PAREN':
        op = tz.require_next()
        if op.kind != 'OP':
            raise LispParseError('expected operator, got "{}"'.format(op.value))
        left = match_expr(tz)
        right = match_expr(tz)
        right_paren = tz.require_next()
        if right_paren.kind != 'RIGHT_PAREN':
            raise LispParseError('expected right parenthesis, got "{}"'.format(op.value))
        return ParseTree(op.value, left, right)
    elif token.kind == 'NUMBER':
        return LispInt(token.value)
    else:
        raise LispParseError('expected left parenthesis or number, got "{}"'.format(token.value))


Token = namedtuple('Token', ['kind', 'value'])


class Tokenizer:
    """An iterator over the tokens of a Lisp string. The current token is available as `self.token`.
    The tokenizer is said to be "on" a particular token if `next(self)` will return that token.
    See the docstring of `match_expr` above.
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
            raise LispParseError('unexpected end of input') from None

    def __bool__(self):
        return bool(self.it)


# Definition of bytecode instruction names.
BINARY_ADD = 'BINARY_ADD'
BINARY_SUB = 'BINARY_SUB'
BINARY_MUL = 'BINARY_MUL'
LOAD_CONST = 'LOAD_CONST'


def compile_lisp(ast: Union['LispVal', ParseTree]) -> List[Tuple[str, 'LispVal']]:
    """Compile the AST into a list of bytecode instructions of the form (instruction name, arg). arg
    is None if the instruction does not take an argument.
    """
    if isinstance(ast, ParseTree):
        rest = compile_lisp(ast.left) + compile_lisp(ast.right)
        if ast.value == '+':
            rest.append( (BINARY_ADD, None) )
        elif ast.value == '-':
            rest.append( (BINARY_SUB, None) )
        elif ast.value == '*':
            rest.append( (BINARY_MUL, None) )
        else:
            raise LispCompileError('unknown AST value "{}"'.format(ast.value))
        return rest
    else:
        return [(LOAD_CONST, ast)]


# A type hierarchy for the Lisp dialect we're implementing. We could just use the native Python
# types list, int and str, but this enforces a conceptual distinction between types in the language
# we're using (Python) versus types in the language we're implementing (a Lisp dialect).

class LispVal:
    def __str__(self):
        raise NotImplementedError

class LispList(LispVal, list):
    pass

class LispInt(LispVal, int):
    pass

class LispStr(LispVal, str):
    pass


def execute_lisp(codeobj: List[Tuple[str, LispVal]], env: Dict[str, LispVal]) -> LispVal:
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
            raise LispExecutionError('unrecognized bytecode instruction "{}"'.format(inst))
    return stack.pop()


class ExecutionStack(list):
    """A stack class identical to Python lists except ExecutionStack.pop raises a useful exception
    when the list is empty instead of the generic Python exception.
    """

    def pop(self, *args, **kwargs):
        try:
            return super().pop(*args, **kwargs)
        except IndexError:
            raise LispExecutionError('pop from empty stack') from None


class LispError(Exception):
    pass

class LispParseError(LispError):
    pass

class LispCompileError(LispError):
    pass

class LispExecutionError(LispError):
    pass


if __name__ == '__main__':
    # The read-eval-print loop (REPL).
    env = {} # type: Dict[str, LispVal]
    try:
        while True:
            expr = input('>>> ')
            try:
                res = execute_lisp(compile_lisp(parse_lisp(expr)), env)
            except LispError as e:
                print('Error:', e)
            else:
                if res is not None:
                    print(res)
    except (KeyboardInterrupt, EOFError):
        print()
        pass
