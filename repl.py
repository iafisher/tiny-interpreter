#!/usr/bin/env python3
"""A tiny interpreter for a stupid arithmetic language."""
import readline
import re
import unittest
import argparse
import sys
from collections import namedtuple, ChainMap
from typing import Union, Dict, Optional, List, Tuple, cast


###################
#  PARSING STAGE  #
###################

# Node types returned by the parser.
OpNode = namedtuple('OpNode', ['value', 'left', 'right'])
CallNode = namedtuple('CallNode', ['name', 'args'])
DefineNode = namedtuple('DefineNode', ['symbol', 'expr'])

class Function(namedtuple('Function', ['name', 'parameters', 'code'])):
    """Internal representation of functions. The `name` field is only used for nicer error messages.
    """
    def __str__(self):
        return '<function "{0.name}">'.format(self)

# Type declarations for mypy.
ASTType = Union[OpNode, CallNode, DefineNode, Function, int, str]
EnvType = Union[Dict[str, int], ChainMap]
BytecodeType = Tuple[str, Union[Function, int, str]]


def parse_expr(expr: str) -> ASTType:
    """Parse the expression string according to this grammar:

    start -> expr | define
    define -> LET IDENT EQ expr
    function -> FUNCTION IDENT+ EQ EXPR
    expr -> ( OP expr expr )  |  ( IDENT expr* )  |  NUMBER  |  IDENT
    """
    tz = Tokenizer(expr)
    token = tz.require_next()
    if token.kind in EXPR_FIRST:
        return match_expr(tz)
    elif token.kind == 'LET':
        return match_define(tz)
    elif token.kind == 'FUNCTION':
        return match_function_decl(tz)
    else:
        raise MyParseError('expected "[", "(", number or identifier, got "{}"'.format(token.value))


# The FIRST set of the expr rule: the set of all tokens that could start an expr production.
EXPR_FIRST = frozenset(['LEFT_PAREN', 'NUMBER', 'IDENT'])

def match_expr(tz: 'Tokenizer') -> ASTType:
    """Match an expression."""
    if tz.token.kind == 'LEFT_PAREN':
        op = tz.require_next()
        if op.kind == 'OP':
            tz.require_next()
            left = match_expr(tz)
            right = match_expr(tz)
            if tz.token.kind != 'RIGHT_PAREN':
                raise MyParseError('expected ")", got "{}"'.format(tz.token.value))
            tz.optional_next()
            return OpNode(op.value, left, right)
        elif op.kind == 'IDENT':
            tz.require_next()
            args = match_expr_star(tz)
            if tz.token.kind != 'RIGHT_PAREN':
                raise MyParseError('expected ")", got "{}"'.format(tz.token.value))
            tz.optional_next()
            return CallNode(op.value, args)
        else:
            raise MyParseError('expected operator, got "{}"'.format(op.value))
    elif tz.token.kind == 'NUMBER':
        ret = tz.token.value
        tz.optional_next()
        return int(ret)
    elif tz.token.kind == 'IDENT':
        ret = tz.token.value
        tz.optional_next()
        return ret
    else:
        raise MyParseError('expected "(" or number, got "{}"'.format(tz.token.value))


def match_expr_star(tz: 'Tokenizer') -> List[ASTType]:
    """Match zero or more expressions."""
    ret = []
    while tz.token.kind in EXPR_FIRST:
        ret.append(match_expr(tz))
    return ret


def match_define(tz: 'Tokenizer') -> ASTType:
    """Match a define statement."""
    ident = tz.require_next()
    if ident.kind != 'IDENT':
        raise MyParseError('expected identifier, got "{}"'.format(ident.value))
    eq = tz.require_next()
    if eq.kind != 'EQ':
        raise MyParseError('expected "=", got "{}"'.format(eq.value))
    # So that the tokenizer will be correctly positioned for match_expr.
    tz.require_next()
    expr = match_expr(tz)
    return DefineNode(ident.value, expr)


def match_function_decl(tz: 'Tokenizer') -> ASTType:
    """Match a function declaration."""
    name = tz.require_next()
    if name.kind != 'IDENT':
        raise MyParseError('expected identifier, got "{}"'.format(name.value))
    parameters = []
    param = tz.require_next()
    while param.kind == 'IDENT':
        parameters.append(param.value)
        param = tz.require_next()
    if param.kind != 'EQ':
        raise MyParseError('expected "=", got "{}"'.format(param.value))
    # So that the tokenizer will be correctly positioned for match_expr.
    tz.require_next()
    expr = match_expr(tz)
    code = compile_ast(expr)
    return DefineNode(name.value, Function(name.value, parameters, code))


Token = namedtuple('Token', ['kind', 'value'])


class Tokenizer:
    """An iterator over the tokens of an expression string. After iteration has started, the last
    token returned by __next__ is available as `self.token`.
    """

    tokens = (
        ('LEFT_PAREN', r'\('),
        ('RIGHT_PAREN', r'\)'),
        ('LEFT_BRACKET', r'\['),
        ('RIGHT_BRACKET', r'\]'),
        # Keywords must come before IDENT or the latter will override the former.
        ('LET', r'let'),
        ('FUNCTION', r'function'),
        ('IDENT', r'[A-Za-z_]+'),
        ('EQ', r'='),
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

    def optional_next(self) -> Optional[Token]:
        """Same as __next__ except None is returned on StopIteration."""
        try:
            return next(self) # type: ignore
        except StopIteration:
            return None

    def __bool__(self) -> bool:
        return bool(self.it)


#######################
#  COMPILATION STAGE  #
#######################

# Definition of bytecode instruction names.
BINARY_ADD    = 'BINARY_ADD'
BINARY_SUB    = 'BINARY_SUB'
BINARY_MUL    = 'BINARY_MUL'
LOAD_CONST    = 'LOAD_CONST'
STORE_NAME    = 'STORE_NAME'
LOAD_NAME     = 'LOAD_NAME'
CALL_FUNCTION = 'CALL_FUNCTION'


# Placeholder value for instructions without arguments.
NO_ARG = 0


def compile_ast(ast: ASTType) -> List[BytecodeType]:
    """Compile the AST into a list of bytecode instructions of the form (instruction, arg)."""
    if isinstance(ast, OpNode):
        ret = compile_ast(ast.left) + compile_ast(ast.right)
        if ast.value == '+':
            ret.append( (BINARY_ADD, NO_ARG) )
        elif ast.value == '-':
            ret.append( (BINARY_SUB, NO_ARG) )
        elif ast.value == '*':
            ret.append( (BINARY_MUL, NO_ARG) )
        else:
            raise ValueError('unknown AST value "{}"'.format(ast.value))
        return ret
    elif isinstance(ast, CallNode):
        ret = []
        for arg in reversed(ast.args):
            ret += compile_ast(arg)
        ret.append( (LOAD_NAME, ast.name) )
        ret.append( (CALL_FUNCTION, len(ast.args)) )
        return ret
    elif isinstance(ast, DefineNode):
        ret = compile_ast(ast.expr)
        ret.append( (STORE_NAME, ast.symbol) )
        return ret
    elif isinstance(ast, str):
        return [(LOAD_NAME, ast)]
    else:
        return [(LOAD_CONST, ast)]


#####################
#  EXECUTION STAGE  #
#####################

def execute_code(codeobj: List[BytecodeType], env: EnvType) -> Optional[int]:
    """Execute a code object in the given environment."""
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
        elif inst == CALL_FUNCTION:
            new_env = ChainMap({}, env)
            function = stack.pop()
            if isinstance(function, Function):
                # Make sure enough arguments were passed to the function originally.
                if len(function.parameters) != arg:
                    msg = 'wrong number of arguments to function "{}"'.format(function.name)
                    raise MyExecutionError(msg)
                for param in function.parameters:
                    val = stack.pop()
                    new_env[param] = val
                res = execute_code(function.code, new_env)
                if res is not None:
                    stack.append(res)
            else:
                raise MyExecutionError('first value of expression must be function')
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


# This error hierarchy allows me to distinguish between errors in my code, signalled by regular
# Python exceptions, and errors in the code being interpreted, signalled by MyError exceptions.

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
        self.assertEqual(parse_expr('1'), 1)
        self.assertEqual(parse_expr('x'), 'x')
        self.assertEqual(parse_expr('(+ 8 2)'), OpNode('+', 8, 2))
        self.assertEqual(parse_expr('(+ (* 4 2) (- 3 1))'), OpNode('+', OpNode('*', 4, 2),
                                                                        OpNode('-', 3, 1)))
        self.assertEqual(parse_expr('(f 1 2 3 4)'), CallNode('f', [1, 2, 3, 4]))
        self.assertEqual(parse_expr('(f 1 (+ 1 1) 3 4)'), 
                         CallNode('f', [1, OpNode('+', 1, 1), 3, 4]))

    def test_define(self):
        self.assertEqual(parse_expr('let x = 10'), DefineNode('x', 10))
        self.assertEqual(parse_expr('let x = (* 5 2)'), DefineNode('x', OpNode('*', 5, 2)))
        self.assertEqual(parse_expr('let x = (* 5 (+ 1 1))'), 
                         DefineNode('x', OpNode('*', 5, OpNode('+', 1, 1))))

    def test_function(self):
        self.assertEqual(parse_expr('function f x = 10'), 
                         DefineNode('f', Function('f', ['x'], [(LOAD_CONST, 10)])))


class ExecTests(unittest.TestCase):
    def test_arithmetic(self):
        env = {}
        self.assertEqual(execute_expr('(+ 1 1)', env), 2)
        self.assertEqual(execute_expr('(+ 31 11)', env), 42)
        self.assertEqual(execute_expr('(+ (- 33 2) 11)', env), 42)
        self.assertEqual(execute_expr('(+ (- 33 2) (- (* 10 2) 9))', env), 42)

    def test_binding(self):
        env = {}
        self.assertEqual(execute_expr('let x = 10', env), None)
        self.assertEqual(execute_expr('x', env), 10)
        self.assertEqual(execute_expr('let y = (* 5 x)', env), None)
        self.assertEqual(execute_expr('y', env), 50)
        self.assertEqual(execute_expr('(+ 1 y)', env), 51)

    def test_function(self):
        env = {}
        self.assertEqual(execute_expr('function f x y = (* (+ x x) (+ y y))', env), None)
        self.assertEqual(execute_expr('(f 5 3)', env), 60)
        self.assertEqual(execute_expr('(f (f (+ 3 2) 3) 3)', env), 720)
        # Make sure that function parameters do not have global scope.
        with self.assertRaises(MyExecutionError):
            execute_expr('x', env)
        with self.assertRaises(MyExecutionError):
            execute_expr('y', env)
        execute_expr('let x = 42', env)
        execute_expr('(f 1 1)', env)
        # Make sure function calls don't overwrite local variables with their parameters.
        self.assertEqual(execute_expr('x', env), 42)
        # Test wrong number of arguments.
        with self.assertRaises(MyExecutionError):
            execute_expr('(f 5)', env)
        with self.assertRaises(MyExecutionError):
            execute_expr('(f (f 5) 3 5)', env)


####################
#  USER INTERFACE  #
####################

if __name__ == '__main__':
    aparser = argparse.ArgumentParser()
    aparser.add_argument('--test', default=False, action='store_true', help='Run the test suite')
    args = aparser.parse_args()
    if args.test:
        unittest.main(argv=sys.argv[:1])
    else:
        env = {} # type: EnvType
        try:
            while True:
                expr = input('>>> ').strip()
                if expr.startswith('!dis'):
                    try:
                        code = compile_ast(parse_expr(expr[4:]))
                    except MyError as e:
                        print('Error:', e)
                    else:
                        for inst, arg in code:
                            print(inst, repr(arg))
                else:
                    try:
                        res = execute_expr(expr, env)
                    except MyError as e:
                        print('Error:', e)
                    else:
                        if res is not None:
                            print(res)
        except (KeyboardInterrupt, EOFError):
            print()
            pass
