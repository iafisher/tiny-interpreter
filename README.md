A tiny interpreter for a simple programming language. Includes a parser, bytecode compiler, and virtual machine, as well as a full test suite, with no dependencies other than Python 3 and the Python standard library.

```
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
```

The entire interpreter is contained in a single executable Python file, `tiny.py`.
