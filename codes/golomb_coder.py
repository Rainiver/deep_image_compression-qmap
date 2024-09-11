from math import log2, ceil, floor
from typing import List


def _unary(n: int, zero: str, one: str) -> str:
    return one*n + zero

def unary(n: int) -> str:
    """Return string representing given number in unary.
        n:int, number to convert to unary.
    """
    return _unary(n, "0", "1")

def unary_decoding(s: str) -> int:
    return len(s) - 1

def inverted_unary(n: int) -> str:
    """Return string representing given number in inverted unary.
        n:int, number to convert to unary.
    """
    return _unary(n, "1", "0")

def minimal_binary_coding(n: int, b: int) -> str:
    """Return string representing given number in minimal binary coding.
        n:int, number to convert to minimal binary encoding.
        b:int, maximal size.
    """
    if n == 0 and b == 1:
        return ""
    s = ceil(log2(b))
    if n < 2**s - b:
        return f"{{0:0{s-1}b}}".format(n)
    return f"{{0:0{s}b}}".format(n-b+2**s)

def minimal_binary_decoding(s: str, b: int) -> int:
    if s == "":
        return 0
    d = int(s, 2)
    m = ceil(log2(b))
    if d < 2**m -b:
        return d
    return d+b-2**m

def golomb_coding(n: int, b: int) -> str:
    """Return string representing given number in golomb coding.
        n:int, number to convert to golomb coding.
        b:int, module.
        ref: 
        https://en.wikipedia.org/wiki/Golomb_coding 
        (Golomb–Rice codes)
    """
    q = floor((n - 1) / b)
    r = n - q*b - 1
    return unary(q) + minimal_binary_coding(r, b)

def golomb_decoding(s: str, b:int):
    q, r = s.split('0', 1)
    q += '0'
    return minimal_binary_decoding(r, b) + 1 + b*unary_decoding(q)


if __name__ == '__main__':
    print("unary coding")
    print(unary(5)) #111110
    print(inverted_unary(5)) #000001

    print("unary decoding")
    print(unary_decoding("111110")) # 5


    print("minimal binary coding")
    print(minimal_binary_coding(0, 1)) # ""
    print(minimal_binary_coding(0, 2)) # 0
    print(minimal_binary_coding(1, 2)) # 1
    print(minimal_binary_coding(2, 5)) # 10
    print(minimal_binary_coding(3, 5)) # 110
    print(minimal_binary_coding(4, 5)) # 111

    print("minimal binary decoding")
    print(minimal_binary_decoding("110", 5)) # 3
    print(minimal_binary_decoding("10", 5)) # 2
    print(minimal_binary_decoding("1", 2)) # 1
    print(minimal_binary_decoding("", 2)) # 0

    print("golomb coding")
    print(golomb_coding(1,1)) # 0
    print(golomb_coding(3,1)) # 110
    print(golomb_coding(1, 4)) # 0 00
    print(golomb_coding(2, 4)) # 0 01
    print(golomb_coding(3, 4)) # 0 10
    print(golomb_coding(4, 4)) # 0 11
    print(golomb_coding(5, 4)) # 10 00
    print(golomb_coding(6, 4)) # 10 01
    print(golomb_coding(7, 4)) # 10 10
    print(golomb_coding(8, 4)) # 10 11
    print(golomb_coding(9, 4)) # 110 00

    print("golomb decoding")
    print(golomb_decoding("110", 1)) # 3
    print(golomb_decoding("000", 4)) # 1
    print(golomb_decoding("001", 4)) # 2
    print(golomb_decoding("1001", 4)) # 6
    print(golomb_decoding("1011", 4)) # 8
    print(golomb_decoding("11000", 4)) # 9   