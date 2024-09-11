import decimal
import math

decimal.getcontext().prec = 500


def encode(bits):
    l = decimal.Decimal('0')
    r = decimal.Decimal('1')
    mid = decimal.Decimal('0.5')
    sum0 = 0
    sum1 = 0
    ret = ''
    for i in range(len(bits)):
        if bits[i] == '0':
            sum0 += 1
        else:
            sum1 += 1
    mid = (r - l) / (sum0 + sum1) * sum0
    for i in range(len(bits)):
        if bits[i] == '0':
            r = mid
            # sum0 += 1
        else:
            l = mid + (r - l) / decimal.Decimal('10000')
            # sum1 += 1
        mid = (r - l) / (sum0 + sum1) * sum0 + l
    length = math.ceil(math.log2(decimal.Decimal('1') / (r - l)))
    for i in range(length):
        mid = mid * decimal.Decimal('2')
        if mid > decimal.Decimal('1'):
            mid -= decimal.Decimal('1')
            ret += '1'
        else:
            ret += '0'
    return ret, (decimal.Decimal('1') - decimal.Decimal('0')) / (sum0 + sum1) * sum0


def decode(bits, mid, length):
    r = decimal.Decimal('1')
    l = decimal.Decimal('0')
    rate = mid
    input = decimal.Decimal('0')
    base = decimal.Decimal('0.5')
    for i in range(len(bits)):
        if bits[i] == '1':
            input += base
        base /= decimal.Decimal('2')
    ret = ''
    for i in range(length):
        if input > mid:
            ret += '1'
            l = mid + (r - l) / decimal.Decimal('10000')
        else:
            ret += '0'
            r = mid
        mid = (r - l) * rate + l
    return ret


if __name__ == '__main__':
    a = "1001"
    for i in range(4):
        a = a + a
    for i in range(0,10000):
        encode(a)
