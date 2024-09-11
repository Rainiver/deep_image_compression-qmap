import time

times = []
temp = []
names = []


def mark(name, new_pipeline=False):
    global temp, times, names
    x = time.time()
    if new_pipeline and len(temp) > 0:
        times.append(temp)
        temp = [x]
        names = []
    else:
        if not new_pipeline:
            names.append(name)
        temp.append(x)


def clear():
    global times, temp, names
    times = []
    temp = []
    names = []


def output():
    global temp, times, names
    times.append(temp)
    for i in range(len(times)):
        for j in range(len(times[i]) - 1):
            times[i][j] = times[i][j + 1] - times[i][j]
    tot = 0.
    if len(times[0]) != len(names) + 1:
        print(len(times[0]), len(names))
        assert 0

    # (hedailan 20210615)
    # when testing the speed, drop the first `pp` images
    # considering device warmup and other initialization overhead
    pp = 6 if len(times) > 12 else 0

    for i in range(len(times[0]) - 1):
        t = 0.
        for j in range(pp, len(times)):
            t += times[j][i]
        t /= (len(times) - pp)
        tot += t
        print(format(t, '.15f'), " :\t" + names[i])
    print("total time per image: \t", tot)
