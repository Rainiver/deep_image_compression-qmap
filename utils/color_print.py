BLACK, RED, GREEN, YELLOW, BLUE, MAGENTA, CYAN, WHITE = range(30, 38)
RESET_SEQ = "\033[0m"
COLOR_SEQ = "\033[1;%dm"


def cprint(*args, **kwargs):
    if 'color' in kwargs:
        color = kwargs['color']
    else:
        color = RED
    print(COLOR_SEQ % color, end='', flush=True)
    print(*args, **kwargs)
    print(RESET_SEQ, end='', flush=True)
