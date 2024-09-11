import subprocess


def execute(*args):
    arguments = [str(arg) for arg in args]
    process = subprocess.Popen(
        args=arguments,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    process.stdout.flush()
    _stdout, _stderr = process.communicate()
    _return_code = process.wait()

    _stdout, _stderr = _stdout.decode(), _stderr.decode()
    return _return_code, _stdout, _stderr


TEST_COMMAND = ["cat", "/proc/version"]


def test_execute():
    _return_code, _stdout, _stderr = execute(*TEST_COMMAND)
    print("return code :", _return_code)
    print()

    print("stdout ({length} byte(s)):".format(length=len(_stdout)))
    print(_stdout)
    print()

    print("stderr ({length} byte(s)):".format(length=len(_stderr)))
    print(_stderr)
    print()

    assert _return_code == 0, _return_code
    assert _stdout, _stdout
    assert not _stderr, _stderr


if __name__ == "__main__":
    test_execute()
