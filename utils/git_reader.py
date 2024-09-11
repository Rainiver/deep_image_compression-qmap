import subprocess


class GitReader:
    def _cmd(self, cmd):
        try:
            res = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, encoding="utf-8", timeout=30)
        except:
            return f'fail to read git info, cmd: {cmd}'
        if res.returncode == 0:
            return res.stdout
        else:
            return f'git reader err: {res}'

    def branch_info(self):
        return self._cmd('git branch | grep \\*')

    def commit_info(self):
        return self._cmd('git log -n 1')

    def branch_and_commit_info(self):
        res = f'git info:\nbranch: {self.branch_info()}\ncommit:\n{self.commit_info()}'
        return res


def read_git():
    return GitReader().branch_and_commit_info()
