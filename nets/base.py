class PreparableMixin:
    def prepare(self):
        pass

    def prepare_recursively(self):
        for m in self.modules():
            if m is self:
                continue
            if hasattr(m, 'prepare'):
                m.prepare()
                print(f'prepared {m}')

        self.prepare()
