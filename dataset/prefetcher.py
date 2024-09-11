import torch
# from torch.utils.data import DataLoader
from .dataloader import DataLoader
import time
try:
    import spring.linklink as link
except:
    link = None

class DataPrefetcher:
    """
    refer: NVIDIA apex

    https://github.com/NVIDIA/apex/blob/f5cd5ae937f168c763985f627bbf850648ea5f3f/examples/imagenet/main_amp.py#L256
    """

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.loader_iterator = iter(loader)
        if device.type == 'cuda':
            # use a low priority
            self.stream = torch.cuda.Stream(priority=100)
            # print('using a cuda stream in pre-fetcher')
        else:
            self.stream = None

        self.next_input = None
        self.poison_pill = None
        self.preload()

    def preload(self):
        if self.stream is None:  # using cpu
            return

        try:
            self.next_input = next(self.loader_iterator)
        # except OSError:
        #     try:
        #         self.next_input = next(self.loader_iterator)
        #     except StopIteration:
        #         # start = time.time()
        #         self.loader_iterator = iter(self.loader)  # reset data loader
        #         # print(f"rank{link.get_rank()} {time.time() - start}", flush=True)
        #         try:
        #             self.next_input = next(self.loader_iterator)
        #         except StopIteration:
        #             raise ValueError('error: empty data loader')
        #         self.poison_pill = object()  # set a poison pill to stop current iteration
        except StopIteration:
            # start = time.time()
            self.loader_iterator = iter(self.loader)  # reset data loader
            # print(f"rank{link.get_rank()} {time.time() - start}", flush=True)
            try:
                self.next_input = next(self.loader_iterator)
            except StopIteration:
                raise ValueError('error: empty data loader')
            self.poison_pill = object()  # set a poison pill to stop current iteration

        with torch.cuda.stream(self.stream):
            # self.next_input = self.next_input['img'].cuda(non_blocking=True)
            for _key in ["img", "quantization", "y_coefficients", "cb_coefficients", "cr_coefficients"]:
                _data = self.next_input.get(_key, None)
                if _data is not None:
                    self.next_input[_key] = _data.cuda(non_blocking=True)

    def __iter__(self):
        if self.stream is None:
            yield from self.loader

        while True:
            torch.cuda.current_stream().wait_stream(self.stream)
            next_input = self.next_input
            self.preload()
            yield next_input
            if self.poison_pill is not None:
                self.poison_pill = None
                break

    def __len__(self):
        return len(self.loader)
