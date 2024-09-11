from torch import nn


class DCTCoefPre(nn.Module):
    def __init__(self, flatten=False, normalize=True, scale_factor=255):
        """Preprocess for DCT coefficients compression.

        Args:
            flatten:
            normalize:
            scale_factor:
        """
        super(DCTCoefPre, self).__init__()
        self.flatten = flatten
        self.normalize = normalize
        self.scale_factor = scale_factor

    def forward(self, coef):
        coef = coef.float()
        if self.flatten:
            raise NotImplementedError
        if self.normalize:
            if self.scale_factor is None:
                self.scale_factor = 255
            # coef = (coef + 128) / self.scale_factor
            coef = coef / self.scale_factor

        return coef
