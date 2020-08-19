import numpy as np


class mask_generator(object):
    def __init__(self, band_num, total_size):
        self.band_num = band_num
        self.total_size = total_size
        self.mask_indices = np.arange(total_size - band_num, total_size)

    def ordered_next(self):
        self.mask_indices = (self.mask_indices + self.band_num) % self.total_size
        mask = np.zeros(self.total_size)
        mask[self.mask_indices] = 1
        return mask, self.mask_indices

    def uniform_next(self):
        self.mask_indices = np.random.randint(0, self.total_size, size=self.band_num)
        mask = np.zeros(self.total_size)
        mask[self.mask_indices] = 1
        return mask, self.mask_indices


def get_channel_coef(style, channel_number, h_coef):
    h_m = []
    for i in range(len(style)):
        if style[i] == "error_free":
            h_m.append(np.ones(channel_number))
        else:
            h_m.append(np.random.rayleigh(scale=np.sqrt(2 / np.pi) * h_coef, size=channel_number))

    return h_m
