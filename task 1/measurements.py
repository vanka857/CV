from abc import ABC, abstractmethod
from cv_methods import windowed
import numpy as np
import time


class Measurements(ABC):
    def __init__(self, cfa_array, original_array, window):
        self.cfa_array = cfa_array
        self.original_array = original_array

        self.window = window
        if window == None:
            self.window = [0, cfa_array.shape[0], 0, cfa_array.shape[1]]

        self.w = self.window[3] - self.window[2]
        self.h = self.window[1] - self.window[0]

        self.time = None,
        self.res_image = None
        self.PSNR = None

        assert cfa_array.shape == original_array.shape[:-1]
        # print(original_array.shape[:-1])
        
    @abstractmethod
    def process(self):
        assert False

    def process_time(self):
        start_time = time.time()
        self.process()
        self.time = time.time() - start_time

    def Y(self, value):
        r, g, b = value
        return 0.299 * r + 0.587 * g + 0.114 * b

    def calc_PSNR(self):
        original = windowed(self.original_array, self.window)

        y = np.apply_along_axis(self.Y, -1, self.res_image)
        y_ref = np.apply_along_axis(self.Y, -1, original)
        mse = 1 / (self.w * self.h) * np.sum((y - y_ref) ** 2)

        Y_MAX = 255
        self.PSNR = 10 * np.log10(Y_MAX ** 2 / mse)

    def get_image(self):
        return self.res_image

    def measure(self):
        self.process_time()
        self.calc_PSNR()

        mp = self.w * self.h / 10**6
        sec_to_mp = self.time / mp

        print(f'MegaPixels: {mp}, Total time: {self.time:.2f} seconds, Seconds to MegaPixel: {sec_to_mp:.2f}, PSNR: {self.PSNR}')

        return self.res_image, mp, self.time, sec_to_mp, self.PSNR