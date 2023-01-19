import numpy as np


class ExponentialAverageCalculator:
    def __init__(self, alpha:float, statistics_accumulation_start:int=0):
        self.alpha = alpha
        self.y_t = 0
        self.statistics_accumulation_start = statistics_accumulation_start

    def __call__(self, x_t:float, t:int, learn:bool=False):
        if t >= self.statistics_accumulation_start:
            if learn:
                w_t = self.alpha * (1 - self.alpha**t) / (1 - self.alpha ** (t + 1))
                self.y_t = w_t * self.y_t + (1 - w_t) * x_t
        return self.y_t

    def load(self, y_t):
        self.y_t = y_t


class RunningStd:
    def __init__(self, alpha:float=0.999, statistics_accumulation_start:int=2):
        self.statistics_accumulation_start = statistics_accumulation_start
        self.alpha = alpha
        self.y_t = 0
        self.Ex = self.Ex2 = 0.0

    def __call__(self, x_t:float, average:float, t:int, learn:bool=False):
        if t >= self.statistics_accumulation_start:
            if learn:
                self.Ex += x_t - average
                self.Ex2 += (x_t - average) ** 2
                variance = (self.Ex2 - self.Ex**2 / t) / (t - 1)
                w_t = self.alpha * (1 - self.alpha**t) / (1 - self.alpha ** (t + 1))
                self.y_t = w_t * self.y_t + (1 - w_t) * variance
            return np.sqrt(self.y_t)
        else:
            return np.inf if self.y_t == 0 else np.sqrt(self.y_t)

    def load(self, y_t, Ex, Ex2):
        self.y_t = y_t
        self.Ex = Ex
        self.Ex2 = Ex2