import numpy as np


class Snake:
    points = []
    img = None

    def __init__(self, img):
        self.points = []
        self.img = img
        self.s = None


    def snake_spline(self, points, s):
        # We can represent a snake mathematically. Considering a snake a spline, we can represent points of that spline as V(s).
        # V(s) = (x(s), y(s)) 0 ≤s≤1
        # We can represent the spline as a function of s, where s is the parameter of the spline.
        if self.s is None:
            self.s = np.linspace(0, 1, len(points))
        else:
            self.s = s

    def curvature



