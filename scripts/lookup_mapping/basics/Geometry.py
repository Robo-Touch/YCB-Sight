class Circle:
    def __init__(self, cx, cy, r):
        self.center = [cx,cy]
        self.radius = r

class Square:
    def __init__(self, cx, cy, hx, hy):
        self.center = [cx,cy]
        self.half_x = hx
        self.half_y = hy

class Pyramid:
    def __init__(self, cx, cy, h, l):
        self.center = [cx,cy]
        self.height = h
        self.diag = l
