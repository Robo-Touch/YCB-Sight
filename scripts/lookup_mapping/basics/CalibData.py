import numpy as np
class CalibData:
    def __init__(self, dataPath):
        self.dataPath = dataPath
        data = np.load(dataPath)

        self.bins = data['bins']
        self.grad_mag = data['grad_mag']
        self.grad_dir = data['grad_dir']
