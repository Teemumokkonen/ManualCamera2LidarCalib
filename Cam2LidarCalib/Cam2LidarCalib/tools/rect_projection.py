"""
Projection tools
"""
import numpy as np

class PinHoleProjection():
    def __init__(self) -> None:
        #self.instrincs = instrincs
        self.K = np.array([[903.7596, 0.0, 695.7519],
              [0.0, 901.9653, 224.2509],
              [0.0, 0.0, 1.0]])
        
        self.p = np.array([
        [721.5377, 0.0, 609.5593, -339.5242],
        [0.0, 721.5377, 172.854, 2.199936],
        [0.0 ,0.0, 1.0, 0.002729905]])

    def project_pinhole(self, points):
        # skip extrinsics
        coords = []

        for point in points:
            x_norm = point[0]/point[2]
            y_norm = point[1]/point[2]
            u = self.K[0][0] * x_norm + self.K[0][2]
            v = self.K[1][1] * y_norm + self.K[1][2]
            coords.append((u, v))

        return coords