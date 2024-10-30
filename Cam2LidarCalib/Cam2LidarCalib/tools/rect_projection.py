import numpy as np
from scipy.spatial.transform import Rotation as R

class PinHoleProjection():
    def __init__(self) -> None:
        # Intrinsic matrix (K)
        self.K = np.array([[903.7596, 0.0, 695.7519],
                           [0.0, 901.9653, 224.2509],
                           [0.0, 0.0, 1.0]])
        
        # Projection matrix (P) for rectified images
        self.p = np.array([[721.5377, 0.0, 609.5593, 0.0],
                           [0.0, 721.5377, 172.854, 0.0],
                           [0.0, 0.0, 1.0, 0.002729905]])
        self.p = np.array([
            [7.215377e+02, 0.000000e+00, 6.095593e+02, 0.000000e+00],
            [0.000000e+00, 7.215377e+02, 1.728540e+02, 0.000000e+00],
            [0.000000e+00, 0.000000e+00, 1.000000e+00, 0.000000e+00]
        ])

        #self.p = np.ones((3, 4))
        # Extrinsic matrix (identity here as a placeholder)
        self.extrinsics = np.array([[0.0000000,  1.0000000,  0.0000000, 0.27],
                                    [0.0000000,  0.0000000,  1.0000000, -0.48],
                                    [1.0000000,  0.0000000,  0.0000000, -0.08],
                                    [0.0, 0.0, 0.0, 1.0] ])
        self.filter_by_intensity = True
        self.intensity_lim = 0.5
        

    def create_rotation_matrix(self, roll, pitch, yaw):
        # Create a rotation object from Euler angles (roll, pitch, yaw)
        rotation = R.from_euler('zyx', [roll, pitch, yaw], degrees=False)
        return rotation.as_matrix()

    def update_extrinsics(self, roll, pitch, yaw, x, y, z):
        # Create rotation matrix from the provided roll, pitch, and yaw
        rotation_matrix = self.create_rotation_matrix(roll, pitch, yaw)
        
        # Construct the new translation vector
        translation_vector = np.array([x, y, z])

        # Update the extrinsic matrix
        self.extrinsics[:3, :3] = rotation_matrix
        self.extrinsics[:3, 3] = translation_vector
        #print(self.extrinsics)


    def project_pinhole(self, points):
        # Ensure points are in homogeneous coordinates
        if points.ndim == 2 and points.shape[-1] == 3:
            points = np.hstack([points, np.ones((points.shape[0], 1))])
            
        points = np.expand_dims(points, axis=-1)  # Convert to shape (N, 4, 1) for matrix multiplication
        intensity = points[:, -1]  # Last column contains intensity values
        avg_intensity = np.mean(intensity)  # Calculate average intensity

        # Filter points based on intensity
        intensity_mask = np.where(np.any(intensity <= self.intensity_lim, axis=1))  # Mask for points with intensity >= average
        points = points[intensity_mask]  # Keep only points with sufficient intensity
        # Apply extrinsics and projection matrix
        uvw = self.p @ self.extrinsics @ points
        uvw = np.squeeze(uvw, axis=-1)  # Remove the last singleton dimension
        uvw /= uvw[:, [2]]  # Normalize homogeneous coordinates
        # Image dimensions
        height, width = 375, 1242
    
        # Validity check for points within image bounds and positive depth
        indices = np.where((points[:, 0, 0] > 0) &
                           (uvw[:, 0] >= 0) & (uvw[:, 0] < width) &
                           (uvw[:, 1] >= 0) & (uvw[:, 1] < height) &
                           (uvw[:, 2] > 0))[0]  # Indices of valid points

        uv = uvw[indices][:, :2].astype(int)  # Extract (u, v) coordinates of valid points
        return indices, uv, intensity[indices]

# Example usage