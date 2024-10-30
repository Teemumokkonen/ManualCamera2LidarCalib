import OpenGL.GL as gl
import pangolin
import cv2
import open3d as o3d
import numpy as np

class PangolinViz():
    def __init__(self, database, projection) -> None:
        
        self.database = database
        self.projection = projection

    def init_pangolin(self):
        pangolin.CreateWindowAndBind('Main', 1242 * 2, 375 * 2)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        # Set up the camera with correct aspect ratio
        panel = pangolin.CreatePanel('ui')
        panel.SetBounds(0.0, 1.0, 0.0, 0.1)  # Adjusted for better balance

        scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(1242, 375, 500, 420, 621, 187.5, 0.1, 1000),
            pangolin.ModelViewLookAt(1, 1, -1, 0, 0, 0, pangolin.AxisDirection.AxisY)
        )

        # Main display
        dcam = pangolin.CreateDisplay()
        dcam.SetHandler(pangolin.Handler3D(scam))
        dcam.SetBounds(0.3, 0.5, 1.0, 1.0, 1242.0 / 375.0)  # Adjusted to leave space for panel

        # Image display for fullscreen image
        dimg = pangolin.Display('image')
        dimg.SetBounds(0.0, 1.0, 0.1, 1.0, 1242.0 / 375.0)  # Full screen except for the panel at the bottom
        dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)
        return scam, dcam, dimg
    
    def viz_cloud(self, cloud):
        pcd = o3d.geometry.PointCloud()
        print(cloud[:, :3])
        pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
        o3d.visualization.draw_geometries([pcd])

    def run(self):
        gl.glEnable(gl.GL_DEPTH_TEST)

        scam, dcam, dimg = self.init_pangolin()

        float_slider = pangolin.VarFloat('ui.x', value=0.27, min=-0.5, max=0.5)
        float_slider_1 = pangolin.VarFloat('ui.y', value=-0.48, min=-0.5, max=0.5)
        float_slider_2 = pangolin.VarFloat('ui.z', value=-0.08, min=-0.5, max=0.5)
        float_slider_3 = pangolin.VarFloat('ui.roll', value=-1.57, min=-3.141592, max=3.14159)
        float_slider_4 = pangolin.VarFloat('ui.pitch', value=-1.57, min=-3.14159/2, max=3.14159/2)
        float_slider_5 = pangolin.VarFloat('ui.yaw', value=-0.0, min=-3.14159, max=3.14159)
        texture = pangolin.GlTexture(1242, 375, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        pop_frame = pangolin.VarBool('ui.pop_frame', value=False, toggle=False)

        while not pangolin.ShouldQuit() and self.database.has_frames:
            if pangolin.Pushed(pop_frame):
                self.database.pop_frame()
            #slider_value = float(float_slider)
            x = float(float_slider)            
            y = float(float_slider_1)
            z = float(float_slider_2)
            roll = float(float_slider_3)
            pitch = float(float_slider_4)
            yaw = float(float_slider_5)            
            self.projection.update_extrinsics(roll, pitch, yaw, x, y, z)
            # Main Pangolin display loop
            image, cloud = self.database.get_frame()
            #cloud = np.array([[1.0, 2.0, 100.0]])  # Example points in 3D
            #                  #[3.0, 1.0, 10.0],
            #                  #[-2.0, -1.0, 10.0]])
#
            coords = self.projection.project_pinhole(cloud)
            #self.viz_cloud(cloud)
            #print(coords)
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            gl.glClearColor(0.95, 0.95, 0.95, 1.0)

            dcam.Activate(scam)

            cv_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
            cv_image_rgb = cv2.flip(cv_image_rgb, 0)
            cv_image_rgb = cv2.flip(cv_image_rgb, 1) 
            height, width, _ = cv_image_rgb.shape
            color = (255, 0, 0)  # Green color
            #cv2.circle(cv_image_rgb, (45, 45), radius=5, color=color, thickness=-1)
            for u, v in coords[1]:
                if 0 <= u < width and 0 <= v < height:  # Check if coordinates are within image dimensions
                    #print("point fit")
                    cv2.circle(cv_image_rgb, (u, v), radius=1, color=color, thickness=-1)  # radius=5, thickness=-1 for filled circle

            #cv_image_rgb = cv2.flip(cv_image_rgb, 0) 
            #cv_image_rgb = cv2.flip(cv_image_rgb, 1) 
            #cv2.imshow('image',cv_image_rgb)
            #cv2.waitKey(0)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
            texture.Upload(cv_image_rgb, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

            # Display the image
            dimg.Activate()
            gl.glColor3f(1.0, 1.0, 1.0)
            texture.RenderToViewport()

            pangolin.FinishFrame()
            pangolin.glDrawColouredCube()

        print("Ran out of frames")
