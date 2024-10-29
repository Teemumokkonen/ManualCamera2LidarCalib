import OpenGL.GL as gl
import pangolin
import cv2
import open3d as o3d

class PangolinViz():
    def __init__(self, database, projection ) -> None:
        scam, dcam, dimg = self.init_pangolin()
        self.scam = scam
        self.dcam = dcam
        self.dimg = dimg
        self.database = database
        self.texture = pangolin.GlTexture(1242, 375, gl.GL_RGB, False, 0, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)
        self.projection = projection

    def init_pangolin(self):
        pangolin.CreateWindowAndBind('Main', 1242, 375)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)

        # Set up the camera with correct aspect ratio
        scam = pangolin.OpenGlRenderState(
            pangolin.ProjectionMatrix(1242, 375, 500, 420, 621, 187.5, 0.1, 1000),
            pangolin.ModelViewLookAt(1, 1, -1, 0, 0, 0, pangolin.AxisDirection.AxisY))

        # Main display
        dcam = pangolin.CreateDisplay()
        dcam.SetBounds(0.0, 1.0, 0.0, 1.0, 1242.0 / 375.0)  # Full window bounds

        # Image display for fullscreen image
        dimg = pangolin.Display('image')
        dimg.SetBounds(0.0, 1.0, 0.0, 1.0, 1242.0 / 375.0)  # Full window bounds
        dimg.SetLock(pangolin.Lock.LockLeft, pangolin.Lock.LockTop)

        return scam, dcam, dimg
    
    def viz_cloud(self, cloud):
        pcd = o3d.geometry.PointCloud()
        print(cloud[:, :3])
        pcd.points = o3d.utility.Vector3dVector(cloud[:, :3])
        o3d.visualization.draw_geometries([pcd])

    def run(self):
        while not pangolin.ShouldQuit():
            while self.database.has_frames:
            # Main Pangolin display loop
                image, cloud = self.database.get_frame()
                coords = self.projection.project_pinhole(cloud)
                self.viz_cloud(cloud)
                #print(coords)
                gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
                gl.glClearColor(0.95, 0.95, 0.95, 1.0)

                self.dcam.Activate(self.scam)

                cv_image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
                height, width, _ = cv_image_rgb.shape
                color = (255, 0, 0)  # Green color
                cv2.circle(cv_image_rgb, (45, 45), radius=5, color=color, thickness=-1)
                print(len(coords))
                for u, v in coords:

                    if 0 <= u < width and 0 <= v < height:  # Check if coordinates are within image dimensions
                        print("point fit")
                        cv2.circle(cv_image_rgb, (u, v), radius=5, color=color, thickness=-1)  # radius=5, thickness=-1 for filled circle


                cv2.imshow('image',cv_image_rgb)
                cv2.waitKey(0)
                cv_image_rgb = cv2.flip(cv_image_rgb, 0) 
                gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 1)
                self.texture.Upload(cv_image_rgb, gl.GL_RGB, gl.GL_UNSIGNED_BYTE)

                # Display the image
                self.dimg.Activate()
                gl.glColor3f(1.0, 1.0, 1.0)
                self.texture.RenderToViewport()

                pangolin.FinishFrame()
                self.database.pop_frame()

            print("Ran out of frames")
