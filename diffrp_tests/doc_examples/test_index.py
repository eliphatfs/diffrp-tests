import diffrp
import trimesh.creation
from diffrp.utils import *
from diffrp_tests.screenshot_testing import ScreenshotTestingCase


class TestIndexExample(ScreenshotTestingCase):

    def test_get_started(self):
        # create the mesh (cpu)
        mesh = trimesh.creation.icosphere(radius=0.8)
        # initialize the DiffRP scene
        scene = diffrp.Scene()
        # register the mesh, load vertices and faces arrays to GPU
        scene.add_mesh_object(diffrp.MeshObject(diffrp.DefaultMaterial(), gpu_f32(mesh.vertices), gpu_i32(mesh.faces)))
        # default camera at [0, 0, 3.2] looking backwards
        camera = diffrp.PerspectiveCamera()
        # create the SurfaceDeferredRenderSession, a deferred-rendering rasterization pipeline session
        rp = diffrp.SurfaceDeferredRenderSession(scene, camera)
        # convert output tensor to PIL Image and save
        to_pil(rp.false_color_camera_space_normal()).save
        self.compare('doctest/get-started', rp.false_color_camera_space_normal())
