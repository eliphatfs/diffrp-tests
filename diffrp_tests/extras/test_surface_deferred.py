import os
import glob
import torch
import trimesh
import trimesh.creation
import nvdiffrast.torch as dr

from diffrp import *
from diffrp.utils import *
from diffrp.resources.hdris import newport_loft
from diffrp_tests.screenshot_testing import ScreenshotTestingCase


class CameraSpaceVertexNormalMaterial(SurfaceMaterial):

    def shade(self, su: SurfaceUniform, si: SurfaceInput) -> SurfaceOutputStandard:
        return SurfaceOutputStandard(
            transform_vector(normalized(si.world_normal), su.V) / 2 + 0.5,
            alpha=full_like_vec(si.world_normal, 0.5, 1),
            aovs={'test': full_like_vec(si.world_normal, 0.4, 3)}
        )


class TestSurfaceDeferredRP(ScreenshotTestingCase):
    
    @torch.no_grad()
    def test_highres(self):
        cam = PerspectiveCamera(h=4096, w=3072)
        mesh = trimesh.creation.icosphere(radius=0.8)
        scene = Scene()
        scene.add_mesh_object(MeshObject(DefaultMaterial(), gpu_f32(mesh.vertices), gpu_i32(mesh.faces)))
        rp = SurfaceDeferredRenderSession(scene, cam)
        self.compare('sdrp/highres', rp.false_color_camera_space_normal())
        
    @torch.no_grad()
    def test_highmesh(self):
        scene = Scene()
        mesh = trimesh.creation.icosphere(10, radius=0.7)
        scene.add_mesh_object(MeshObject(DefaultMaterial(), gpu_f32(mesh.vertices), gpu_i32(mesh.faces), normals='smooth'))
        mesh = trimesh.creation.icosphere(8, radius=0.8)
        scene.add_mesh_object(MeshObject(DefaultMaterial(), gpu_f32(mesh.vertices), gpu_i32(mesh.faces), normals='smooth'))
        camera = PerspectiveCamera(h=640, w=640)
        rp = SurfaceDeferredRenderSession(scene, camera, ctx=dr.RasterizeGLContext())
        self.compare('sdrp/highmesh', rp.false_color_world_space_normal().rgb)
        rp = PathTracingSession(scene, camera, PathTracingSessionOptions(ray_depth=2, ray_spp=1, deterministic=True))
        self.compare('sdrp/highmesh', rp.pbr()[-1]['world_normal'] * 0.5 + 0.5, psnr_threshold=30)

    @torch.no_grad()
    def test_cylinder(self):
        # renders a cylinder
        cam = PerspectiveCamera(h=512, w=512)
        mesh = trimesh.creation.cylinder(0.3, 1.0)
        v, f = mesh.vertices, mesh.faces
        scene = Scene()
        scene.add_mesh_object(MeshObject(
            CameraSpaceVertexNormalMaterial(),
            gpu_f32(v), gpu_i32(f),
            M=gpu_f32(trimesh.transformations.identity_matrix()[[0, 2, 1, 3]])
        ))
        rp = SurfaceDeferredRenderSession(scene, cam, False)
        fb = rp.false_color_camera_space_normal()
        aov = rp.aov('test', [0.1, 0.3, 0.1])
        self.compare("sdrp/cylinder-transparent", fb)
        self.compare("sdrp/cylinder-aov", aov)


@torch.no_grad()
def normalize(gltf: Scene):
    bmin = gpu_f32([1e30] * 3)
    bmax = gpu_f32([-1e30] * 3)
    world_v = [transform_point4x3(prim.verts, prim.M) for prim in gltf.objects]
    for verts, prim in zip(world_v, gltf.objects):
        bmin = torch.minimum(bmin, verts.min(0)[0])
        bmax = torch.maximum(bmax, verts.max(0)[0])
    center = (bmin + bmax) / 2
    radius = max(length(verts - center).max() for verts in world_v).item()
    T = trimesh.transformations.translation_matrix(-center.cpu().numpy())
    S = trimesh.transformations.scale_matrix(1 / radius)
    M = gpu_f32(S @ T)
    for prim in gltf.objects:
        prim.M = M @ prim.M
    return gltf


class TestGLTF(ScreenshotTestingCase):

    def render_gltf(self, fp):
        name = os.path.splitext(os.path.basename(fp))[0]
        cam = PerspectiveCamera.from_orbit(640, 640, 3.8, 30, 20, [0, 0, 0])
        gltf = normalize(load_gltf_scene(fp, compute_tangents=True))
        rp = SurfaceDeferredRenderSession(gltf, cam, False)
        frame = rp.albedo()
        frame = float4(linear_to_srgb(frame.rgb), frame.a)
        self.compare("sdrp/gltf-%s-albedo" % name, frame)
        frame = rp.false_color_mask_mso()
        self.compare("sdrp/gltf-%s-mask" % name, frame)
        frame = rp.false_color_camera_space_normal()
        self.compare("sdrp/gltf-%s-normal" % name, frame)

    def test_tangent(self):
        cam = PerspectiveCamera.from_orbit(1024, 1024, 5.0, 0, 0, [0, 0, 0])
        gltf = load_gltf_scene('data/tangent/NormalTangentTest.glb', compute_tangents=True)
        gltf.add_light(ImageEnvironmentLight(1.0, gpu_f32([1] * 3), gpu_f32(newport_loft())))
        rp = SurfaceDeferredRenderSession(gltf, cam, False, options=SurfaceDeferredRenderSessionOptions(max_layers=3))
        rgb = background_alpha_compose(1, rp.pbr())
        self.compare('sdrp/gltf-tangent-test', agx_base_contrast(rgb))

    @torch.no_grad()
    def test_gltfs(self):
        for fp in glob.glob("data/*.glb"):
            self.render_gltf(fp)
