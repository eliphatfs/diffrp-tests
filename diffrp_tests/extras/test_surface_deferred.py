import os
import glob
import torch
import trimesh
import trimesh.creation

from diffrp import *
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

    @torch.no_grad()
    def test_gltfs(self):
        for fp in glob.glob("data/*.glb"):
            self.render_gltf(fp)
