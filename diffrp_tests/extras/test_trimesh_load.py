import torch
import trimesh

from diffrp import *
from diffrp.resources.hdris import newport_loft
from diffrp_tests.screenshot_testing import ScreenshotTestingCase


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


class TestTrimesh(ScreenshotTestingCase):

    def render_trimesh(self, name, fp):
        cam = PerspectiveCamera.from_orbit(640, 640, 3.8, 30, 20, [0, 0, 0])
        scene = normalize(from_trimesh_scene(trimesh.load(fp), compute_tangents=True))
        scene.add_light(ImageEnvironmentLight(intensity=1.0, color=torch.ones(3, device='cuda'), image=newport_loft().cuda()))
        rp = SurfaceDeferredRenderSession(scene, cam, False)
        frame = rp.albedo()
        frame = float4(linear_to_srgb(frame.rgb), frame.a)
        self.compare("loader/trimesh-%s-albedo" % name, frame)
        frame = rp.false_color_mask_mso()
        self.compare("loader/trimesh-%s-mask" % name, frame)
        frame = rp.false_color_camera_space_normal()
        self.compare("loader/trimesh-%s-normal" % name, frame)
        frame = agx_base_contrast(rp.pbr().rgb)
        self.compare("loader/trimesh-%s-pbr" % name, frame)

    @torch.no_grad()
    def test_spot(self):
        self.render_trimesh('spot', 'data/spot/cow.obj')

    @torch.no_grad()
    def test_scissors(self):
        self.render_trimesh('scissors', 'data/scissors/obj_000017.ply')

    @torch.no_grad()
    def test_vccube(self):
        self.render_trimesh('vccube', 'data/vccube.ply')

    @torch.no_grad()
    def test_banana(self):
        self.render_trimesh('banana', 'data/banana.ply')
