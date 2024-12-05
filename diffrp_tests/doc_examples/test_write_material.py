import diffrp
import trimesh.creation
from diffrp.utils import *
from diffrp_tests.screenshot_testing import ScreenshotTestingCase


class ProceduralMaterial(diffrp.SurfaceMaterial):

    def shade(self, su: diffrp.SurfaceUniform, si: diffrp.SurfaceInput) -> diffrp.SurfaceOutputStandard:
        p = si.world_pos
        r = torch.exp(torch.sin(p.x) + torch.cos(p.y) - 2)
        g = torch.exp(torch.sin(p.y) + torch.cos(p.z) - 2)
        b = torch.exp(torch.sin(p.z) + torch.cos(p.x) - 2)
        v = torch.where(p.z > 0.5, 0.5 * torch.exp(-p.x * p.x * p.y * p.y * 1000), 0)
        albedo = float3(r, g, b) + v
        return diffrp.SurfaceOutputStandard(albedo=albedo)


class NeuralMaterial(diffrp.SurfaceMaterial):
    def __init__(self) -> None:
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 16),
            torch.nn.SiLU(),
            torch.nn.Linear(16, 3),
            torch.nn.Sigmoid()
        ).cuda()

    def shade(self, su: diffrp.SurfaceUniform, si: diffrp.SurfaceInput) -> diffrp.SurfaceOutputStandard:
        return diffrp.SurfaceOutputStandard(albedo=self.net(si.world_normal))


class TestWritingMaterialExample(ScreenshotTestingCase):
    
    def test_procedural_material(self):
        mesh = trimesh.creation.icosphere(radius=0.8)
        scene = diffrp.Scene().add_mesh_object(diffrp.MeshObject(ProceduralMaterial(), gpu_f32(mesh.vertices), gpu_i32(mesh.faces)))
        camera = diffrp.PerspectiveCamera(h=2048, w=2048)
        rp = diffrp.SurfaceDeferredRenderSession(scene, camera)
        to_pil(rp.albedo_srgb()).save
        self.compare('doctest/procedural-albedo', rp.albedo_srgb())

    def test_neural_material(self):
        mesh = trimesh.creation.icosphere(radius=0.8)
        torch.manual_seed(42)
        material = NeuralMaterial()
        scene = diffrp.Scene().add_mesh_object(diffrp.MeshObject(material, gpu_f32(mesh.vertices), gpu_i32(mesh.faces)))
        camera = diffrp.PerspectiveCamera(h=512, w=512)
        rp = diffrp.SurfaceDeferredRenderSession(scene, camera)
        to_pil(rp.albedo_srgb()).save
        self.compare('doctest/neural-albedo', rp.albedo_srgb())
        
        pred = rp.albedo_srgb()
        torch.nn.functional.mse_loss(pred, torch.rand_like(pred)).backward()
        self.assertGreater(material.net[0].weight.grad.abs().max().item(), 0.0)
        rp = diffrp.SurfaceDeferredRenderSession(scene, camera)
        pred = rp.albedo()
        torch.nn.functional.mse_loss(pred, torch.rand_like(pred)).backward()
