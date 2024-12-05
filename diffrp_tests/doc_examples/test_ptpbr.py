import torch
import diffrp
from diffrp.utils import *
from diffrp.resources import hdris
from diffrp_tests.screenshot_testing import ScreenshotTestingCase


class TestPTPBRExample(ScreenshotTestingCase):
    
    def test_snippets(self):
        scene = diffrp.load_gltf_scene("data/beautygame.glb", compute_tangents=True).static_batching()
        scene.add_light(diffrp.ImageEnvironmentLight(
            intensity=1.0, color=torch.ones(3, device='cuda'),
            image=hdris.newport_loft().cuda(), render_skybox=True
        ))
        camera = diffrp.PerspectiveCamera.from_orbit(
            h=720, w=1280,  # resolution
            radius=0.9, azim=50, elev=10,  # orbit camera position, azim/elev in degrees
            origin=[0, 0, 0],  # orbit camera focus point
            fov=28.12, near=0.005, far=100.0  # intrinsics
        )
        torch.manual_seed(42)
        rp = diffrp.PathTracingSession(scene, camera, diffrp.PathTracingSessionOptions(ray_spp=8, deterministic=True))
        pbr, alpha, extras = rp.pbr()
        pbr_mapped = agx_base_contrast(pbr)
        to_pil(pbr_mapped).save
        self.compare("doctest/ptpbr-beauty-game-noisy", pbr_mapped)
        
        denoiser = diffrp.get_denoiser()
        pbr_denoised = agx_base_contrast(diffrp.run_denoiser(denoiser, pbr, linear_to_srgb(extras['albedo']), extras['world_normal']))
        to_pil(pbr_denoised).save
        self.compare("doctest/ptpbr-beauty-game-denoised", pbr_denoised)
        
        to_pil(float4(pbr_denoised, alpha)).save
        self.compare("doctest/ptpbr-beauty-game-denoised-alpha", float4(pbr_denoised, alpha))

    @torch.no_grad()
    def test_full(self):
        scene = diffrp.load_gltf_scene("data/beautygame.glb", True).static_batching()
        scene.add_light(diffrp.ImageEnvironmentLight(
            intensity=1.0, color=torch.ones(3, device='cuda'),
            image=hdris.newport_loft().cuda(), render_skybox=True
        ))
        camera = diffrp.PerspectiveCamera.from_orbit(
            h=720, w=1280,  # resolution
            radius=0.9, azim=50, elev=10,  # orbit camera position, azim/elev in degrees
            origin=[0, 0, 0],  # orbit camera focus point
            fov=28.12, near=0.005, far=100.0  # intrinsics
        )
        torch.manual_seed(42)
        rp = diffrp.PathTracingSession(scene, camera, diffrp.PathTracingSessionOptions(ray_spp=8))
        pbr, alpha, extras = rp.pbr()
        denoiser = diffrp.get_denoiser()
        pbr_mapped = agx_base_contrast(pbr)
        pbr_denoised = agx_base_contrast(diffrp.run_denoiser(denoiser, pbr, linear_to_srgb(extras['albedo']), extras['world_normal']))
        to_pil(pbr_mapped).save
        to_pil(pbr_denoised).save
        to_pil(float4(pbr_denoised, alpha)).save
        self.compare("doctest/ptpbr-beauty-game-noisy-full", pbr_mapped)
        self.compare("doctest/ptpbr-beauty-game-denoised-full", pbr_denoised)
        self.compare("doctest/ptpbr-beauty-game-denoised-alpha-full", float4(pbr_denoised, alpha))
