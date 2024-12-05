import diffrp
from diffrp.utils import *
from diffrp_tests.screenshot_testing import ScreenshotTestingCase


class TestGLTFExample(ScreenshotTestingCase):
    
    def test_snippets(self):
        scene = diffrp.load_gltf_scene("data/spheres.glb")
        scene = diffrp.load_gltf_scene("data/spheres.glb", compute_tangents=True)
        
        scene = scene.static_batching()
        
        camera = diffrp.PerspectiveCamera.from_orbit(
            h=1080, w=1920,  # resolution
            radius=0.02, azim=0, elev=0,  # orbit camera position, azim/elev in degrees
            origin=[0.003, 0.003, 0],  # orbit camera focus point
            fov=30, near=0.005, far=10.0  # intrinsics
        )
        
        rp = diffrp.SurfaceDeferredRenderSession(scene, camera)
        
        nor = rp.false_color_camera_space_normal()
        mso = rp.false_color_mask_mso()
        
        rp.camera_space_normal()
        
        nor.xyz
        nor.rgb
        nor.a
        nor.rgb * nor.a
        background_alpha_compose([0.5, 0.5, 0.5], nor)
        background_alpha_compose(0.5, nor)
        to_pil(nor).show
        to_pil(mso).show
        self.compare('doctest/gltf-spheres-nor', nor)
        self.compare('doctest/gltf-spheres-mso', mso)
        albedo = rp.albedo()
        albedo_srgb = rp.albedo_srgb()
        to_pil(albedo_srgb).show
        self.compare('doctest/gltf-spheres-albedo-srgb', albedo_srgb)
        float4(linear_to_srgb(albedo.rgb), albedo.a)
        
        rp.aov('my_aov', [0.])
        rp.albedo()
        rp.albedo_srgb()
        rp.emission()
        with self.assertRaises(ValueError):
            rp.pbr()
        rp.false_color_camera_space_normal()
        rp.false_color_world_space_normal()
        rp.false_color_mask_mso()
        rp.false_color_nocs()
        rp.camera_space_normal()
        rp.world_space_normal()
        rp.local_position()
        rp.world_position()
        rp.depth()
        rp.distance()
        rp.view_dir()
        
        import torch
        from diffrp.resources.hdris import newport_loft
        scene.add_light(diffrp.ImageEnvironmentLight(intensity=1.0, color=torch.ones(3, device='cuda'), image=newport_loft().cuda()))
        
        rp = diffrp.SurfaceDeferredRenderSession(scene, camera)
        pbr = rp.pbr()
        to_pil(pbr).show
        self.compare('doctest/gltf-spheres-pbr', pbr)
        
        pbr = diffrp.agx_base_contrast(pbr.rgb)
        pbr_aa = rp.nvdr_antialias(pbr)
        to_pil(pbr_aa).show
        self.compare('doctest/gltf-spheres-pbr-aa', pbr_aa)
        
        pbr = ssaa_downscale(pbr, 2)
        
        scene.lights.clear()
        scene.add_light(diffrp.ImageEnvironmentLight(intensity=1.0, color=torch.ones(3, device='cuda'), image=newport_loft().cuda(), render_skybox=False))
        rp = diffrp.SurfaceDeferredRenderSession(scene, camera)
        
        pbr = rp.pbr()
        pbr = float4(agx_base_contrast(pbr.rgb), pbr.a)
        
        pbr = diffrp.background_alpha_compose(1, pbr)
        
        rp = diffrp.SurfaceDeferredRenderSession(scene, camera, opaque_only=False)
        pbr_premult = rp.compose_layers(
            rp.pbr_layered() + [torch.zeros([1080, 1920, 3], device='cuda')],
            rp.alpha_layered() + [torch.zeros([1080, 1920, 1], device='cuda')]
        )
        
        pbr = ssaa_downscale(pbr_premult, 2)
        pbr_1 = saturate(float4(agx_base_contrast(pbr.rgb) / torch.clamp_min(pbr.a, 0.0001), pbr.a))
        pbr_2 = float4(agx_base_contrast(pbr.rgb / torch.clamp_min(pbr.a, 0.0001)), pbr.a)
        self.compare('doctest/gltf-spheres-pbr-transparent-1', pbr_1)
        self.compare('doctest/gltf-spheres-pbr-transparent-2', pbr_2)

    def test_full(self):
        import torch
        from diffrp.resources import hdris

        scene = diffrp.load_gltf_scene("data/spheres.glb")
        scene.add_light(diffrp.ImageEnvironmentLight(
            intensity=1.0, color=torch.ones(3, device='cuda'),
            image=hdris.newport_loft().cuda()
        ))

        camera = diffrp.PerspectiveCamera.from_orbit(
            h=1080, w=1920,  # resolution
            radius=0.02, azim=0, elev=0,  # orbit camera position, azim/elev in degrees
            origin=[0.003, 0.003, 0],  # orbit camera focus point
            fov=30, near=0.005, far=10.0  # intrinsics
        )

        rp = diffrp.SurfaceDeferredRenderSession(scene, camera)
        pbr = rp.pbr()

        pbr_aa = rp.nvdr_antialias(pbr.rgb)
        to_pil(agx_base_contrast(pbr_aa)).save
        self.compare('doctest/gltf-spheres-full', agx_base_contrast(pbr_aa))
