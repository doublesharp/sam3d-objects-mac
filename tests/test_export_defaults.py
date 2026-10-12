import ply2glb


def test_mps_auto_decode_uses_verified_cpu_path(monkeypatch):
    monkeypatch.delenv('SAM3D_MESH_DECODE_DEVICE',raising=False)
    monkeypatch.delenv('SAM3D_CPU_DECODE_VOXELS',raising=False)
    assert ply2glb.choose_mesh_decode_device('mps',24322)[0]=='cpu'
    assert ply2glb.choose_mesh_decode_device('cuda',24322)[0]=='cuda'


def test_export_defaults_enable_verified_repair_and_fast_texture(monkeypatch):
    for key in ['SAM3D_TEXTURE_MODE','SAM3D_SURFACE_REPAIR','SAM3D_REPAIR_RESOLUTION','SAM3D_REPAIR_RADIUS']:
        monkeypatch.delenv(key,raising=False)
    settings=ply2glb.export_settings()
    assert settings['texture_mode']=='gaussian'
    assert settings['surface_repair'] is True
    assert settings['repair_resolution'] is None
    assert settings['repair_closing_radius'] is None


def test_export_settings_allow_legacy_texture_and_repair_disable(monkeypatch):
    monkeypatch.setenv('SAM3D_TEXTURE_MODE','average')
    monkeypatch.setenv('SAM3D_SURFACE_REPAIR','0')
    settings=ply2glb.export_settings()
    assert settings['texture_mode']=='average' and settings['surface_repair'] is False
