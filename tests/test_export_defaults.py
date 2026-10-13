import ply2glb
import pytest


def test_texture_profiles_use_2k_preview_and_4k_final(monkeypatch):
    monkeypatch.delenv('SAM3D_TEXTURE_SIZE', raising=False)
    assert ply2glb.texture_size_for_profile('preview') == 2048
    assert ply2glb.texture_size_for_profile('final') == 4096


def test_explicit_texture_size_overrides_profile(monkeypatch):
    monkeypatch.setenv('SAM3D_TEXTURE_SIZE', '8192')
    assert ply2glb.texture_size_for_profile('preview') == 8192
    assert ply2glb.texture_size_for_profile('final') == 8192


@pytest.mark.parametrize('size', ['not-an-int', '0', '-1', '16384'])
def test_invalid_texture_size_is_rejected(monkeypatch, size):
    monkeypatch.setenv('SAM3D_TEXTURE_SIZE', size)
    with pytest.raises(ValueError, match='TEXTURE_SIZE'):
        ply2glb.texture_size_for_profile('final')


def test_invalid_texture_profile_is_rejected(monkeypatch):
    monkeypatch.delenv('SAM3D_TEXTURE_SIZE', raising=False)
    with pytest.raises(ValueError, match='TEXTURE_PROFILE'):
        ply2glb.texture_size_for_profile('unknown')


def test_texture_profile_cli_defaults_to_final_and_accepts_preview(monkeypatch):
    monkeypatch.delenv('SAM3D_TEXTURE_PROFILE', raising=False)
    monkeypatch.setattr('sys.argv', ['ply2glb.py', '/tmp/source'])
    assert ply2glb.parse_args().texture_profile == 'final'
    monkeypatch.setattr('sys.argv', ['ply2glb.py', '/tmp/source', '--texture-profile', 'preview'])
    assert ply2glb.parse_args().texture_profile == 'preview'


def test_texture_profile_environment_applies_to_cli(monkeypatch):
    monkeypatch.setenv('SAM3D_TEXTURE_PROFILE', 'preview')
    monkeypatch.setattr('sys.argv', ['ply2glb.py', '/tmp/source'])
    assert ply2glb.parse_args().texture_profile == 'preview'


def test_invalid_texture_size_fails_before_loading_source(monkeypatch):
    monkeypatch.setenv('SAM3D_TEXTURE_SIZE', 'not-an-int')
    monkeypatch.setattr('sys.argv', ['ply2glb.py', '/missing/source'])
    with pytest.raises(ValueError, match='TEXTURE_SIZE'):
        ply2glb.main()


def test_mps_auto_decode_uses_verified_cpu_path(monkeypatch):
    monkeypatch.delenv('SAM3D_MESH_DECODE_DEVICE',raising=False)
    monkeypatch.delenv('SAM3D_CPU_DECODE_VOXELS',raising=False)
    assert ply2glb.choose_mesh_decode_device('mps',24322)[0]=='cpu'
    assert ply2glb.choose_mesh_decode_device('cuda',24322)[0]=='cuda'


def test_export_defaults_enable_verified_repair_and_fast_texture(monkeypatch):
    monkeypatch.delenv('SAM3D_REPAIR_FORCE',raising=False)
    for key in ['SAM3D_TEXTURE_MODE','SAM3D_SURFACE_REPAIR','SAM3D_REPAIR_RESOLUTION','SAM3D_REPAIR_RADIUS']:
        monkeypatch.delenv(key,raising=False)
    settings=ply2glb.export_settings()
    assert settings['texture_mode']=='gaussian'
    assert settings['surface_repair'] is True
    assert settings['repair_force'] is False
    assert settings['repair_resolution'] is None
    assert settings['repair_closing_radius'] is None


def test_export_settings_allow_legacy_texture_and_repair_disable(monkeypatch):
    monkeypatch.setenv('SAM3D_TEXTURE_MODE','average')
    monkeypatch.setenv('SAM3D_SURFACE_REPAIR','0')
    settings=ply2glb.export_settings()
    assert settings['texture_mode']=='average' and settings['surface_repair'] is False



def test_porous_surface_repair_is_explicitly_opt_in(monkeypatch):
    monkeypatch.setenv('SAM3D_REPAIR_FORCE','1')
    assert ply2glb.export_settings()['repair_force'] is True


def test_protected_detail_profile_is_explicit_and_invalid_profile_rejected(monkeypatch):
    import pytest
    monkeypatch.delenv('SAM3D_REPAIR_PROFILE',raising=False)
    assert ply2glb.export_settings()['repair_profile']=='surface'
    monkeypatch.setenv('SAM3D_REPAIR_PROFILE','protected-detail')
    assert ply2glb.export_settings()['repair_profile']=='protected-detail'
    monkeypatch.setenv('SAM3D_REPAIR_PROFILE','silently-corrupt')
    with pytest.raises(ValueError,match='PROFILE'):
        ply2glb.export_settings()
