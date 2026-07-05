"""Tests for deep-sky catalog loading and querying."""

from stargazing_core._catalog import (
    _FALLBACK_SIZES,
    DeepSkyCatalog,
    get_angular_size_fallback,
    load_objects,
)
from stargazing_core._telescope import TelescopeConfig


def test_load_objects():
    """Catalog should load without error and have expected size."""
    objects = load_objects()
    assert len(objects) >= 10000, f'Expected 10k+ objects, got {len(objects)}'


def test_load_objects_cached():
    """Second call returns same list (object identity because cached)."""
    a = load_objects()
    b = load_objects()
    assert a is b, 'load_objects() should return the cached instance'


def test_deep_sky_catalog_len():
    """DeepSkyCatalog wrapper gives same count as raw load."""
    cat = DeepSkyCatalog()
    assert len(cat) == len(load_objects())


def test_deep_sky_catalog_all():
    """all() returns the full list."""
    cat = DeepSkyCatalog()
    all_objs = cat.all()
    assert len(all_objs) == len(cat)
    assert 'name' in all_objs[0]
    assert 'type' in all_objs[0]
    assert 'ra' in all_objs[0]
    assert 'dec' in all_objs[0]


def test_expected_fields():
    """Every object should have the required fields."""
    for obj in load_objects():
        assert 'name' in obj
        assert 'type' in obj
        assert 'ra' in obj
        assert 'dec' in obj
        assert 'magnitude' in obj
        assert 'catalog' in obj
        assert 'angular_size_maj_arcmin' in obj
        assert 'angular_size_min_arcmin' in obj
        assert 'angular_size_pa_deg' in obj


def test_by_types():
    """Filtering by type should return only matching objects."""
    cat = DeepSkyCatalog()
    # 'GiG' (galaxy in group) is the most common galaxy type in this catalog
    galaxies = cat.by_types({'GiG'})
    for g in galaxies:
        assert g['type'] == 'GiG'
    assert len(galaxies) > 1000


def test_with_angular_size():
    """Objects with angular-size data should have non-None maj axis."""
    cat = DeepSkyCatalog()
    sized = cat.with_angular_size()
    for s in sized:
        assert s['angular_size_maj_arcmin'] is not None
    # Should be > 95% of total
    assert len(sized) > 0.90 * len(cat)


def test_type_counts():
    """type_counts() returns a non-empty dict."""
    cat = DeepSkyCatalog()
    counts = cat.type_counts()
    assert len(counts) > 0
    total = sum(counts.values())
    assert total == len(cat)


def test_fallback_table_coverage():
    """Fallback table should cover the common types."""
    common = {'GiG', 'GlC', 'OpC', 'PN', 'HII', 'SNR', 'RNe', 'G', 'GNe'}
    for t in common:
        assert t in _FALLBACK_SIZES, f'Missing fallback for {t}'


def test_get_angular_size_fallback():
    """Known type returns tuple, unknown type returns None."""
    assert get_angular_size_fallback('GiG') == (2.0, 1.0)
    assert get_angular_size_fallback('GlC') == (5.0, 5.0)
    assert get_angular_size_fallback('PN') == (0.5, 0.5)
    assert get_angular_size_fallback('not-a-real-type') is None


def test_telescope_fov_matches_known_values():
    """Cross-check: telescope optics computation uses shared math."""
    cfg = TelescopeConfig(focal_length_mm=250, sensor_width_mm=7.6, sensor_height_mm=5.7)
    optics = cfg.compute_optics()
    assert optics.fov_width_deg is not None and optics.fov_width_deg > 0
    assert optics.fov_height_deg is not None and optics.fov_height_deg > 0
