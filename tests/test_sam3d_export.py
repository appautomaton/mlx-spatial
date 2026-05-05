import numpy as np

from mlx_spatial.sam3d_export import (
    SAM3D_GAUSSIAN_PLY_FIELDS,
    pack_sam3d_gaussian_rows,
    read_sam3d_gaussian_ply_vertex_count,
    sam3d_basic_glb_payload,
    sam3d_binary_row_size,
    write_sam3d_basic_glb,
    write_sam3d_gaussians_ply,
)
from mlx_spatial.sam3d_mesh import assemble_sam3d_mesh_fields, extract_sam3d_mesh_from_features


def _fixture_gaussians():
    return {
        "xyz": np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
        "features_dc": np.array([[[1.0, 0.0, 0.0]], [[0.0, 1.0, 0.0]]], dtype=np.float32),
        "opacity": np.array([[0.7], [0.8]], dtype=np.float32),
        "scale": np.array([[0.01, 0.02, 0.03], [0.03, 0.02, 0.01]], dtype=np.float32),
        "rotation": np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float32),
    }


def _interior_mesh_features():
    coords = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
        ],
        dtype=np.int32,
    )
    feats = np.zeros((8, 101), dtype=np.float32)
    feats[:, :8] = 2.0
    for row, coord in enumerate(coords[:, 1:]):
        local_corner = tuple(1 - coord)
        corner_index = int(local_corner[0] + local_corner[1] * 2 + local_corner[2] * 4)
        feats[row, corner_index] = -2.0
    feats[:, 53:101] = 0.25
    return coords, feats


def test_pack_sam3d_gaussian_rows_uses_official_column_order():
    rows = pack_sam3d_gaussian_rows(**_fixture_gaussians())

    assert rows.shape == (2, len(SAM3D_GAUSSIAN_PLY_FIELDS))
    assert rows[0, :3].tolist() == [0.10000000149011612, 0.20000000298023224, 0.30000001192092896]
    assert rows[0, 3:6].tolist() == [0.0, 0.0, 0.0]
    assert rows[0, 6:9].tolist() == [1.0, 0.0, 0.0]
    assert rows[0, 9] == np.float32(0.7)


def test_write_sam3d_gaussians_ply_writes_binary_official_fields(tmp_path):
    path = tmp_path / "gaussians.ply"

    stats = write_sam3d_gaussians_ply(path, **_fixture_gaussians())
    payload = path.read_bytes()

    assert stats.vertex_count == 2
    assert stats.format == "binary_little_endian"
    assert stats.fields == SAM3D_GAUSSIAN_PLY_FIELDS
    assert b"format binary_little_endian 1.0\n" in payload
    assert b"property float f_dc_0\n" in payload
    assert b"property float opacity\n" in payload
    assert b"property float scale_2\n" in payload
    assert b"property float rot_3\n" in payload
    assert read_sam3d_gaussian_ply_vertex_count(path) == 2
    assert stats.bytes_written == len(payload)
    assert len(payload.split(b"end_header\n", 1)[1]) == 2 * sam3d_binary_row_size()


def test_write_sam3d_gaussians_ply_can_write_ascii_debug_output(tmp_path):
    path = tmp_path / "gaussians_ascii.ply"

    stats = write_sam3d_gaussians_ply(path, binary=False, **_fixture_gaussians())

    text = path.read_text(encoding="utf-8")
    assert stats.format == "ascii"
    assert "format ascii 1.0" in text
    assert "element vertex 2" in text


def test_write_sam3d_basic_glb_writes_blender_readable_container(tmp_path):
    path = tmp_path / "mesh.glb"
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    colors = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    stats = write_sam3d_basic_glb(path, vertices=vertices, faces=faces, colors=colors)
    payload = path.read_bytes()

    assert payload[:4] == b"glTF"
    assert b"JSON" in payload[:32]
    assert b"BIN\x00" in payload
    assert b"POSITION" in payload
    assert b"COLOR_0" in payload
    assert stats.vertex_count == 3
    assert stats.face_count == 1
    assert stats.has_vertex_color is True
    assert stats.bytes_written == len(payload)


def test_write_sam3d_basic_glb_uses_atomic_replace_and_cleans_temp(tmp_path, monkeypatch):
    path = tmp_path / "mesh.glb"
    vertices = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)
    faces = np.array([[0, 1, 2]], dtype=np.int64)

    import mlx_spatial.sam3d_export as sam3d_export

    def fail_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr(sam3d_export.os, "replace", fail_replace)

    try:
        write_sam3d_basic_glb(path, vertices=vertices, faces=faces)
    except OSError as error:
        assert "replace failed" in str(error)
    else:
        raise AssertionError("expected atomic replace failure to propagate")

    assert not path.exists()
    assert not (tmp_path / ".mesh.glb.tmp").exists()


def test_sam3d_basic_glb_payload_rejects_invalid_faces():
    vertices = np.zeros((3, 3), dtype=np.float32)
    faces = np.array([[0, 1, 3]], dtype=np.int64)

    try:
        sam3d_basic_glb_payload(vertices=vertices, faces=faces)
    except ValueError as error:
        assert "outside the vertex array" in str(error)
    else:
        raise AssertionError("expected invalid face indices to fail")


def test_sam3d_mesh_field_guard_returns_blocker_without_writing_artifact(tmp_path):
    path = tmp_path / "mesh.glb"
    coords = np.array([[0, 0, 0, 0]], dtype=np.int32)
    feats = np.zeros((1, 53), dtype=np.float32)

    result = assemble_sam3d_mesh_fields(
        coords,
        feats,
        extraction_resolution=16,
        max_dense_bytes=1,
    )

    assert result.ready is False
    assert result.fields is None
    assert result.blocker is not None
    assert result.blocker.stage == "mesh-decoder"
    assert result.blocker.metadata["estimated_dense_bytes"] > result.blocker.metadata["max_dense_bytes"]
    assert not path.exists()


def test_write_sam3d_basic_glb_accepts_extracted_mesh_output(tmp_path):
    path = tmp_path / "mesh.glb"
    coords, feats = _interior_mesh_features()
    mesh = extract_sam3d_mesh_from_features(coords, feats, extraction_resolution=2, use_color=True)

    assert mesh.ready is True
    assert mesh.vertices is not None
    assert mesh.faces is not None
    stats = write_sam3d_basic_glb(path, vertices=mesh.vertices, faces=mesh.faces, colors=mesh.colors)

    assert stats.vertex_count == mesh.vertices.shape[0]
    assert stats.face_count == mesh.faces.shape[0]
    assert stats.has_vertex_color is True
    assert path.read_bytes()[:4] == b"glTF"
