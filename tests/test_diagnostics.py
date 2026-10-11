from pathlib import Path

from src.diagnostics import diagnose_manifest


def test_diagnose_manifest_checks_mapping_and_paths(tmp_path: Path):
    root = tmp_path / "cache"
    image = root / "Class A" / "one.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"not decoded by this metadata test")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text(
        "archive_path,label,class_index,split\nClass A/one.jpg,Class A,0,train\n",
        encoding="utf-8",
    )
    report = diagnose_manifest(manifest, root)
    assert report["class_index_mapping_ok"]
    assert report["extracted_paths"]["all_exist"]
    assert report["preprocessing"]["evaluation_deterministic"] is None
