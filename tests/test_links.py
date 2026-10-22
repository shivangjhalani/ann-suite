"""Tests for the data/index symlink manifest and container mounts."""

from __future__ import annotations

from pathlib import Path

import pytest

from ann_suite.utils.links import apply_links, load_manifest, symlink_target_mounts


def _manifest(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "links.yaml"
    path.write_text(body)
    return path


@pytest.fixture
def store(tmp_path: Path) -> Path:
    """An external storage tree with two indices and one dataset."""
    ext = tmp_path / "ext"
    (ext / "starling" / "sift10m").mkdir(parents=True)
    (ext / "starling" / "sift100m").mkdir()
    (ext / "data" / "bigann").mkdir(parents=True)
    return ext


class TestManifest:
    def test_rejects_relative_target(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="absolute"):
            load_manifest(_manifest(tmp_path, "links:\n  data: [rel/path]\n"))

    def test_rejects_non_list(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="list"):
            load_manifest(_manifest(tmp_path, "links:\n  data: /abs\n"))


class TestApplyLinks:
    def test_creates_glob_and_plain_links_then_checks_clean(
        self, tmp_path: Path, store: Path
    ) -> None:
        repo = tmp_path / "repo"
        m = _manifest(
            tmp_path,
            f"links:\n  data: [{store}/data/bigann]\n  indices/starling: ['{store}/starling/*']\n",
        )
        report = apply_links(m, repo)
        assert len(report.created) == 3 and report.problems == 0
        assert (repo / "indices/starling/sift100m").readlink() == store / "starling/sift100m"
        again = apply_links(m, repo, check_only=True)
        assert len(again.ok) == 3 and again.problems == 0

    def test_check_only_reports_missing_without_creating(self, tmp_path: Path, store: Path) -> None:
        repo = tmp_path / "repo"
        m = _manifest(tmp_path, f"links:\n  data: [{store}/data/bigann]\n")
        report = apply_links(m, repo, check_only=True)
        assert report.missing == [repo / "data/bigann"]
        assert not (repo / "data/bigann").exists()

    def test_never_overwrites_real_dir_or_foreign_link(self, tmp_path: Path, store: Path) -> None:
        repo = tmp_path / "repo"
        (repo / "data" / "bigann").mkdir(parents=True)
        (repo / "indices" / "starling").mkdir(parents=True)
        (repo / "indices" / "starling" / "sift10m").symlink_to(tmp_path)
        m = _manifest(
            tmp_path,
            f"links:\n  data: [{store}/data/bigann]\n  indices/starling: ['{store}/starling/*']\n",
        )
        report = apply_links(m, repo)
        assert report.blocked == [repo / "data/bigann"]
        assert report.wrong == [(repo / "indices/starling/sift10m", tmp_path)]
        assert (repo / "data/bigann").is_dir() and not (repo / "data/bigann").is_symlink()

    def test_reports_unmatched_and_dangling(self, tmp_path: Path, store: Path) -> None:
        repo = tmp_path / "repo"
        (repo / "data").mkdir(parents=True)
        (repo / "data" / "old").symlink_to(tmp_path / "gone")
        m = _manifest(tmp_path, f"links:\n  data: ['{store}/nothing-*', {store}/data/bigann]\n")
        report = apply_links(m, repo)
        assert report.unmatched == [f"{store}/nothing-*"]
        assert report.dangling == [repo / "data/old"]


class TestSymlinkTargetMounts:
    def test_mounts_outside_targets_once_and_skips_inside_ones(
        self, tmp_path: Path, store: Path
    ) -> None:
        data = tmp_path / "repo" / "data"
        idx = tmp_path / "repo" / "indices"
        (idx / "starling").mkdir(parents=True)
        data.mkdir()
        (data / "real").mkdir()
        (data / "bigann").symlink_to(store / "data" / "bigann")
        (data / "inside").symlink_to(data / "real")
        (idx / "starling" / "sift10m").symlink_to(store / "starling" / "sift10m")
        (idx / "starling" / "all").symlink_to(store / "starling")  # parent of sift10m
        mounts = symlink_target_mounts([data, idx])
        assert mounts == [(store / "data" / "bigann").resolve(), (store / "starling").resolve()]

    def test_relative_escaping_link_is_not_mounted(self, tmp_path: Path, store: Path) -> None:
        data = tmp_path / "repo" / "data"
        data.mkdir(parents=True)
        (data / "rel").symlink_to(Path("../../ext/data/bigann"))
        assert symlink_target_mounts([data]) == []
