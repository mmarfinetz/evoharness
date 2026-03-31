from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any, Iterable

from harness.models import CandidateSnapshot


def create_snapshot_from_paths(
    *,
    snapshots_root: Path,
    candidate_id: str,
    adapter_name: str,
    label: str,
    artifact_sources: Iterable[tuple[Path, str]],
    notes_source: Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> CandidateSnapshot:
    sources = list(artifact_sources)
    if not sources:
        raise SystemExit(f"cannot create snapshot {candidate_id}: no artifact sources")

    snapshots_root.mkdir(parents=True, exist_ok=True)
    destination = snapshots_root / candidate_id
    if destination.exists():
        raise SystemExit(f"snapshot already exists: {destination}")

    temp_dir = Path(tempfile.mkdtemp(prefix=f"{candidate_id}-", dir=snapshots_root))
    artifacts_dir = temp_dir / "artifacts"
    hashes: dict[str, str] = {}
    artifact_paths: list[str] = []

    try:
        for source_path, relative_path in sources:
            if not source_path.exists():
                raise SystemExit(
                    f"cannot create snapshot {candidate_id}: missing artifact {source_path}"
                )
            artifact_path = artifacts_dir / relative_path
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, artifact_path)
            artifact_paths.append(relative_path)
            hashes[relative_path] = file_sha256(artifact_path)

        notes_path = None
        if notes_source is not None and notes_source.exists():
            notes_path = temp_dir / "notes.md"
            shutil.copyfile(notes_source, notes_path)

        artifact_hash = combined_artifact_hash(artifacts_dir, artifact_paths)
        manifest_path = temp_dir / "manifest.json"
        manifest = {
            "adapter": adapter_name,
            "candidate_id": candidate_id,
            "label": label,
            "artifact_hash": artifact_hash,
            "artifact_paths": artifact_paths,
            "artifact_hashes": hashes,
            "notes_path": None if notes_path is None else notes_path.name,
            "metadata": metadata or {},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        temp_dir.rename(destination)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise

    final_notes_path = destination / "notes.md"
    return CandidateSnapshot(
        candidate_id=candidate_id,
        label=label,
        path=destination,
        artifact_paths=tuple(artifact_paths),
        artifact_hash=artifact_hash,
        adapter_name=adapter_name,
        notes_path=final_notes_path if final_notes_path.exists() else None,
        metadata_path=destination / "manifest.json",
    )


def clone_snapshot(
    *,
    source: CandidateSnapshot,
    snapshots_root: Path,
    candidate_id: str,
    label: str,
    metadata: dict[str, Any] | None = None,
) -> CandidateSnapshot:
    artifact_sources = [
        (source.path / "artifacts" / relative_path, relative_path)
        for relative_path in source.artifact_paths
    ]
    base_metadata: dict[str, Any] = {"source_snapshot": str(source.path)}
    if metadata:
        base_metadata.update(metadata)
    return create_snapshot_from_paths(
        snapshots_root=snapshots_root,
        candidate_id=candidate_id,
        adapter_name=source.adapter_name,
        label=label,
        artifact_sources=artifact_sources,
        notes_source=source.notes_path,
        metadata=base_metadata,
    )


def materialize_snapshot(snapshot: CandidateSnapshot, workspace_root: Path) -> None:
    artifacts_dir = snapshot.path / "artifacts"
    for relative_path in snapshot.artifact_paths:
        source_path = artifacts_dir / relative_path
        destination_path = workspace_root / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination_path)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def combined_artifact_hash(artifacts_dir: Path, artifact_paths: Iterable[str]) -> str:
    digest = hashlib.sha256()
    for relative_path in sorted(artifact_paths):
        digest.update(relative_path.encode("utf-8"))
        digest.update(b"\0")
        digest.update((artifacts_dir / relative_path).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
