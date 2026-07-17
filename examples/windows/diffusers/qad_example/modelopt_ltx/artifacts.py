from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path  # noqa: TC003
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


@dataclass(frozen=True)
class DeployManifest:
    schema_version: int
    recipe: str
    base_model: str
    modelopt_state: str
    modelopt_state_sha256: str
    model_weights: str | None
    model_weights_sha256: str | None
    native_checkpoint: str
    native_checkpoint_sha256: str
    native_contract: str

    @classmethod
    def load(cls, bundle: Path) -> DeployManifest:
        return cls(**json.loads((bundle / "deploy_manifest.json").read_text()))


def _copy_independent(source: Path, destination: Path) -> str:
    if not source.is_file():
        raise FileNotFoundError(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    os.chmod(temporary, source.stat().st_mode & 0o777)
    temporary.replace(destination)
    if destination.stat().st_nlink != 1:
        raise RuntimeError(f"Deploy artifact is hardlinked: {destination}")
    if source.stat().st_ino == destination.stat().st_ino:
        raise RuntimeError(f"Deploy artifact aliases source inode: {destination}")
    return sha256_file(destination)


def resolve_checkpoint(checkpoint: Path) -> tuple[Path, Path | None]:
    if checkpoint.is_dir():
        states = sorted(checkpoint.glob("**/modelopt_state_step_*.pth"))
        if len(states) != 1:
            raise ValueError(
                f"Expected exactly one ModelOpt state below {checkpoint}, found {len(states)}"
            )
        weights = sorted(checkpoint.glob("**/model_weights_step_*.safetensors"))
        if len(weights) > 1:
            raise ValueError(
                f"Expected at most one weights file below {checkpoint}, found {len(weights)}"
            )
        return states[0], weights[0] if weights else None

    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if checkpoint.name.startswith("modelopt_state_step_"):
        return checkpoint, None
    if not checkpoint.name.startswith("model_weights_step_"):
        raise ValueError(f"Unsupported checkpoint name: {checkpoint.name}")
    state = checkpoint.with_name(
        checkpoint.name.replace("model_weights_", "modelopt_state_")
    ).with_suffix(".pth")
    if not state.is_file():
        raise FileNotFoundError(state)
    return state, checkpoint


def create_deploy_bundle(
    checkpoint: Path,
    output: Path,
    *,
    base_model: Path,
    recipe: str,
    native_checkpoint: Path,
    native_contract: Path,
) -> DeployManifest:
    state, weights = resolve_checkpoint(checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    state_name = "modelopt_state.pth"
    weights_name = "model_weights.safetensors" if weights else None
    state_hash = _copy_independent(state, output / state_name)
    weights_hash = (
        _copy_independent(weights, output / weights_name)
        if weights is not None and weights_name is not None
        else None
    )
    native_name = "native_fp8.safetensors"
    contract_name = "native_fp8.contract.json"
    native_hash = _copy_independent(native_checkpoint, output / native_name)
    _copy_independent(native_contract, output / contract_name)
    manifest = DeployManifest(
        schema_version=1,
        recipe=recipe,
        base_model=str(base_model),
        modelopt_state=state_name,
        modelopt_state_sha256=state_hash,
        model_weights=weights_name,
        model_weights_sha256=weights_hash,
        native_checkpoint=native_name,
        native_checkpoint_sha256=native_hash,
        native_contract=contract_name,
    )
    atomic_json(output / "deploy_manifest.json", asdict(manifest))
    return manifest


def verify_deploy_bundle(bundle: Path) -> DeployManifest:
    manifest = DeployManifest.load(bundle)
    state = bundle / manifest.modelopt_state
    if sha256_file(state) != manifest.modelopt_state_sha256:
        raise RuntimeError(f"ModelOpt state checksum mismatch: {state}")
    if manifest.model_weights:
        weights = bundle / manifest.model_weights
        if sha256_file(weights) != manifest.model_weights_sha256:
            raise RuntimeError(f"Model weights checksum mismatch: {weights}")
    native_checkpoint = bundle / manifest.native_checkpoint
    if sha256_file(native_checkpoint) != manifest.native_checkpoint_sha256:
        raise RuntimeError(f"Native FP8 checkpoint checksum mismatch: {native_checkpoint}")
    if not (bundle / manifest.native_contract).is_file():
        raise RuntimeError(f"Native FP8 contract missing: {bundle / manifest.native_contract}")
    return manifest
