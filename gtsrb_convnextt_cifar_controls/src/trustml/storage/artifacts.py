from __future__ import annotations

import json
import math
import time
import warnings
from pathlib import Path
from typing import Any

from omegaconf import DictConfig

from trustml.core.paths import sanitize_name
from trustml.storage.s3 import S3ArtifactStorage


def relative_artifact_path(run_dir: str | Path, path: str | Path) -> str:
    run_dir = Path(run_dir).resolve()
    path = Path(path).resolve()
    try:
        return str(path.relative_to(run_dir))
    except ValueError:
        return path.name


class ArtifactStore:
    """Local artifact registry with optional W&B or S3-compatible backing storage."""

    def __init__(
        self,
        root: str | Path = "outputs",
        registry_dir: str | Path | None = None,
        cache_dir: str | Path | None = None,
        upload_cfg: DictConfig | dict[str, Any] | None = None,
    ) -> None:
        self.root = Path(root)
        self.registry_dir = Path(registry_dir) if registry_dir else self.root / "_artifacts"
        self.cache_dir = Path(cache_dir) if cache_dir else self.root / "_cache"
        self.upload_cfg = upload_cfg or {}
        self.registry_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "ArtifactStore":
        artifacts_cfg = cfg.get("artifacts", None)
        local_dir = cfg.logging.local.get("dir", "outputs") if cfg.get("logging", None) else "outputs"
        if artifacts_cfg is None:
            return cls(root=local_dir)
        root = artifacts_cfg.get("root", local_dir)
        return cls(
            root=root,
            registry_dir=artifacts_cfg.get("registry_dir", None),
            cache_dir=artifacts_cfg.get("cache_dir", None),
            upload_cfg=artifacts_cfg.get("upload", None),
        )

    def register_checkpoint(
        self,
        *,
        experiment_name: str,
        run_dir: str | Path,
        checkpoint_path: str | Path,
        alias: str,
        epoch: int | None = None,
        metrics: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        checkpoint_path = Path(checkpoint_path)
        run_dir = Path(run_dir)
        exp_id = sanitize_name(experiment_name)
        alias_id = sanitize_name(alias)
        run_id = run_dir.name
        remote_path = f"{exp_id}/{run_id}/checkpoints/{checkpoint_path.name}"
        remote_uri = self._remote_uri(exp_id, alias_id, remote_path) if self._upload_enabled() else None

        record = {
            "schema_version": 1,
            "artifact_type": "checkpoint",
            "experiment": experiment_name,
            "experiment_id": exp_id,
            "alias": alias_id,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "local_path": str(checkpoint_path),
            "relative_path": relative_artifact_path(run_dir, checkpoint_path),
            "root_relative_path": self._relative_to_root(checkpoint_path),
            "remote_uri": remote_uri,
            "epoch": epoch,
            "metrics": _jsonify(metrics or {}),
            "created_at": time.time(),
        }

        self._write_run_manifest(run_dir, record)
        alias_path = self.alias_path(exp_id, alias_id)
        _write_json(alias_path, record)
        if self._upload_enabled():
            self._upload_artifact(
                checkpoint_path=checkpoint_path,
                alias_path=alias_path,
                remote_path=remote_path,
                record=record,
            )
        return record

    def resolve(self, uri_or_path: str | Path | None) -> Path | None:
        if uri_or_path is None:
            return None

        text = str(uri_or_path).strip()
        if text == "" or text.lower() in {"none", "null"}:
            return None
        if text.startswith("artifact://"):
            return self._resolve_artifact_uri(text)
        if text.startswith("s3://"):
            return self._download_s3_uri(text)
        if text.startswith("wandb://"):
            return self._download_wandb_uri(text)
        if text.startswith("file://"):
            return Path(text.removeprefix("file://"))
        return Path(text)

    def alias_path(self, experiment_id: str, alias: str) -> Path:
        return self.registry_dir / sanitize_name(experiment_id) / f"{sanitize_name(alias)}.json"

    def _resolve_artifact_uri(self, uri: str) -> Path:
        experiment_id, alias = _parse_artifact_uri(uri)
        alias_path = self.alias_path(experiment_id, alias)
        # Downloading must work even when this run itself does not upload
        # (e.g. a finetune-only run on a fresh SSH/DataSphere machine).
        if not alias_path.exists() and self._backend() == "s3" and _cfg_get(self.upload_cfg, "bucket", None):
            self._try_download(self._remote_alias_path(experiment_id, alias), alias_path)
        if not alias_path.exists() and self._backend() == "wandb":
            return self._download_wandb_uri(self._wandb_uri(experiment_id, alias))
        if not alias_path.exists():
            raise FileNotFoundError(
                f"Artifact alias not found: {uri}. Expected registry file: {alias_path}"
            )

        record = _read_json(alias_path)
        local_path = self._record_local_path(record)
        if local_path is not None and local_path.exists():
            return local_path

        remote_uri = record.get("remote_uri")
        if remote_uri and str(remote_uri).startswith("s3://"):
            return self._download_s3_uri(str(remote_uri))
        if remote_uri and str(remote_uri).startswith("wandb://"):
            return self._download_wandb_uri(str(remote_uri))

        raise FileNotFoundError(
            f"Artifact {uri} has no existing local file and no remote_uri. "
            f"Registry file: {alias_path}"
        )

    def _record_local_path(self, record: dict[str, Any]) -> Path | None:
        candidates = []
        local_path = record.get("local_path")
        if local_path:
            candidates.append(Path(str(local_path)))
        root_relative_path = record.get("root_relative_path")
        if root_relative_path:
            candidates.append(self.root / str(root_relative_path))
        run_dir = record.get("run_dir")
        relative_path = record.get("relative_path")
        if run_dir and relative_path:
            candidates.append(Path(str(run_dir)) / str(relative_path))

        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else None

    def _write_run_manifest(self, run_dir: Path, record: dict[str, Any]) -> None:
        manifest_path = run_dir / "artifacts.json"
        manifest = {"schema_version": 1, "artifacts": []}
        if manifest_path.exists():
            try:
                manifest = _read_json(manifest_path)
            except json.JSONDecodeError:
                manifest = {"schema_version": 1, "artifacts": []}

        artifacts = [
            artifact
            for artifact in manifest.get("artifacts", [])
            if not (
                artifact.get("artifact_type") == record["artifact_type"]
                and artifact.get("alias") == record["alias"]
            )
        ]
        artifacts.append(record)
        manifest["artifacts"] = artifacts
        _write_json(manifest_path, manifest)

    def _relative_to_root(self, path: Path) -> str | None:
        try:
            return str(path.resolve().relative_to(self.root.resolve()))
        except ValueError:
            return None

    def _upload_enabled(self) -> bool:
        return _as_bool(_cfg_get(self.upload_cfg, "enabled", False))

    def _backend(self) -> str:
        return str(_cfg_get(self.upload_cfg, "backend", "s3")).lower()

    def _remote_uri(self, experiment_id: str, alias: str, remote_path: str) -> str | None:
        try:
            if self._backend() == "s3":
                return self._s3_uri(remote_path)
            if self._backend() == "wandb":
                return self._wandb_uri(experiment_id, alias)
            raise ValueError(f"Unsupported artifact upload backend: {self._backend()}")
        except Exception as exc:  # noqa: BLE001 - local registry still remains useful.
            warnings.warn(f"Artifact remote URI creation failed: {exc}")
            return None

    def _upload_artifact(
        self,
        *,
        checkpoint_path: Path,
        alias_path: Path,
        remote_path: str,
        record: dict[str, Any],
    ) -> None:
        if self._backend() == "s3":
            self._try_upload(checkpoint_path, remote_path)
            self._try_upload(alias_path, self._remote_alias_path(record["experiment_id"], record["alias"]))
            return
        if self._backend() == "wandb":
            self._try_upload_wandb(checkpoint_path, alias_path, record)
            return
        warnings.warn(f"Unsupported artifact upload backend: {self._backend()}")

    def _storage(self) -> S3ArtifactStorage:
        if self._backend() != "s3":
            raise ValueError(f"Unsupported S3 storage backend: {self._backend()}")
        bucket = _cfg_get(self.upload_cfg, "bucket", None)
        if not bucket:
            raise ValueError("artifacts.upload.bucket must be set when upload is enabled.")
        prefix = _cfg_get(self.upload_cfg, "prefix", "")
        endpoint_url = _cfg_get(self.upload_cfg, "endpoint_url", None)
        return S3ArtifactStorage(bucket=str(bucket), prefix=str(prefix or ""), endpoint_url=endpoint_url)

    def _s3_uri(self, remote_path: str) -> str:
        bucket = _cfg_get(self.upload_cfg, "bucket", None)
        if not bucket:
            raise ValueError("artifacts.upload.bucket must be set when upload is enabled.")
        prefix = str(_cfg_get(self.upload_cfg, "prefix", "") or "").strip("/")
        key = f"{prefix}/{remote_path.strip('/')}" if prefix else remote_path.strip("/")
        return f"s3://{bucket}/{key}"

    def _remote_alias_path(self, experiment_id: str, alias: str) -> str:
        return f"_artifacts/{sanitize_name(experiment_id)}/{sanitize_name(alias)}.json"

    def _wandb_artifact_name(self, experiment_id: str) -> str:
        return f"{sanitize_name(experiment_id)}-checkpoint"

    def _wandb_uri(self, experiment_id: str, alias: str) -> str:
        project = _cfg_get(self.upload_cfg, "project", None)
        if not project:
            raise ValueError("artifacts.upload.project must be set for W&B artifacts.")
        entity = _cfg_get(self.upload_cfg, "entity", None)
        artifact_name = self._wandb_artifact_name(experiment_id)
        if entity:
            return f"wandb://{entity}/{project}/{artifact_name}:{sanitize_name(alias)}"
        return f"wandb://{project}/{artifact_name}:{sanitize_name(alias)}"

    def _try_upload(self, local_path: str | Path, remote_path: str) -> None:
        try:
            self._storage().upload_file(local_path, remote_path)
        except Exception as exc:  # noqa: BLE001 - artifact upload must not kill training.
            warnings.warn(f"Artifact upload failed for {local_path}: {exc}")

    def _try_download(self, remote_path: str, local_path: str | Path) -> None:
        try:
            self._storage().download_file(remote_path, local_path)
        except Exception as exc:  # noqa: BLE001 - caller handles the missing alias.
            warnings.warn(f"Artifact registry download failed for {remote_path}: {exc}")

    def _try_upload_wandb(
        self,
        checkpoint_path: Path,
        alias_path: Path,
        record: dict[str, Any],
    ) -> None:
        try:
            import wandb
        except ImportError as exc:
            warnings.warn(f"W&B artifact upload skipped because wandb is not installed: {exc}")
            return

        run = wandb.run
        if run is None:
            warnings.warn(
                "W&B artifact upload skipped because no active wandb run exists. "
                "Set logging.wandb.enabled=true when artifacts.upload.backend=wandb."
            )
            return

        artifact = wandb.Artifact(
            self._wandb_artifact_name(str(record["experiment_id"])),
            type=str(_cfg_get(self.upload_cfg, "type", "model")),
            metadata=record,
        )
        artifact.add_file(str(checkpoint_path), name="checkpoint.pt")
        artifact.add_file(str(alias_path), name="artifact.json")
        run.log_artifact(
            artifact,
            aliases=[str(record["alias"]), f"{record['alias']}-{record['run_id']}"],
        )

    def _download_s3_uri(self, uri: str) -> Path:
        bucket, key = _parse_s3_uri(uri)
        cache_path = self.cache_dir / "s3" / bucket / key
        if cache_path.exists():
            return cache_path
        endpoint_url = _cfg_get(self.upload_cfg, "endpoint_url", None)
        S3ArtifactStorage(bucket=bucket, endpoint_url=endpoint_url).download_file(key, cache_path)
        return cache_path

    def _download_wandb_uri(self, uri: str) -> Path:
        try:
            import wandb
        except ImportError as exc:
            raise RuntimeError("W&B artifact download requires wandb.") from exc

        artifact_path = _wandb_api_path(uri)
        safe_name = artifact_path.replace("/", "_").replace(":", "_")
        try:
            artifact = wandb.Api().artifact(artifact_path, type=str(_cfg_get(self.upload_cfg, "type", "model")))
        except Exception as exc:  # noqa: BLE001 - offline fallback below.
            cached = self._cached_wandb_checkpoint(safe_name)
            if cached is not None:
                warnings.warn(
                    f"W&B API unavailable for {uri} ({exc}); using possibly stale cached copy: {cached}"
                )
                return cached
            raise

        # Aliases like :best are mutable, so the cache is keyed by content digest.
        digest = str(getattr(artifact, "digest", "") or "nodigest").replace(":", "_")[:16]
        cache_path = self.cache_dir / "wandb" / f"{safe_name}__{digest}"
        checkpoint_path = cache_path / "checkpoint.pt"
        if checkpoint_path.exists():
            return checkpoint_path

        artifact_dir = Path(artifact.download(root=str(cache_path)))
        checkpoint_path = artifact_dir / "checkpoint.pt"
        if checkpoint_path.exists():
            return checkpoint_path
        candidates = sorted(artifact_dir.glob("*.pt"))
        if candidates:
            return candidates[0]
        raise FileNotFoundError(f"W&B artifact has no checkpoint .pt file: {uri}")

    def _cached_wandb_checkpoint(self, safe_name: str) -> Path | None:
        wandb_cache = self.cache_dir / "wandb"
        candidates = sorted(
            wandb_cache.glob(f"{safe_name}__*/checkpoint.pt"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        legacy_cache = wandb_cache / safe_name / "checkpoint.pt"
        if legacy_cache.exists():
            candidates.append(legacy_cache)
        return candidates[0] if candidates else None


def resolve_artifact_path(uri_or_path: str | Path | None, cfg: DictConfig | None = None) -> Path | None:
    if cfg is None:
        return ArtifactStore().resolve(uri_or_path)
    return ArtifactStore.from_config(cfg).resolve(uri_or_path)


def _parse_artifact_uri(uri: str) -> tuple[str, str]:
    value = uri.removeprefix("artifact://").strip("/")
    parts = value.split("/")
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"Artifact URI must be artifact://<experiment>/<alias>, got: {uri}")
    return sanitize_name(parts[0]), sanitize_name(parts[1])


def _parse_s3_uri(uri: str) -> tuple[str, str]:
    value = uri.removeprefix("s3://")
    bucket, _, key = value.partition("/")
    if not bucket or not key:
        raise ValueError(f"S3 URI must be s3://<bucket>/<key>, got: {uri}")
    return bucket, key


def _wandb_api_path(uri: str) -> str:
    value = uri.removeprefix("wandb://").strip("/")
    parts = value.split("/")
    if len(parts) == 2:
        project, artifact = parts
        return f"{project}/{artifact}"
    if len(parts) == 3:
        entity, project, artifact = parts
        return f"{entity}/{project}/{artifact}"
    raise ValueError(f"W&B URI must be wandb://<project>/<artifact>:<alias> or wandb://<entity>/<project>/<artifact>:<alias>, got: {uri}")


def _cfg_get(cfg: DictConfig | dict[str, Any] | None, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    return cfg.get(key, default)


def _as_bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _write_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(_jsonify(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def _read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _jsonify(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(v) for v in value]
    if hasattr(value, "detach"):
        tensor = value.detach().cpu()
        return tensor.item() if tensor.numel() == 1 else tensor.tolist()
    if hasattr(value, "item") and value.__class__.__module__.startswith("numpy"):
        return value.item()
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value
