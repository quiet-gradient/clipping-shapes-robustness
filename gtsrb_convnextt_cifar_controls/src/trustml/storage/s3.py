from __future__ import annotations

from pathlib import Path


class ArtifactStorage:
    def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        raise NotImplementedError

    def download_file(self, remote_path: str, local_path: str | Path) -> None:
        raise NotImplementedError


class LocalOnlyStorage(ArtifactStorage):
    def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        return None

    def download_file(self, remote_path: str, local_path: str | Path) -> None:
        raise FileNotFoundError("LocalOnlyStorage cannot download remote artifacts.")


class S3ArtifactStorage(ArtifactStorage):
    def __init__(self, bucket: str, prefix: str = "", endpoint_url: str | None = None) -> None:
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError("S3ArtifactStorage requires boto3. Install trustml[s3].") from exc
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.client = boto3.client("s3", endpoint_url=endpoint_url)

    def _key(self, remote_path: str) -> str:
        remote_path = remote_path.strip("/")
        return f"{self.prefix}/{remote_path}" if self.prefix else remote_path

    def upload_file(self, local_path: str | Path, remote_path: str) -> None:
        self.client.upload_file(str(local_path), self.bucket, self._key(remote_path))

    def download_file(self, remote_path: str, local_path: str | Path) -> None:
        local_path = Path(local_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, self._key(remote_path), str(local_path))

