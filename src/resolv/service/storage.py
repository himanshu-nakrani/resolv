"""Artifact storage: local filesystem or S3-compatible object storage.

Keys are tenant-scoped (``tenants/<installation>/repos/<repo>/runs/<run>/...``)
and validated so they can never escape the storage root.
"""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Protocol

_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*$")


def check_key(key: str) -> str:
    if not _KEY_RE.match(key) or ".." in key.split("/") or "//" in key:
        raise ValueError(f"invalid storage key: {key!r}")
    return key


def run_prefix(installation_id: int, repo_id: int, run_id: str) -> str:
    return check_key(f"tenants/{installation_id}/repos/{repo_id}/runs/{run_id}")


class Storage(Protocol):
    def put_bytes(self, key: str, data: bytes) -> None: ...
    def get_bytes(self, key: str) -> bytes | None: ...
    def list(self, prefix: str) -> list[str]: ...
    def delete_prefix(self, prefix: str) -> int: ...


def put_dir(storage: Storage, prefix: str, directory: Path) -> int:
    count = 0
    for path in sorted(directory.rglob("*")):
        if path.is_file() and not path.is_symlink():
            storage.put_bytes(f"{prefix}/{path.relative_to(directory).as_posix()}", path.read_bytes())
            count += 1
    return count


class LocalStorage:
    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        path = (self.root / check_key(key)).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"invalid storage key: {key!r}")
        return path

    def put_bytes(self, key: str, data: bytes) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_bytes(data)
        os.replace(tmp, path)

    def get_bytes(self, key: str) -> bytes | None:
        path = self._path(key)
        return path.read_bytes() if path.is_file() else None

    def list(self, prefix: str) -> list[str]:
        base = self._path(prefix)
        if not base.is_dir():
            return []
        return sorted(p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file())

    def delete_prefix(self, prefix: str) -> int:
        base = self._path(prefix)
        if not base.exists():
            return 0
        count = sum(1 for p in base.rglob("*") if p.is_file())
        shutil.rmtree(base)
        return count


class S3Storage:
    """S3 or S3-compatible storage (requires boto3: ``pip install resolv[s3]``)."""

    def __init__(self, bucket: str, prefix: str = "", client=None) -> None:
        if client is None:
            import boto3

            client = boto3.client("s3", endpoint_url=os.environ.get("RESOLV_S3_ENDPOINT_URL") or None)
        self.client = client
        self.bucket = bucket
        self.prefix = prefix.strip("/")

    def _key(self, key: str) -> str:
        check_key(key)
        return f"{self.prefix}/{key}" if self.prefix else key

    def put_bytes(self, key: str, data: bytes) -> None:
        self.client.put_object(Bucket=self.bucket, Key=self._key(key), Body=data)

    def get_bytes(self, key: str) -> bytes | None:
        try:
            return self.client.get_object(Bucket=self.bucket, Key=self._key(key))["Body"].read()
        except self.client.exceptions.NoSuchKey:
            return None

    def list(self, prefix: str) -> list[str]:
        keys, token = [], None
        strip = len(self.prefix) + 1 if self.prefix else 0
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": self._key(prefix) + "/"}
            if token:
                kwargs["ContinuationToken"] = token
            page = self.client.list_objects_v2(**kwargs)
            keys += [obj["Key"][strip:] for obj in page.get("Contents", [])]
            if not page.get("IsTruncated"):
                return sorted(keys)
            token = page.get("NextContinuationToken")

    def delete_prefix(self, prefix: str) -> int:
        keys = self.list(prefix)
        for i in range(0, len(keys), 1000):
            batch = [{"Key": self._key(k)} for k in keys[i : i + 1000]]
            self.client.delete_objects(Bucket=self.bucket, Delete={"Objects": batch, "Quiet": True})
        return len(keys)


def open_storage(spec: str) -> Storage:
    if spec.startswith("s3://"):
        bucket, _, prefix = spec[5:].partition("/")
        return S3Storage(bucket, prefix)
    return LocalStorage(spec)
