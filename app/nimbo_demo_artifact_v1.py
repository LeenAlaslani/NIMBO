"""Local-only inspection of the *already exported* reviewed ONNX artifact.

Do not confuse file identity verification with model inference, accuracy verification,
or deploying uploaded model weights. No arbitrary path, model URL or model execution.
"""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse

router = APIRouter(prefix='/nimbo/demo/model-artifact', tags=['local-model-artifact'])
EXPECTED_SHA256 = 'a4d615ada60cd7e9c4869939bf0039c42e717fdad77db0875279ceedaca39c6a'
BASE_DIR = Path('results/onnx_review_candidates_v21')
REVIEW_DIR = Path('../NIMBO_GITHUB_READY_20261031/results/onnx_review_candidates_v21')
MAX_BYTES = 420 * 1024 * 1024
MIN_BYTES = 1_000_000


def _local_only(request: Request) -> None:
    """These large, local artifact endpoints are intentionally inaccessible remotely."""
    host = request.client.host if request.client else ''
    if host not in ('127.0.0.1', '::1', 'localhost', 'testclient'):
        raise HTTPException(status_code=403, detail='Model artifact transfer is local-only')


@lru_cache(maxsize=16)
def _digest(path_string: str, mtime_ns: int, byte_count: int) -> str:
    h = hashlib.sha256()
    with open(path_string, 'rb') as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def find_verified_artifact(root: Path | None = None, expected_sha: str | None = None) -> Path | None:
    """Only a previously exported artifact with the exact recorded SHA can pass."""
    if root is None:
        root = BASE_DIR
        roots = (root, REVIEW_DIR)
    else:
        roots = (root,)
    if expected_sha is None:
        expected_sha = EXPECTED_SHA256
    for search_root in roots:
      if not search_root.is_dir():
        continue
      for file in sorted(search_root.glob('*/model.onnx')):
        try:
            if not file.is_file() or file.is_symlink():
                continue
            real_root = search_root.resolve(strict=True)
            if not file.resolve(strict=True).is_relative_to(real_root):
                continue
            stat = file.stat()
            if not MIN_BYTES <= stat.st_size <= MAX_BYTES:
                continue
            if _digest(str(file), stat.st_mtime_ns, stat.st_size) == expected_sha:
                return file
        except (OSError, ValueError):
            continue
    return None


@router.get('/status')
def model_status(request: Request):
    _local_only(request)
    found = find_verified_artifact()
    return {'available': found is not None,
            'artifact': 'lvwerra/distilbert-imdb ONNX FP32',
            'model_sha256': EXPECTED_SHA256 if found else None,
            'format': 'ONNX',
            'review_only': True,
            'status_note': 'Previously exported candidate; not a new benchmark, automatic deployment or active upload.'}


@router.get('/download')
def download_model(request: Request):
    _local_only(request)
    found = find_verified_artifact()
    if found is None:
        raise HTTPException(status_code=404, detail='No SHA-matched local ONNX review candidate. Run the supported local export first.')
    return FileResponse(found, media_type='application/octet-stream',
                        filename='nimbo-distilbert-imdb-reviewed-fp32.onnx',
                        headers={'Cache-Control': 'no-store', 'X-Model-Review': 'review-only'})


async def digest_upload(chunks, *, max_bytes=MAX_BYTES):
    """Read streamed file bytes without buffering or executing untrusted ONNX graphs."""
    digest = hashlib.sha256()
    size = 0
    async for chunk in chunks:
        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(status_code=413, detail='ONNX artifact exceeds 420 MB')
        digest.update(chunk)
    return digest.hexdigest(), size


@router.post('/verify')
async def verify_uploaded_model(request: Request):
    _local_only(request)
    if request.headers.get('content-type', '').split(';', 1)[0].strip().lower() != 'application/octet-stream':
        raise HTTPException(status_code=415, detail='Upload a raw ONNX file (application/octet-stream)')
    actual_sha, size = await digest_upload(request.stream())
    if not MIN_BYTES <= size <= MAX_BYTES:
        raise HTTPException(status_code=400, detail='Uploaded ONNX artifact has an unexpected size')
    return {'matched': actual_sha == EXPECTED_SHA256,
            'sha256': actual_sha,
            'bytes': size,
            'not_deployed': True,
            'not_executed': True,
            'note': 'File identity check only. Upload did not register or execute a new model.'}
