"""Identity-only ONNX demo route. File uploads never execute the weights."""
import asyncio
import hashlib
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest
from app import nimbo_demo_artifact_v1 as art


def fake_file(tmp_path):
    directory=tmp_path/'heldout_run';directory.mkdir()
    file=directory/'model.onnx'; file.write_bytes(b'ONNX TEST FIXTURE'+(b'z'*1_000_000))
    return file,hashlib.sha256(file.read_bytes()).hexdigest()


def test_selects_only_proven_artifact(tmp_path):
    file,sha=fake_file(tmp_path)
    assert art.find_verified_artifact(tmp_path,sha)==file
    assert art.find_verified_artifact(tmp_path,'f'*64) is None


def test_model_status_download_and_upload_are_identity_only(monkeypatch,tmp_path):
    file,sha=fake_file(tmp_path)
    monkeypatch.setattr(art,'BASE_DIR',tmp_path)
    monkeypatch.setattr(art,'REVIEW_DIR',tmp_path/'nonexistent')
    monkeypatch.setattr(art,'EXPECTED_SHA256',sha)
    app=FastAPI();app.include_router(art.router)
    client=TestClient(app)
    status=client.get('/nimbo/demo/model-artifact/status');assert status.status_code==200
    assert status.json()['available'] is True
    response=client.get('/nimbo/demo/model-artifact/download')
    assert response.status_code==200
    assert hashlib.sha256(response.content).hexdigest()==sha
    verify=client.post('/nimbo/demo/model-artifact/verify',data=file.read_bytes(),headers={'Content-Type':'application/octet-stream'})
    assert verify.status_code==200 and verify.json()['matched'] is True
    assert verify.json()['not_deployed'] is True and verify.json()['not_executed'] is True


def test_upload_rejects_non_octet_and_excess_size():
    app=FastAPI();app.include_router(art.router)
    client=TestClient(app)
    invalid=client.post('/nimbo/demo/model-artifact/verify',data='not a model',headers={'Content-Type':'text/plain'})
    assert invalid.status_code==415
    invalid=client.post('/nimbo/demo/model-artifact/verify',data=b'abcd',headers={'Content-Type':'application/octet-stream'})
    assert invalid.status_code==400


def test_stream_max_bytes():
    async def over():
        yield b'123'
        yield b'456'
    with pytest.raises(Exception) as exc:
        asyncio.run(art.digest_upload(over(),max_bytes=5))
    assert exc.value.status_code==413
