"""HTTP boundaries and worker lifecycle for the hosted application."""
import json
from pathlib import Path
import subprocess
import sys
import time

from fastapi.testclient import TestClient
import pytest

from app import nimbo_reviewer as reviewer
from app.nimbo_reviewer_catalog import CASE, REQUEST


ORIGIN = 'https://reviewer.example'
URL = '/api/reviewer/jobs'


@pytest.fixture
def public_app(tmp_path, monkeypatch):
    inputs, web = tmp_path / 'inputs', tmp_path / 'web'
    inputs.mkdir()
    (web / 'assets').mkdir(parents=True)
    for name in ('model.json', 'pilot.csv', 'heldout.csv', 'pilot.source.json', 'heldout.source.json'):
        (inputs / name).touch()
    replay = Path(__file__).resolve().parents[1] / 'deploy' / 'replay.json'
    app = reviewer.create_app(state=tmp_path / 'state', inputs=inputs, web=web, replay=replay, origin=ORIGIN)
    processes = []

    def pending_job(root, request, *, spawn):
        # A subprocess fixture tests lifecycle only. Production starts the real engine.
        assert request == REQUEST
        job_id = '1' * 32
        prefix = root / reviewer.engine.ROOT_FOLDER / job_id
        prefix.mkdir(parents=True)
        (prefix / 'status.json').write_text('{"seq":1,"state":"RUNNING"}')
        (prefix / 'events.jsonl').write_text(
            '{"seq":1,"stage":"analyze","state":"RUNNING","message":"Worker fixture"}\n')
        process = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'], start_new_session=True)
        processes.append(process)
        # Use the real lifecycle in Reviewer with the fixture process.
        spawn([sys.executable, job_id], cwd=str(root), start_new_session=True)
        return {'job_id': job_id}

    original_popen = subprocess.Popen

    def fixture_popen(argv, **kwargs):
        if 'tools.run_nimbo_reviewer_job' in argv:
            return processes[-1]
        return original_popen(argv, **kwargs)

    monkeypatch.setattr(reviewer.subprocess, 'Popen', fixture_popen)
    monkeypatch.setattr(reviewer.engine, 'submit', pending_job)
    with TestClient(app, base_url=ORIGIN) as client:
        yield app, client, processes


def start(client, data=None, origin=ORIGIN):
    return client.post(URL, headers={'Origin': origin}, json={'case': CASE} if data is None else data)


def test_only_public_routes_are_mounted(public_app):
    _, client, _ = public_app
    assert client.get('/healthz').json() == {'status': 'ready'}
    assert [case['id'] for case in client.get('/api/reviewer/catalog').json()['cases']] == [CASE]
    replay = client.get('/api/reviewer/replay').json()
    assert replay['job']['eligible_for_review_download'] is False
    assert replay['events'][-1]['seq'] == 17
    for route in ('/docs', '/openapi.json', '/api/registry', '/nimbo/auto/jobs',
                  '/nimbo/agentic/ready', '/nimbo/evidence/unified'):
        assert client.get(route).status_code == 404


def test_rejects_unapproved_request_inputs(public_app):
    _, client, processes = public_app
    assert start(client, origin='https://other.example').status_code == 403
    assert start(client, {'case': 'custom-model'}).status_code == 422
    assert start(client, {'case': CASE, 'file': 'outside.csv'}).status_code == 422
    assert start(client, {'case': 'x' * 256}).status_code == 413
    assert client.post(URL, headers={'Origin': ORIGIN}, content='not JSON').status_code == 415
    assert processes == []


def test_single_job_and_browser_bound_data(public_app):
    app, client, processes = public_app
    accepted = start(client)
    assert accepted.status_code == 202
    cookie = accepted.headers['set-cookie'].lower()
    assert all(flag in cookie for flag in ('httponly', 'secure', 'samesite=strict'))
    path = URL + '/' + accepted.json()['job_id']
    assert client.get(path).status_code == 200
    assert client.get(path + '/package').status_code == 404
    other = TestClient(app, base_url=ORIGIN)
    for suffix in ('', '/events', '/resources', '/package'):
        assert other.get(path + suffix).status_code == 404
    assert start(other).status_code == 409
    assert len(processes) == 1


def test_wall_limit_marks_real_worker_failure_and_cleans_state(public_app, monkeypatch):
    app, client, processes = public_app
    monkeypatch.setattr(reviewer, 'WALL_SECONDS', 0)
    accepted = start(client)
    assert accepted.status_code == 202
    path = URL + '/' + accepted.json()['job_id']
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        status = client.get(path).json()
        if status['state'] == 'FAILED':
            break
        time.sleep(0.03)
    assert status['state'] == 'FAILED'
    assert 'wall limit' in status['reason']
    assert status['eligible_for_review_download'] is False
    assert status['comparison'] is None
    assert client.get(path + '/package').status_code == 404
    assert processes[0].poll() is not None
    service = app.state.reviewer
    assert not any(service.jobs[status['job_id']].root.iterdir())
    assert start(TestClient(app, base_url=ORIGIN)).status_code == 429


def test_result_reports_only_reviewed_fields(tmp_path):
    path = tmp_path / 'heldout.json'
    metrics = {'accuracy': 0.925, 'mean_warm_forward_ms': 10, 'private': 'not public'}
    path.write_text(json.dumps({'rounds': 3, 'heldout_unique_examples': 200, 'baseline': metrics,
                                'candidate': metrics, 'decision': {'quality_pass': True, 'private': 17}}))
    report = reviewer.public_report({'heldout_report': str(path)}, tmp_path)
    assert report['gates'] == {'quality_pass': True}
    assert report['baseline'] == {'accuracy': 0.925, 'mean_warm_forward_ms': 10}
    with pytest.raises(ValueError, match='escaped'):
        reviewer.public_report({'heldout_report': str(path)}, tmp_path / 'subdir')


def test_package_quota_reserves_bytes_and_releases_download_slot(public_app, monkeypatch):
    app, client, _ = public_app
    accepted = start(client)
    job_id = accepted.json()['job_id']
    service = app.state.reviewer
    with service.lock:
        job = service.jobs[job_id]
        service.kill(job)
        service.active = None
        job.ended = time.monotonic()
        job.package = job.root / 'fixture.zip'
        job.package.write_bytes(b'zip fixture')
    monkeypatch.setattr(reviewer, 'DOWNLOAD_BUDGET', len(b'zip fixture'))
    path = URL + '/' + job_id + '/package'
    assert client.get(path, headers={'Range': 'bytes=0-2'}).status_code == 416
    assert client.get(path).content == b'zip fixture'
    assert service.streaming is False
    assert client.get(path).status_code == 429
