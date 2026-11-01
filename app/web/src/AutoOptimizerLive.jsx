/** NIMBO M8a — local operator console for REAL M7 job data (not simulated 3D). */
import React, { useCallback, useEffect, useRef, useState } from 'react';
import { Activity, CheckCircle2, Download, ExternalLink, FileArchive, RefreshCcw, ShieldCheck, XCircle } from 'lucide-react';
import './auto-optimizer-live.css';
import { stageState, canReviewDownload } from './autoOptimizerJobView.mjs';

const BASE = 'http://127.0.0.1:8005';
const JOB_ID = /^[0-9a-f]{32}$/;
const TERMINAL = new Set(['COMPLETED_REVIEW_ONLY','COMPLETED_REJECTED','FAILED','INTERRUPTED']);
const FILENAME = /^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,90}\.(csv|jsonl)$/;
const STAGES = [
  { id: 'analyze', label: 'ANALYZE', worker: 'CAPTAIN', hint: 'Baseline and model intake' },
  { id: 'optimize', label: 'OPTIMIZE', worker: 'MAKO', hint: 'Real ONNX candidate and paired inference' },
  { id: 'verify', label: 'VERIFY', worker: 'VERA', hint: 'Quality, CPU and heldout evidence' },
  { id: 'deliver', label: 'DELIVER', worker: 'METRI', hint: 'Package and offline-reload evidence' },
];
async function jsonGet(path, signal) {
  const response = await fetch(BASE + path, { signal, cache: 'no-store' });
  if (!response.ok) throw new Error(`Local optimizer returned HTTP ${response.status}`);
  return response.json();
}
function pct(value) { return typeof value === 'number' && Number.isFinite(value) ? `${(value * 100).toFixed(2)}%` : 'Not reported'; }

export default function AutoOptimizerLive() {
  const [form, setForm] = useState({ model_id: '', revision: '', pilot_file: 'pilot.csv', heldout_file: 'tweeteval_sentiment_test_200.csv', preprocess: 'cardiff_tweet', objective: 'speed' });
  const [lookup, setLookup] = useState('');
  const [id, setId] = useState('');
  const [job, setJob] = useState(null);
  const [events, setEvents] = useState([]);
  const [message, setMessage] = useState('');
  const [connected, setConnected] = useState(false);
  const [busy, setBusy] = useState(false);
  const activeRequest = useRef(0);
  const edit = (field, value) => setForm(prev => ({ ...prev, [field]: value }));

  const refresh = useCallback(async (selectedId, signal) => {
    if (!JOB_ID.test(selectedId)) return;
    const token = ++activeRequest.current;
    try {
      const [status, timeline] = await Promise.all([
        jsonGet(`/nimbo/auto/jobs/${selectedId}`, signal),
        jsonGet(`/nimbo/auto/jobs/${selectedId}/events`, signal),
      ]);
      if (token !== activeRequest.current || signal?.aborted) return;
      setJob(status);
      setEvents(Array.isArray(timeline.events) ? timeline.events : []);
      setConnected(true);
      setMessage('');
    } catch (err) {
      if (signal?.aborted || token !== activeRequest.current) return;
      setConnected(false);
      setMessage(`Cannot read the local M7 API. ${err.message}. Check that port 8005 is running.`);
    }
  }, []);

  useEffect(() => {
    if (!id) return;
    const controller = new AbortController();
    refresh(id, controller.signal);
    // Live events are polled, never generated from a decorative timer.
    const timer = (!job || !TERMINAL.has(job.state)) ? setInterval(() => refresh(id, controller.signal), 2000) : null;
    return () => { controller.abort(); if (timer) clearInterval(timer); activeRequest.current += 1; };
  }, [id, job?.state, refresh]);

  const load = (event) => {
    event.preventDefault();
    const next = lookup.trim();
    if (!JOB_ID.test(next)) { setMessage('Enter a valid 32-character job ID.'); return; }
    setJob(null); setEvents([]); setConnected(false); setMessage('');
    if (next === id) refresh(next);
    else setId(next);
  };

  const submit = async (event) => {
    event.preventDefault();
    if (!/^[0-9a-f]{40}$/.test(form.revision.trim()) || !/^[A-Za-z0-9][A-Za-z0-9._-]+\/[A-Za-z0-9][A-Za-z0-9._-]+$/.test(form.model_id.trim()) ||
        !FILENAME.test(form.pilot_file.trim()) || !FILENAME.test(form.heldout_file.trim())) {
      setMessage('Provide a supported owner/model ID, immutable 40-character revision SHA, and simple CSV/JSONL filenames.'); return;
    }
    setBusy(true); setMessage('');
    try {
      const response = await fetch(BASE + '/nimbo/auto/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...form, model_id: form.model_id.trim(), revision: form.revision.trim(), pilot_file: form.pilot_file.trim(), heldout_file: form.heldout_file.trim() }) });
      const result = await response.json().catch(() => ({}));
      if (response.status !== 202 || !JOB_ID.test(result.job_id || '')) throw new Error(result.detail || `HTTP ${response.status}`);
      setJob(null); setEvents([]); setLookup(result.job_id); setId(result.job_id); setConnected(true);
    } catch (err) {
      setMessage(`Job not accepted: ${String(err.message || err)}. Inputs must already exist in the project's local results folders.`);
    } finally { setBusy(false); }
  };
  const ended = job && TERMINAL.has(job.state);
  const last = events[events.length - 1];
  return <section className="nimauto" aria-label="Auto-Optimizer live jobs">
    <header className="nimautoHeader"><span className="nimautoKicker"><Activity size={16}/> MODEL ENGINEERING / LIVE JOBS</span><h1>Auto-Optimizer</h1><p>One real job: inspect baseline, generate ONNX, verify quality and CPU, then deliver a review-only package when every gate passes. No synthetic agent reasoning or progress.</p>
      <span className={'nimautoConnection ' + (connected ? 'ok' : 'notReady')}>{connected ? 'M7 API connected · local only' : 'Connect to M7 API on 127.0.0.1:8005'}</span>
    </header>
    <div className="nimautoColumns">
      <form className="nimautoCard" onSubmit={submit}>
        <h2>Start a supported-model job</h2><p>Local developer workflow. Provide public Hugging Face classifier ID and files already staged on this Mac. Only ONNX FP32 speed optimization is connected in M7a.</p>
        <label>Model ID <input value={form.model_id} onChange={e=>edit('model_id', e.target.value)} placeholder="owner/model" required maxLength={193}/></label>
        <label>Immutable model revision (40-digit SHA) <input value={form.revision} onChange={e=>edit('revision', e.target.value)} placeholder="40 lowercase hex characters" required maxLength={40}/></label>
        <div className="nimautoTwo"><label>Pilot filename <input value={form.pilot_file} onChange={e=>edit('pilot_file', e.target.value)} required/></label><label>Heldout filename <input value={form.heldout_file} onChange={e=>edit('heldout_file', e.target.value)} required/></label></div>
        <label>Preprocessing <select value={form.preprocess} onChange={e=>edit('preprocess',e.target.value)}><option value="identity">Identity</option><option value="cardiff_tweet">Cardiff Tweet</option></select></label>
        <div className="nimautoHelp"><b>File locations:</b><code>results/auto_optimizer_job_inputs_v1/</code> for pilot, <code>results/auto_optimizer_heldout_v1/</code> for heldout. Filename only, not an upload.</div>
        <button className="nimautoPrimary" type="submit" disabled={busy}>{busy ? 'Submitting…' : 'Start real optimization job'}</button>
      </form>
      <div className="nimautoCard nimautoMonitor">
        <h2>Job monitor</h2><form onSubmit={load} className="nimautoLookup"><label>Open existing job ID <input value={lookup} onChange={e=>setLookup(e.target.value)} placeholder="32-character job ID" maxLength={32}/></label><button type="submit">Load job</button></form>
        {message && <p role="alert" className="nimautoError">{message}</p>}
        {job ? <><div className="nimautoStatus"><b>{job.state}</b><small>{job.job_id}</small><button type="button" onClick={()=>refresh(id)}><RefreshCcw size={15}/> Refresh</button></div>
          <div className="nimautoStages">{STAGES.map(stage => { const state = stageState(stage.id,events,job.state); return <div key={stage.id} className={'nimautoStage state-'+state}><span>{state === 'rejected' ? <XCircle size={17}/> : state === 'verified' ? <CheckCircle2 size={17}/> : <ShieldCheck size={17}/>}</span><div><b>{stage.label}</b><small>{stage.worker} · {stage.hint}</small><em>{state}</em></div></div>; })}</div>
          <div className="nimautoResult"><b>Decision:</b> {job.decision || (ended ? job.state : 'Pending verification')}<small>{last?.message || 'Waiting for real backend events'}</small></div>
          {last?.evidence?.accuracy != null && <p>Latest measured labeled accuracy: <b>{pct(last.evidence.accuracy)}</b></p>}
          {connected && canReviewDownload(job) ? <a className="nimautoDownload" href={`${BASE}/nimbo/auto/jobs/${job.job_id}/package`}><Download size={17}/> Download checksum-verified ONNX review ZIP <ExternalLink size={15}/></a> : <p className="nimautoNoRelease"><FileArchive size={16}/> No package available until offline verification succeeds. No production approval.</p>}
        </> : <p className="nimautoEmpty">Enter a previously issued job ID or start a new job. Status appears only after the API responds.</p>}
      </div>
    </div>
    <div className="nimautoCard nimautoEvidence"><h2>Backend event stream <span>{events.length} actual events</span></h2>{events.length ? <ol>{events.map(e => <li key={e.seq}><time>{e.at || ''}</time><b>{e.stage?.toUpperCase()} · {e.state}</b><span>{e.message}</span>{e.evidence && <code>{JSON.stringify(e.evidence)}</code>}</li>)}</ol> : <p>No recorded events loaded yet.</p>}</div>
    <footer>Local M7a only · Source model may have seen public evaluation data · Warm forward timing is not electrical energy, FLOPs or cloud billing. INT8 remains separate and unapproved.</footer>
  </section>;
}
