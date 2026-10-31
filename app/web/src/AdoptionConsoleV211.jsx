// NIMBO_V21_1_CONTROLLED_ADOPTION — local-only model trial; never changes /api/run.
import React, {useCallback, useEffect, useState} from 'react';
import {Activity, ArrowRight, CheckCircle2, CircleAlert, GitCompareArrows, RotateCcw, ShieldCheck, Zap} from 'lucide-react';
import './adoption-v21-1.css';
const URL='http://127.0.0.1:8000/nimbo/adoption/v21';
const EXAMPLES=[
  'The customer support team solved my problem quickly.',
  'I waited for hours and the service was disappointing.',
  'The product is simple, reliable, and pleasant to use.',
  'The application kept crashing after each update.',
  'The experience was fine, but the response time was slow.'
];
function fmt(x, digits=1){return typeof x==='number'&&Number.isFinite(x)?x.toFixed(digits):'—'}
export default function AdoptionConsoleV211(){
  const [data,setData]=useState(null);
  const [offline,setOffline]=useState(false);
  const [working,setWorking]=useState(false);
  const [error,setError]=useState('');
  const [latest,setLatest]=useState(null);
  const [parity,setParity]=useState(null);
  const [text,setText]=useState('The customer support team solved my problem quickly.');
  const [readRisk,setReadRisk]=useState(false);
  const [expanded,setExpanded]=useState(false);
  const refresh=useCallback(async()=>{
    try{const r=await fetch(URL+'/status',{cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status}`);setData(await r.json());setOffline(false)}
    catch(e){setOffline(true);setError(`Adoption service unavailable: ${e.message}. Restart the Python backend.`)}
  },[]);
  useEffect(()=>{refresh()},[refresh]);
  async function send(path,body){const r=await fetch(URL+path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});const raw=await r.json().catch(()=>({}));if(!r.ok)throw Error(typeof raw.detail==='string'?raw.detail:`HTTP ${r.status}`);return raw}
  async function action(name,body){setWorking(true);setError('');try{const result=await send(name,body);setData(result);setLatest(null)}catch(e){setError(e.message)}finally{setWorking(false);await refresh()}}
  async function classify(input=text){if(!input.trim())return;setWorking(true);setError('');try{setLatest(await send('/classify',{text:input.trim()}))}catch(e){setError(e.message)}finally{setWorking(false);await refresh()}}
  async function directParity(){
    if(!text.trim())return;
    setWorking(true);setError('');setParity(null);
    try{setParity(await send('/parity',{text:text.trim()}))}
    catch(e){setError(e.message)}
    finally{setWorking(false);await refresh()}
  }
  async function shadowBatch(){setWorking(true);setError('');try{
    for(const sentence of EXAMPLES) setLatest(await send('/classify',{text:sentence}));
  }catch(e){setError(e.message)}finally{setWorking(false);await refresh()}}
  const reviews=data?.reviews||[];const candidate=reviews[0];const s=data?.state||{};const mode=s.mode||'baseline';const m=data?.shadow_metrics||{};
  const eligible=mode==='shadow'&&s.shadow_count>=5&&s.shadow_count===s.shadow_agreement&&m.reduction_fraction>=.05;
  return <section className="nim211Section" aria-labelledby="nim211-title">
    <div className="nim211Header">
      <div><div className="nim211Eyebrow"><ShieldCheck size={17}/> MODEL ADOPTION / LOCAL LAB</div><h2 id="nim211-title">Review. Test. Switch back.</h2>
        <p>Take the verified ONNX candidate beyond the benchmark, without changing the application's production route.</p></div>
      <button type="button" className="nim211Soft" onClick={refresh}><RotateCcw size={17}/> Refresh</button>
    </div>
    {offline?<div role="alert" className="nim211Error">{error}</div>:<>
      <div className="nim211Kpis">
        <div><span>Candidate review</span><strong>{candidate?'Ready for local trial':'Not staged'}</strong><small>{candidate?`${candidate.sample_count} heldout · ${candidate.rounds} rounds`:'Complete and export v21 validation first'}</small></div>
        <div><span>Local execution route</span><strong>{mode==='baseline'?'PyTorch FP32':mode==='shadow'?'Shadow comparison':'ONNX canary'}</strong><small>{mode==='shadow'?'Baseline remains served':mode==='canary'?'Candidate served locally; 1-in-5 safety check':'Original implementation'}</small></div>
        <div><span>Paired checks</span><strong>{s.shadow_agreement||0} / {s.shadow_count||0}</strong><small>{mode==='canary'?`Initial guard window: ${Math.min(5,s.v22_guarded_count||0)}/5`:'Same-model agreement checks'}</small></div>
        <div><span>Paired pipeline difference</span><strong>{m.reduction_fraction!=null?`${fmt(m.reduction_fraction*100)}%`:'Not measured'}</strong><small>{m.paired_samples||0} paired local requests · tokenizer + CPU</small></div>
      </div>
      <div className="nim211Body">
        <div className="nim211Control">
          <h3>Controlled rollout</h3>
          <div className="nim211Steps"><span className={mode==='baseline'?'current':'finished'}>01 · Review</span><span className={mode==='shadow'?'current':mode==='canary'?'finished':''}>02 · Shadow</span><span className={mode==='canary'?'current':''}>03 · Local canary</span></div>
          {candidate&&<p className="nim211Identity"><b>Candidate</b> lvwerra/distilbert-imdb · ONNX FP32 <small>Run: {candidate.run_id}</small></p>}
          <label className="nim211Optin"><input type="checkbox" checked={readRisk} onChange={e=>setReadRisk(e.target.checked)}/><span>I understand: this changes only the local test endpoint, not NIMBO production routing. Shadow runs both models and saves no compute.</span></label>
          <div className="nim211Buttons">
            {mode==='baseline'&&<button type="button" disabled={!candidate||!readRisk||working} className="nim211Primary" onClick={()=>action('/shadow',{run_id:candidate.run_id,confirmation:'START LOCAL SHADOW'})}>Start shadow comparison <ArrowRight size={16}/></button>}
            {mode==='shadow'&&<button type="button" disabled={!eligible||!readRisk||working} className="nim211Primary" onClick={()=>action('/canary',{run_id:s.run_id,confirmation:'ACTIVATE LOCAL CANARY'})}>Activate local ONNX canary <Zap size={16}/></button>}
            {mode!=='baseline'&&<button type="button" disabled={working} className="nim211Danger" onClick={()=>action('/rollback',{confirmation:'ROLL BACK TO BASELINE'})}><RotateCcw size={16}/> Roll back to PyTorch</button>}
          </div>
          {mode==='shadow'&&<p className="nim211Guidance">Canary unlocks after five paired checks, 100% model agreement and at least 5% lower tokenizer + inference latency in shadow. Current data: {s.shadow_count||0} / 5 checks.</p>}
          {mode==='canary'&&<p className="nim211Guidance"><b>Safety monitoring:</b> the first five Canary requests are paired with PyTorch. Afterward, every fifth request is checked. Candidate-only requests are <b>not</b> verified individually. Any observed label mismatch or score drift greater than 2.5 percentage points triggers a local rollback.</p>}
        </div>
        <div className="nim211Trial">
          <h3>Live local request</h3><p>Enter an IMDb-style English sentiment sample and see which model actually served it.</p>
          <textarea aria-label="Local trial request" value={text} onChange={e=>setText(e.target.value)} maxLength={2000} rows={3}/>
          <div className="nim211Buttons"><button type="button" disabled={working||!candidate||!text.trim()} className="nim211Primary" onClick={()=>classify()}>Run real inference <Activity size={16}/></button>
            <button type="button" disabled={working||!candidate||!text.trim()} className="nim211Soft" onClick={directParity}><GitCompareArrows size={16}/> Check both models</button>
            {mode==='shadow'&&<button type="button" disabled={working} className="nim211Soft" onClick={shadowBatch}>Run 5 shadow samples</button>}</div>
          <p className="nim22Hint">Parity check: runs original PyTorch and optimized ONNX on identical tokens. It diagnoses differences but does not save compute or prove accuracy.</p>
          {parity&&<div className={`nim22Parity ${parity.parity_pass?'nim22Pass':'nim22Fail'}`} aria-live="polite">
            <div className="nim22ParityHeading"><ShieldCheck size={18}/><strong>{parity.parity_pass?'Same-input parity passed':`Safety issue: ${parity.status.replaceAll('_',' ')}`}</strong></div>
            <div className="nim22Compare"><div><span>Original PyTorch</span><b>{parity.baseline.label}</b><small>Confidence {fmt(parity.baseline.confidence*100)}%</small></div><div><span>Optimized ONNX</span><b>{parity.candidate.label}</b><small>Confidence {fmt(parity.candidate.confidence*100)}%</small></div></div>
            <p>Same {parity.token_count} input tokens · positive-class probability delta {fmt(parity.positive_probability_delta*100,3)} percentage points</p>
            {parity.auto_rollback&&<p><b>Automatically restored the PyTorch local route.</b></p>}
          </div>}
          {latest&&<div className="nim211Output" aria-live="polite"><span><CheckCircle2 size={18}/> Measured local result</span><strong>{latest.result_label}</strong><p>Served by <b>{latest.route_served==='candidate_onnx'?'ONNX Runtime FP32':'PyTorch FP32'}</b> · {fmt(latest.served_pipeline_ms)} ms tokenizer + inference · {fmt(latest.request_wall_ms)} ms local request wall time</p>
            {latest.comparison_performed&&<p>Measured pair: PyTorch {fmt(latest.baseline?.tokenizer_plus_forward_ms)} ms · ONNX {fmt(latest.candidate?.tokenizer_plus_forward_ms)} ms</p>}
            <div className="nim22RunStatus">{latest.parity_status==='PASS'?<span className="nim22Good">Paired safety check: PASS</span>:
              latest.parity_status==='LABEL_MISMATCH'||latest.parity_status==='SCORE_DRIFT'?<span className="nim22Bad">Safety issue: {latest.parity_status.replaceAll('_',' ')}</span>:
              <span>Unpaired {mode==='canary'?'Canary request':'request'} — this individual output has not been compared.</span>}</div>
            {latest.parity_checked&&<p>PyTorch {latest.baseline?.label} · ONNX {latest.candidate?.label} {typeof latest.probability_delta==='number'?`· probability delta ${fmt(latest.probability_delta*100,3)} pp`:''}</p>}
            {latest.auto_rollback&&<p className="nim211Error">Safety rollback: {latest.rollback_reason}</p>}</div>}
        </div>
      </div>
      {!!data?.recent_requests?.length&&<div className="nim211History"><button type="button" onClick={()=>setExpanded(!expanded)} className="nim211HistoryToggle">{expanded?'Hide':'Show'} last local test measurements ({data.recent_requests.length})</button>
        {expanded&&<div className="nim211HistoryRows">{[...data.recent_requests].reverse().map((r,i)=><div key={r.timestamp_utc+i}><span>{r.mode} · {r.route_served}</span><span>PyTorch: {fmt(r.baseline_pipeline_ms)} ms</span><span>ONNX: {fmt(r.candidate_pipeline_ms)} ms</span><span>{r.safety_check==='PROBABILITY_DRIFT'?'Score drift':r.safety_check==='MISMATCH'?'Label mismatch':r.parity_verified?'Parity passed':r.prediction_agreement==null?'Not checked':r.prediction_agreement?'Agree':'Mismatch'}</span></div>)}</div>}</div>}
      {error&&<div className="nim211Error" role="alert"><CircleAlert size={16}/> {error}</div>}
    </>}
    <p className="nim211Footer">This is a local opt-in trial, not production deployment. Canary sampling cannot guarantee every unpaired prediction. Warm benchmark improvements and request-level performance are different measurements; neither proves energy or billing savings.</p>
  </section>
}
