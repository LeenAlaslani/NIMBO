// NIMBO_V17_EVIDENCE_MODAL
import React, {useEffect, useMemo, useState} from 'react';
import {createPortal} from 'react-dom';
import './evidence-dashboard.css';

const API = 'http://127.0.0.1:8000/nimbo/evidence/unified';
const LABELS = {
  runtime_measurement:'Live runtime',
  offline_labeled_benchmark:'Human-labeled benchmark',
  offline_replay:'Offline replay',
  local_development_benchmark:'Development benchmark',
  local_development_run:'Development run',
  oracle_upper_bound:'Oracle bound — not deployable',
  local_paired_cpu_benchmark:'Paired CPU comparison',
  local_image_gpu_execution:'Local image GPU trial',
};
const formatNumber = (n,d=1)=>typeof n==='number' && Number.isFinite(n)?n.toLocaleString(undefined,{maximumFractionDigits:d}):'Not measured';
const kindOf = r=>r.evidence_kind||'unknown';

export default function EvidenceDashboard({onBack}){
  const [data,setData]=useState(null);
  const [imageEvidence,setImageEvidence]=useState(null);
  const [error,setError]=useState('');
  const [loading,setLoading]=useState(false);
  const [kind,setKind]=useState('all');
  const [workload,setWorkload]=useState('all');
  const [selected,setSelected]=useState(null);
  const load=async()=>{
    setLoading(true);setError('');
    try{
      const response=await fetch(API,{cache:'no-store'});
      if(!response.ok)throw new Error(`HTTP ${response.status}`);
      const json=await response.json();
      if(!Array.isArray(json.records)||!json.runtime_summary)throw new Error('Unexpected evidence API schema');
      setData(json);
    }catch(e){setError('Evidence service unavailable: '+e.message)}finally{setLoading(false)}
  };
  useEffect(()=>{load()},[]);
  useEffect(()=>{fetch('http://127.0.0.1:8000/nimbo/image/v22/evidence').then(r=>r.ok?r.json():null).then(setImageEvidence).catch(()=>{});},[data]);
  useEffect(()=>{
    if(!selected)return;
    const escape=(e)=>{if(e.key==='Escape')setSelected(null)};
    window.addEventListener('keydown',escape);
    return()=>window.removeEventListener('keydown',escape);
  },[selected]);
  const records=data?.records||[];
  const filtered=useMemo(()=>records.filter(r=>(kind==='all'||kindOf(r)===kind)&&(workload==='all'||r.workload===workload)),[records,kind,workload]);
  const categories=[...new Set(records.map(kindOf))];
  const workloads=[...new Set(records.map(r=>r.workload))];
  const runtime=records.filter(r=>kindOf(r)==='runtime_measurement');
  const offline=records.filter(r=>kindOf(r)==='offline_labeled_benchmark');
  const latest=runtime.length?runtime[runtime.length-1]:null;
  return <div className="evRoot">
    <div className="evTop"><div><div className="evEyebrow">NIMBO / EVIDENCE CENTER</div>{onBack&&<button className="nim16BackButton" onClick={onBack}>← Back to Mission Floor</button>}<h2>One source. Clear provenance.</h2><p>Real requests and offline experiments are separate. No invented savings or quality claims.</p></div><button className="evRefresh" onClick={load} disabled={loading}>{loading?'Loading…':'↻ Refresh evidence'}</button></div>
    {error&&<div className="evError" role="alert">{error}<div>Run the main NIMBO backend: <code>python -m uvicorn app.mission_api:app --reload --port 8000</code>.</div></div>}
    <div className="evMetrics">
      <div><small>LIVE API REQUESTS</small><strong>{formatNumber(data?.runtime_summary?.request_count,0)}</strong><span>Actual completed requests; no quality label implied</span></div>
      <div><small>OFFLINE LABELED RECORDS</small><strong>{formatNumber(offline.length,0)}</strong><span>Human-reviewed experiments, not live traffic</span></div>
      <div><small>LATEST LATENCY</small><strong>{latest?formatNumber(latest.latency_ms,2)+' ms':'—'}</strong><span>{latest?latest.workload+' · '+(latest.route||'unknown route'):'No live execution yet'}</span></div>
      <div><small>VERIFIED LIVE SUCCESSES</small><strong>—</strong><span>Requires independent success labels</span></div>
    </div>
    {imageEvidence?.paired_runs>0&&<section className="nim22ImageEvidenceSummary"><strong>Image Optimization · local GPU evidence</strong><p>{imageEvidence.complete_human_paired_reviews} of {imageEvidence.paired_runs} paired image experiments have complete human critical-object assessments (not audited).</p><div><span>Fast · seconds per successful object <b>{imageEvidence.fast?.seconds_per_success??'Not yet measured'}</b></span><span>Standard · seconds per successful object <b>{imageEvidence.standard?.seconds_per_success??'Not yet measured'}</b></span></div><small>Actual GPU inference durations only; no electricity, billing or end-to-end request cost claim.</small></section>}
    <div className="evNotice">Quality certification is not inferred from model confidence. Runtime durations are not financial cost, energy, or GPU compute. Oracle bounds never count as measured achievements.</div>
    <div className="evControls"><label>Evidence class<select value={kind} onChange={e=>setKind(e.target.value)}><option value="all">All evidence classes</option>{categories.map(k=><option value={k} key={k}>{LABELS[k]||k}</option>)}</select></label><label>Workload<select value={workload} onChange={e=>setWorkload(e.target.value)}><option value="all">All workloads</option>{workloads.map(w=><option key={w} value={w}>{w}</option>)}</select></label><span>{filtered.length} records · paired runs are not counted as live requests</span></div>
    <div className="evTableWrap"><table className="evTable"><thead><tr><th>Evidence source</th><th>Workload / route</th><th>Quality</th><th>Timing</th><th>Trust boundary</th><th></th></tr></thead><tbody>{filtered.map(r=><tr key={r.id}>
      <td><b>{r.source||'Unknown source'}</b><small>{LABELS[kindOf(r)]||kindOf(r)}</small>{r.criterion&&<small>Criterion: {r.criterion}</small>}</td>
      <td>{r.workload||'—'}<small>{r.route||'—'}</small></td>
      <td>{typeof r.success_rate==='number'?`${formatNumber(r.success_rate*100,2)}% (${r.successes}/${r.sample_count})`:r.evidence_kind==='local_image_gpu_execution'?(r.human_review_available?'Human-reviewed objects · not audited':'Human review pending'):r.evidence_kind==='local_paired_cpu_benchmark'?(r.all_labels_present?`Supplied labels · ${formatNumber((r.candidate_accuracy_supplied_labels??0)*100)}% candidate`:'No labels'):r.success_label===true?'Verified success':r.success_label===false?'Verified failure':'Not independently labeled'}</td>
      <td>{typeof r.latency_ms==='number'?`${formatNumber(r.latency_ms,2)} ms`:r.evidence_kind==='local_paired_cpu_benchmark'&&typeof r.candidate_mean_inference_ms==='number'?`${formatNumber(r.baseline_mean_inference_ms,2)} → ${formatNumber(r.candidate_mean_inference_ms,2)} ms` : r.evidence_kind==='local_image_gpu_execution'?`${formatNumber(r.standard_inference_ms,1)} → ${formatNumber(r.fast_inference_ms,1)} ms GPU` : typeof r.seconds_per_success==='number'?`${formatNumber(r.seconds_per_success,3)} s / success`:'Not comparable'}</td>
      <td><span className={'evPill '+(kindOf(r)==='oracle_upper_bound'?'evOracle':'')}>{kindOf(r)==='runtime_measurement'?'Measured runtime':kindOf(r)==='local_paired_cpu_benchmark'?'Paired CPU · not audited':kindOf(r)==='oracle_upper_bound'?'Theoretical only':'Offline / development'}</span></td>
      <td><button type="button" className="evDetails" aria-label={`View details for ${r.id}`} onClick={()=>setSelected(r)}>Details</button></td>
    </tr>)}</tbody></table>{!filtered.length&&<p className="evEmpty">No evidence matches these filters.</p>}</div>
    {selected&&createPortal(<div className="evModalBackdrop" onMouseDown={e=>{if(e.target===e.currentTarget)setSelected(null)}}>
      <section className="evModal" role="dialog" aria-modal="true" aria-label="Evidence record details">
       <div className="evModalTop"><div><small>RECORDED EVIDENCE</small><h2>{LABELS[kindOf(selected)]||kindOf(selected)}</h2><p>{selected.source} · {selected.workload||'Unknown workload'}</p></div><button type="button" autoFocus onClick={()=>setSelected(null)}>Close ×</button></div>
       {kindOf(selected)==='local_paired_cpu_benchmark'&&<div className="evPairedDetail">
        <div><small>Baseline model</small><strong>{selected.baseline_model||'Unknown'}</strong><b>{formatNumber(selected.baseline_mean_inference_ms,3)} ms / input</b></div>
        <div><small>Candidate model</small><strong>{selected.candidate_model||'Unknown'}</strong><b>{formatNumber(selected.candidate_mean_inference_ms,3)} ms / input</b></div>
        <div><small>Measured latency difference</small><b>{typeof selected.mean_latency_reduction_fraction==='number'?(100*selected.mean_latency_reduction_fraction).toFixed(1)+'% lower candidate latency':'Not measured'}</b></div>
        <div><small>Label check</small><b>{selected.all_labels_present?'Supplied labels (not audited)':'No complete labels'}</b></div>
       </div>}
       <div className="evModalNote">This record is evidence of one measurement, not a guarantee of future performance. Warm inference latency is distinct from loading and end-to-end time. Accuracy labels are not independently certified.</div>
       <details className="evRawDetails"><summary>Show complete recorded fields</summary><pre>{JSON.stringify(selected,null,2)}</pre></details>
       <div className="evModalActions"><button type="button" onClick={()=>{navigator.clipboard?.writeText(JSON.stringify(selected,null,2)).catch(()=>{});}}>Copy JSON</button><button type="button" onClick={()=>setSelected(null)}>Done</button></div>
      </section>
    </div>,document.body)}
    <div className="evLimit"><h3>Evidence limitations</h3>{(data?.limitations||[]).map((item,i)=><p key={i}>• {item}</p>)}</div>
  </div>;
}
