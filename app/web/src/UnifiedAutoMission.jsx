/** Unified NIMBO mission: old 3D lab, real M7 events, one user journey. */
import React,{useEffect,useMemo,useRef,useState} from 'react';
import { ArrowLeft, ArrowRight, Download, RadioTower, RefreshCcw } from 'lucide-react';
import { realEvents, sceneFromTimeline, FINAL_STATES } from './autoOptimizer3DState.mjs';
import { MISSION_WORKERS as WORKERS, MISSION_FLOW as FLOW, missionWorkerForEvent } from './unifiedWorkerMap.mjs';
import { canReviewDownload } from './autoOptimizerJobView.mjs';
import './unified-nimbo.css';
const API='http://127.0.0.1:8005';
const ID=/^[0-9a-f]{32}$/;
export const terminalJob = state => FINAL_STATES.includes(state);

async function read(path,signal) {
  const response=await fetch(API+path,{signal,cache:'no-store'});
  if(!response.ok)throw new Error(`Optimizer API HTTP ${response.status}`);
  return response.json();
}
export default function UnifiedAutoMission({ config, onBack, theme='night', Scene }) {
  const [job,setJob]=useState(null),[events,setEvents]=useState([]),[id,setId]=useState('');
  const [lookup,setLookup]=useState(''),[error,setError]=useState(''),[submitting,setSubmitting]=useState(false);
  const [selectedSeq,setSelectedSeq]=useState(null),[follow,setFollow]=useState(true),[connected,setConnected]=useState(false);
  const requestSeq=useRef(0);
  useEffect(()=>{setId('');setJob(null);setEvents([]);setSelectedSeq(null);setError('')},[config]);
  useEffect(()=>{
    if(!id)return;
    const controller=new AbortController();let timer;
    const poll=async()=>{
      const seq=++requestSeq.current;
      try{
        const [j,t]=await Promise.all([
          read(`/nimbo/auto/jobs/${id}`,controller.signal),
          read(`/nimbo/auto/jobs/${id}/events`,controller.signal),
        ]);
        if(controller.signal.aborted||seq!==requestSeq.current)return;
        setJob(j);setEvents(realEvents(t.events));setConnected(true);setError('');
        if(!terminalJob(j.state))timer=setTimeout(poll,1800);
      }catch(e){if(!controller.signal.aborted){setConnected(false);setError(`Cannot read the local optimizer on port 8005: ${e.message}`);timer=setTimeout(poll,4000)}}
    };
    poll();
    return()=>{controller.abort();clearTimeout(timer);requestSeq.current++};
  },[id]);
  async function start(){
    if(submitting||id)return;
    setSubmitting(true);setError('');
    try{
      const body={model_id:config.model_id,revision:config.revision,pilot_file:config.pilot_file,
        heldout_file:config.heldout_file,preprocess:config.preprocess,objective:'speed'};
      const response=await fetch(API+'/nimbo/auto/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
      const payload=await response.json().catch(()=>({}));
      if(response.status!==202||!ID.test(payload.job_id||''))throw Error(typeof payload.detail==='string'?payload.detail:`Job was not accepted (HTTP ${response.status})`);
      setId(payload.job_id);setConnected(true);
    }catch(e){setError(e.message)}finally{setSubmitting(false)}
  }
  function openPrevious(e){e.preventDefault();if(!ID.test(lookup.trim())){setError('Enter a valid 32-character job ID.');return}setJob(null);setEvents([]);setSelectedSeq(null);setError('');setId(lookup.trim())}
  const timeline=useMemo(()=>realEvents(events),[events]);
  const scene=sceneFromTimeline(job,timeline,selectedSeq);
  const current=scene.event,worker=missionWorkerForEvent(current,timeline)||'captain',started=Boolean(current);
  const finished=Boolean(job&&terminalJob(job.state));
  const canDownload=Boolean(connected&&canReviewDownload(job));
  function selectWorker(name){const entry=[...timeline].reverse().find(e=>missionWorkerForEvent(e,timeline)===name);if(entry)setSelectedSeq(entry.seq)}
  return <div className="gameShell nimuniMission">
    <div className="nimuniMissionNav"><button type="button" onClick={onBack}><ArrowLeft size={16}/> Back to Workspace</button><span>{config.model_id} · ANALYZE → OPTIMIZE → VERIFY → DELIVER</span></div>
    <div className="nimuniMissionFrame">
      <div className="gameWorldTop"><span><RadioTower size={15}/> LIVE MISSION FLOOR</span><div className="gameWorldTopActions"><span>{job?.state||'READY'}</span><button type="button" onClick={()=>setFollow(v=>!v)}>Camera: {follow?'Follow':'Overview'}</button></div></div>
      <div className="nimuniScene">
        <Scene veraLive={null} target={worker} onSelect={selectWorker} busy={scene.running}
          route={''} comparison={null} imageComparison={null} theme={theme} offline={false}
          liveExecute={scene.running} started={started} onArrive={()=>{}} missionSeed={id||'idle'}
          explainingId={started?worker:null} thinkingId={null}
          speech={current?{title:current.message.slice(0,132)}:null}
          celebrating={Boolean(scene.verifiedDelivery)} metrics={{}}
          events={timeline.map(e=>({stage:e.stage,message:e.message,at_ms:null}))}
          guideStep={FLOW.indexOf(worker)} quality={scene.verifiedDelivery?'VERIFIED_OFFLINE_RELOAD':'NOT_APPROVED'}
          flow={FLOW} followWorker={follow} enhancedWorkers={true}/>
        <div className="nimuniSceneLabel"><b>{current?`${WORKERS[worker]} · ${String(current.stage).toUpperCase()}`:'Workers ready'}</b><span>{current?.message||'Start the real optimization mission to activate the stations.'}</span></div>
      </div>
    </div>
    <section className="nimuniMissionReceipt">
      <div className="nimuniMissionHeader"><div><h2>{job?.state||'Ready to optimize'}</h2><p>{!id?'The robots will show real backend stages, not a timed simulation.':finished?'This mission has a recorded final decision.':'Worker status updates only when the API records an event.'}</p></div>
      {!id?<button type="button" className="nimuniPrimary" onClick={start} disabled={submitting}>{submitting?'Submitting…':'Start mission'} <ArrowRight size={16}/></button>:<button type="button" className="nimuniSecondary" onClick={()=>setSelectedSeq(null)}><RefreshCcw size={15}/> Latest event</button>}</div>
      {error&&<p className="nimuniError" role="alert">{error}</p>}
      <div className="nimuniMissionStages">{FLOW.map((w,i)=>{const stage=['analyze','optimize','verify','deliver'][i];const found=timeline.some(e=>e.stage===stage);return <div key={w} className={found?'seen':''}><strong>{stage.toUpperCase()}</strong><span>{WORKERS[w]} · {found?'Recorded':'Not reached'}</span></div>})}</div>
      {job?.decision&&<p className="nimuniDecision"><strong>Decision: </strong>{job.decision} {job.state==='COMPLETED_REJECTED'?'· Original model preserved':''}</p>}
      {canDownload&&<a className="nimuniDownload" href={`${API}/nimbo/auto/jobs/${id}/package`}><Download size={18}/> Download verified ONNX package (review only)</a>}
      {finished&&!canDownload&&<p className="nimuniHint">No optimized package is approved for download for this job. Original model remains unchanged.</p>}
      {id&&<p className="nimuniJobId">Job ID: <code>{id}</code></p>}
      <details className="nimuniEventHistory"><summary>Backend events ({timeline.length})</summary><div>{timeline.map(e=><button key={e.seq} className={e.seq===current?.seq?'selected':''} onClick={()=>setSelectedSeq(e.seq)} type="button"><b>#{e.seq} · {e.stage} · {e.state}</b><span>{e.message}</span></button>)}</div><p>Clicking an event reviews history; it does not rerun inference.</p></details>
      {!id&&<details className="nimuniPrevious"><summary>Review an existing job</summary><form onSubmit={openPrevious}><input placeholder="32-character job ID" value={lookup} maxLength={32} onChange={e=>setLookup(e.target.value)}/><button type="submit">Load</button></form></details>}
    </section>
    <p className="nimuniDisclaimer">Only supported public text classifiers on local CPU. Review-only ZIP after quality, CPU and offline-reload gates. No production deployment, model-size reduction, energy or billing claims.</p>
  </div>;
}
