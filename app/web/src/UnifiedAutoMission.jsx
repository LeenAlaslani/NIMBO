/** Model optimization and recorded playback on the Garden Mission Floor. */
import React,{useEffect,useMemo,useRef,useState} from 'react';
import {ArrowLeft, BookOpen, Camera, Download, Expand, Info, Play, RotateCcw, ShieldAlert, X} from 'lucide-react';
import {realEvents,sceneFromTimeline} from './autoOptimizer3DState.mjs';
import {canReviewDownload} from './autoOptimizerJobView.mjs';
import {eventWorker,workerDialogue,outcome,finalJob} from './livingLabPlan.mjs';
import './living-lab.css';
import NimboGardenWorld from './NimboGardenWorld.jsx'; // M13_GARDEN_WORLD
import {nextRecordedSeq,m12PickWorker} from './nimboM12Replay.mjs';
import {isAtRecordedEnd,replayEventsUntil} from './nimboGardenReplay.mjs';
import {gardenEvidence} from './nimboGardenEvidence.mjs';
import NimboResourceMonitor from './NimboResourceMonitor.jsx';
import ReviewerEvidence from './ReviewerEvidence.jsx';
const JOB=/^[0-9a-f]{32}$/;
const REVIEWER_API='/api/reviewer/jobs';
const REVIEWER_JOB='nimbo-reviewer-active-job';
const NAMES={captain:'CAPTAIN',toki:'TOKI',routa:'ROUTA',model:'MAKO',forge:'FORGE',vera:'VERA',metri:'METRI',bundle:'BUNDLE'};
async function request(path,opts){const response=await fetch(path,{cache:'no-store',...opts});let value=await response.json().catch(()=>({}));if(!response.ok)throw Object.assign(Error(typeof value.detail==='string'?value.detail:`Optimizer HTTP ${response.status}`),{status:response.status});return value}
function rememberReviewerJob(value){
 try{
  if(value===undefined)return sessionStorage.getItem(REVIEWER_JOB);
  if(value===null)sessionStorage.removeItem(REVIEWER_JOB);
  else sessionStorage.setItem(REVIEWER_JOB,value);
 }catch{return null}
}
export default function UnifiedAutoMission({config,onBack,theme='night',reviewer=false}){
 const [job,setJob]=useState(null),[events,setEvents]=useState([]),[id,setId]=useState(''),[oldId,setOldId]=useState('');
 const [error,setError]=useState(''),[submitting,setSubmitting]=useState(false),[camera,setCamera]=useState('room');
 const [jobApi,setJobApi]=useState('/nimbo/agentic/jobs'),[agenticMode,setAgenticMode]=useState('checking');
 const [selectedSeq,setSelectedSeq]=useState(null),[dialogue,setDialogue]=useState(null),[journal,setJournal]=useState(false),[expanded,setExpanded]=useState(false);
 // NIMBO_M12_REPLAY_FIXED
 const [m12Playing,setM12Playing]=useState(false);
 const [m12Speed,setM12Speed]=useState(1);
 const [resource,setResource]=useState(null);
 const [resourceHistory,setResourceHistory]=useState([]);
 const [resourceError,setResourceError]=useState('');
 const runRef=useRef(0);
 const gardenFullRef=useRef(null);
 useEffect(()=>{
   const sync=()=>setExpanded(
     document.fullscreenElement===gardenFullRef.current
   );
   document.addEventListener('fullscreenchange',sync);
   return()=>document.removeEventListener('fullscreenchange',sync);
 },[]);
 useEffect(()=>{setId('');setJob(null);setEvents([]);setDialogue(null);setSelectedSeq(null);setResource(null);setResourceHistory([]);setResourceError('');setError('')},[config]);
 useEffect(()=>{
  if(!reviewer)return;
  const controller=new AbortController();
  setJobApi(REVIEWER_API);
  setAgenticMode(config.replay?'historical':'standard');
  if(config.replay){
   request('/api/reviewer/replay',{signal:controller.signal}).then(record=>{
    setId(record.job.job_id);setJob(record.job);setEvents(realEvents(record.events));setM12Playing(true);
   }).catch(err=>{if(!controller.signal.aborted)setError(err.message)});
  }else{
   const saved=rememberReviewerJob();
   if(JOB.test(saved||''))setId(saved);
  }
  return()=>controller.abort();
 },[config,reviewer]);
 useEffect(()=>{
  if(!id||(reviewer&&config.replay))return;
  const controller=new AbortController();let handle;const token=++runRef.current;
  const poll=async()=>{try{
   const readStatus=request(`${jobApi}/${id}`,{signal:controller.signal});
   const [j,e]=reviewer?[await readStatus,null]:await Promise.all([readStatus,request(`${jobApi}/${id}/events`,{signal:controller.signal})]);
   if(controller.signal.aborted||runRef.current!==token)return;
   setJob(j);setEvents(realEvents(reviewer?j.events:e.events));setError('');
   if(!finalJob(j))handle=setTimeout(poll,1800);
  }catch(err){if(!controller.signal.aborted){
   setError(err.message);
   if(reviewer&&err.status===404){rememberReviewerJob(null);setId('');setJob(null);setEvents([])}
   else handle=setTimeout(poll,4000);
  }}};
  poll();return()=>{controller.abort();clearTimeout(handle);runRef.current++};
 },[id,jobApi,reviewer,config.replay]);
 async function start(){if(id||submitting)return;setSubmitting(true);setError('');try{
  const body=reviewer?{case:config.id}:{model_id:config.model_id,revision:config.revision,pilot_file:config.pilot_file,heldout_file:config.heldout_file,preprocess:config.preprocess,objective:'speed'};
  const preflight=reviewer?{ready:false}:await request('/nimbo/agentic/ready').catch(()=>({ready:false}));
  const path=reviewer?REVIEWER_API:preflight.ready?'/nimbo/agentic/jobs':'/nimbo/auto/jobs';
  setJobApi(path);setAgenticMode(preflight.ready?'agentic':'standard');
  const response=await fetch(path,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await response.json().catch(()=>({}));
  if(response.status!==202||!JOB.test(data.job_id||''))throw Error(typeof data.detail==='string'?data.detail:`Job not accepted: HTTP ${response.status}`);
  if(reviewer)rememberReviewerJob(data.job_id);
  setResource(null);setResourceHistory([]);setResourceError('');
  setId(data.job_id);setJob(null);setEvents([]);setSelectedSeq(null);
 }catch(err){setError(err.message)}finally{setSubmitting(false)}}
 function openExisting(e){e.preventDefault();if(!JOB.test(oldId.trim())){setError('Enter a valid 32-character job ID.');return}setJobApi('/nimbo/agentic/jobs');setAgenticMode('historical');setM12Playing(true);setM12Speed(1);setId(oldId.trim());setResource(null);setResourceHistory([]);setResourceError('');setJob(null);setEvents([]);setSelectedSeq(null);setDialogue(null);setError('')}

 // M12.4: observe live processes; Replay never invents historical utilization.
 useEffect(()=>{
   if(!id || agenticMode==='historical')return;
   const controller=new AbortController();let timer;
   const poll=async()=>{
     try{
       const response=await request(`${jobApi}/${id}/resources`,{signal:controller.signal});
       if(controller.signal.aborted)return;
       setResource(response);setResourceError('');
       if(response.monitor_state==='live'){
         setResourceHistory(before=>[...before,{
           cpu:response.job_cpu_percent_one_core,
           ram:response.job_rss_mib,
           at:response.sampled_at
         }].slice(-60));
       }
     }catch(err){if(!controller.signal.aborted)setResourceError(err.message)}
     if(!controller.signal.aborted && !(job&&finalJob(job)))timer=setTimeout(poll,2500);
   };
   poll();
   return()=>{controller.abort();clearTimeout(timer)};
 },[id,jobApi,agenticMode,job?.state]);

 const timeline=useMemo(()=>realEvents(events),[events]);
 const replaying=agenticMode==='historical';
 const visibleEvents=useMemo(()=>replayEventsUntil(timeline,selectedSeq,replaying),[timeline,selectedSeq,replaying]);
 const visibleJob=isAtRecordedEnd(timeline,selectedSeq,replaying)?job:null;
 const scene=sceneFromTimeline(visibleJob,visibleEvents,replaying?null:selectedSeq);

 useEffect(()=>{
   if(agenticMode!=='historical' || !m12Playing ||
      !timeline.length)return;

   const next=nextRecordedSeq(timeline,selectedSeq);

   if(next===null){
     setM12Playing(false);
     return;
   }

   const previous=timeline.find(e=>e.seq===selectedSeq);
   const upcoming=timeline.find(e=>e.seq===next);

   const sameWorker=previous &&
     m12PickWorker(previous,timeline,eventWorker)===
     m12PickWorker(upcoming,timeline,eventWorker);

   const delay=selectedSeq===null ? 0 :
     Math.max(450,Math.round(
       (sameWorker?1800:5200)/m12Speed
     ));

   const timer=setTimeout(()=>setSelectedSeq(next),delay);
   return ()=>clearTimeout(timer);
 },[agenticMode,m12Playing,m12Speed,timeline,selectedSeq]);

 const current=scene.event;const worker=m12PickWorker(current,visibleEvents,eventWorker)||'captain';
 const checks=useMemo(()=>gardenEvidence(visibleEvents),[visibleEvents]);
 const status=replaying&&!visibleJob?{
   kind:'running',title:'Recorded replay',description:'The outcome appears at the final recorded event.'
 }:visibleJob?.decision==='AGENT_EXECUTION_FAILED'&&checks.package?{
   kind:'failed',title:'Agent review failed',description:'The optimizer verified a review package before agent execution failed.'
 }:outcome(visibleJob);
 const canDownload=canReviewDownload(visibleJob);
 const inspectedEvent=dialogue?visibleEvents.findLast(e=>m12PickWorker(e,visibleEvents,eventWorker)===dialogue):null;
 const bubble=dialogue?workerDialogue(dialogue,inspectedEvent||null,visibleJob):null;
 function inspectWorker(name){setDialogue(name)}
 function chooseRecord(e){setM12Playing(false);setSelectedSeq(e.seq);setDialogue(m12PickWorker(e,timeline,eventWorker));setJournal(false)}
 const missionActive=Boolean(current);const running=Boolean(!replaying&&job?.state==='RUNNING'&&!scene.historical);

 function toggleGardenFullscreen(){
   if(document.fullscreenElement===gardenFullRef.current){
     document.exitFullscreen?.().catch(()=>setExpanded(false));
   }else if(gardenFullRef.current?.requestFullscreen){
     gardenFullRef.current.requestFullscreen().catch(
       ()=>setExpanded(v=>!v)
     );
   }else{
     setExpanded(v=>!v);
   }
 }
 return <div ref={gardenFullRef} className={'nimLiving '+(expanded?'nimLivingExpanded':'')}>
  <div className="nimLivingTop"><button type="button" onClick={onBack}><ArrowLeft size={16}/> Workspace</button>
   <span className="nimLivingKicker">{reviewer?'NIMBO · PUBLIC REVIEWER':agenticMode==='agentic'?'NIMBO · MULTI-AGENT MODEL FORGE':agenticMode==='standard'?'NIMBO · STANDARD MODEL FORGE':'NIMBO · LIVING LABORATORY / MODEL FORGE'}</span><div className="nimLivingTopActions">
   <button type="button" onClick={()=>setCamera(v=>v==='room'?'follow':'room')}><Camera size={16}/> {camera==='room'?'Follow PICO':'Room view'}</button>
   <button type="button" onClick={()=>setJournal(v=>!v)}><BookOpen size={16}/> Mission journal {visibleEvents.length?`(${visibleEvents.length})`:''}</button>
   <button type="button" onClick={toggleGardenFullscreen}><Expand size={16}/> {expanded?'Normal':'Expand'}</button></div></div>

  {agenticMode==='historical'&&
    <div className="nimLivingReplayBar"
         role="group" aria-label="Recorded mission playback">
      <strong>RECORDED REPLAY · NOT LIVE</strong>

      <span>
        {timeline.length
          ? `${Math.max(0,timeline.findIndex(
              e=>e.seq===selectedSeq
            )+1)} / ${timeline.length} events`
          : 'Loading events'}
      </span>

      <button type="button" disabled={!timeline.length}
        onClick={()=>{
          if(m12Playing){
            setM12Playing(false);
          }else{
            if(selectedSeq===timeline.at(-1)?.seq)
              setSelectedSeq(null);
            setM12Playing(true);
          }
        }}>
        {m12Playing?'Pause':'Play'}
      </button>

      <button type="button" disabled={!timeline.length}
        onClick={()=>{
          setSelectedSeq(null);
          setDialogue(null);
          setM12Playing(true);
        }}>
        Restart
      </button>

      <label>
        Speed
        <select value={m12Speed}
          aria-label="Replay speed"
          onChange={e=>setM12Speed(Number(e.target.value))}>
          <option value={1}>1×</option>
          <option value={2}>2×</option>
          <option value={4}>4×</option>
        </select>
      </label>

      {current&&<small>
        #{current.seq} · {NAMES[worker]} ·
        {current.stage} · {current.state}
      </small>}
    </div>}

  <NimboResourceMonitor jobId={id} historical={agenticMode==='historical'} reviewer={reviewer}
   data={resource} history={resourceHistory} error={resourceError}/>

  <div className="nimLivingWorld">
   <NimboGardenWorld target={worker} onSelect={inspectWorker} busy={running}
    theme={theme} liveExecute={running} started={missionActive} followWorker={camera==='follow'}
    modelMission={{events:timeline,mode:agenticMode}} selectedRecordSeq={selectedSeq}/>
   <aside className="nimLivingHUD" aria-live="polite"><span>✦ PICO · YOUR MISSION</span>
    <h2>{missionActive?(agenticMode==='historical'?`Recorded stage · ${NAMES[worker]}`:` ${NAMES[worker]} is at work`):'Welcome to Model Forge'}</h2>
    <p>{missionActive?String(current.message||'Backend stage recorded').slice(0,190):'I will guide your model through the laboratory. Each glowing station shows a real step, not simulated compute.'}</p>
    <div className={'nimLivingStatus '+status.kind}><b>{status.title}</b><small>{status.description}</small></div>
    {!id&&!replaying?<button className="nimLivingStart" type="button" disabled={submitting} onClick={start}><Play size={16}/>{submitting?'Submitting real job…':reviewer?'Run Real Optimization':'Begin real mission'}</button>:<div className="nimLivingJob">{id&&<>Job <code>{id.slice(0,8)}…</code> · </>}{visibleJob?.state||(replaying?'Recorded playback':'Connecting')}</div>}
    {error&&<p className="nimLivingError" role="alert">{error}</p>}
   </aside>
   <div className="nimLivingPico">✦ {missionActive?`PICO leads you to ${NAMES[worker]}`:'PICO is ready to guide you'}</div>
   {bubble&&<div className="nimLivingTalk" role="dialog" aria-label={`${bubble.name} briefing`}><div><b>{bubble.name} · {bubble.recorded?'RECORDED EVIDENCE':'STATION BRIEFING'}</b><button type="button" onClick={()=>setDialogue(null)} aria-label="Close worker dialogue"><X size={16}/></button></div><h3>{bubble.title}</h3><p>{bubble.message}</p>{bubble.recorded&&<small>Explanation based on recorded backend event. This does not trigger new inference.</small>}</div>}
   {journal&&<div className="nimLivingJournal"><div className="nimLivingJournalTitle"><b>Mission Journal · Actual backend events</b><button type="button" onClick={()=>setJournal(false)}><X size={16}/></button></div>
    {!visibleEvents.length?<p>No recorded backend events yet.</p>:visibleEvents.map(e=><button type="button" key={e.seq} onClick={()=>chooseRecord(e)}><strong>#{e.seq} · {NAMES[m12PickWorker(e,visibleEvents,eventWorker)]||'SYSTEM'} · {e.stage}</strong><span>{e.message}</span></button>)}
    <small>Opening history never runs inference or changes the job outcome.</small></div>}
  </div>
  <div className="nimLivingBottom"><div className="nimLivingOutcome"><b>{status.title}</b><span>{visibleJob?.decision||(replaying?'Playback reveals the recorded decision in sequence.':'Backend will decide using measured quality and CPU gates.')}</span></div>
   <div className="nimLivingActions">{scene.historical&&<button type="button" onClick={()=>{setSelectedSeq(null);setDialogue(null)}}><RotateCcw size={15}/> Latest live event</button>}
   {canDownload&&<a href={`${jobApi}/${id}/package`}><Download size={16}/> Verified review package</a>}
   {!id&&!reviewer&&<form onSubmit={openExisting}><input aria-label="Existing job ID" placeholder="Existing 32-character job ID" value={oldId} onChange={e=>setOldId(e.target.value)} maxLength={32}/><button type="submit">Replay a real job</button></form>}</div></div>
  {reviewer&&!replaying&&<ReviewerEvidence job={job}/>}
  <p className="nimLivingCaveat"><Info size={14}/> {reviewer?(replaying?'Historical recording. No new compute is running.':'Hosted deterministic CPU optimization. Optional Qwen agents are unavailable. Packages are for review.'):agenticMode==='agentic'?'LLM agents are making bounded tool decisions.':agenticMode==='standard'?'No local reasoning model connected; the original deterministic engine is running.':'Historical tasks preserve their recorded execution mode.'} {!reviewer&&'Supported public text classifiers on local CPU. Automatic ONNX FP32 speed optimization only. Results are review-only; no production, energy, FLOPs or billing claim.'}</p>
 </div>;
}
