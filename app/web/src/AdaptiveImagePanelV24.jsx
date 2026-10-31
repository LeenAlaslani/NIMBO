// NIMBO_VERA_V26_AUTONOMOUS: one real Fast request, automatic conservative VERA decision.
// After Fast the backend checks, falls back and records; the UI only follows real stages.
import React, {useEffect, useRef, useState} from 'react';
import './nimbo-vera-v26.css';
const API='http://127.0.0.1:8000';
const fmt=(v)=>v===null||v===undefined||!Number.isFinite(Number(v))?'—':`${(Number(v)/1000).toFixed(2)} s`;
const STATES={
  queued:'Waiting for the local VERA worker',
  checking_fast:'Checking the Fast image',
  generating_standard:'Fast could not be verified · generating Standard',
  checking_standard:'Checking the Standard image',
  complete:'Autonomous image decision recorded',
  error:'Verification failed safely'
};
const DECISIONS={
  FAST_ACCEPTED_LOCAL_CHECK:{title:'Fast accepted by local checks',tone:'pass',description:'Standard was not generated. This is a local detector decision, not an independent quality certification.'},
  STANDARD_ACCEPTED_LOCAL_CHECK:{title:'Standard accepted by local checks',tone:'pass',description:'Fast did not pass the supported local checks. Standard was generated and checked; certification is still limited.'},
  FAST_RETURNED_UNVERIFIED:{title:'Fast returned · quality NOT verified',tone:'warning',description:'No proof of required scene details. A second generation was not run without credible evidence it would improve the result. This is not a quality pass.'},
  STANDARD_DELIVERED_UNVERIFIED:{title:'Best available fallback · quality unverified',tone:'warning',description:'Standard was generated automatically, but VERA cannot establish all prompt requirements. The image is provided without claiming task success.'},
  NO_VERIFIED_RESULT:{title:'Required elements were not established',tone:'fail',description:'Both generation paths ran. The final local detector did not establish success; no successful result is claimed.'}
};
async function request(url,method='GET',payload){
 const response=await fetch(API+url,{method,headers:{'Content-Type':'application/json'},body:payload?JSON.stringify(payload):undefined});
 const data=await response.json().catch(()=>({}));
 if(!response.ok)throw new Error(typeof data.detail==='string'?data.detail:`HTTP ${response.status}`);
 return data;
}

export default function AdaptiveImagePanelV24({report,prompt,onUpdate,headless=false,onStatus}){
 const [job,setJob]=useState(null),[error,setError]=useState('');
 const [runningId,setRunningId]=useState('');
 const [busy,setBusy]=useState(false);
 const latest=useRef({report,prompt,onUpdate});
 const startOnce=useRef('');
 const delivered=useRef('');
 const onStatusRef=useRef(onStatus);onStatusRef.current=onStatus;
 latest.current={report,prompt,onUpdate};
 const id=report?.run_id;
 const fast=report?.result?.fast_image_base64 || (report?.adaptive?.auto_selected==='fast'?report?.result?.image_base64:null);
 const v26Report=report?.adaptive?.auto_complete && report?.route==='sd15_adaptive_autonomous_vera_v26';

 useEffect(()=>{setJob(null);setError('');setRunningId('');setBusy(false);startOnce.current='';delivered.current='';},[id]);
 const mergeReport=(incoming)=>{
   const old=latest.current.report;
   if(!old||!incoming)return;
   latest.current.onUpdate({...old,...incoming,
      result:{...(old.result||{}),...(incoming.result||{})},
      adaptive:{...(old.adaptive||{}),...(incoming.adaptive||{})},
      metrics:{...(old.metrics||{}),...(incoming.metrics||{})},
      image_comparison:{...(old.image_comparison||{}),...(incoming.image_comparison||{})},
      trace:[...(old.trace||[]),...(incoming.trace||[])]});
 };
 useEffect(()=>{
   if(!id||!fast||!prompt||v26Report||!report?.adaptive)return;
   if(startOnce.current===id)return;
   startOnce.current=id;
   let disposed=false;
   setBusy(true);setError('');
   (async()=>{
      try{
       const reply=await request('/nimbo/image/vera/v26/launch','POST',{
         run_id:id,prompt,fast_image_base64:fast});
       if(disposed)return;
       setJob(reply);setRunningId(id);
      }catch(e){if(!disposed){setError(`VERA did not start: ${e.message}`);setBusy(false);}}
   })();
   return()=>{disposed=true};
 // Always start only once per run; incoming report metrics can change on updates.
 // eslint-disable-next-line react-hooks/exhaustive-deps
 },[id,fast,prompt,v26Report]);

 useEffect(()=>{
   if(!runningId||v26Report)return;
   let alive=true, pending=false;
   const poll=async()=>{
     if(!alive||pending)return;
     pending=true;
     try{
       const reply=await request(`/nimbo/image/vera/v26/status/${encodeURIComponent(runningId)}`);
       if(!alive)return;
       setJob(reply);
       if(reply.state==='complete'){
         setBusy(false);setRunningId('');
         if(delivered.current!==runningId){delivered.current=runningId;mergeReport(reply.report);}
       }else if(reply.state==='error'){
         setBusy(false);setRunningId('');setError(reply.failure||'VERA could not complete this run. No quality success was recorded.');
       }
     }catch(e){if(alive){setBusy(false);setRunningId('');setError(`Could not retrieve VERA result: ${e.message}. Reload the page to check the live API; no result is fabricated.`);}}
     finally{pending=false;}
   };
   poll();const timer=setInterval(poll,1250);
   return()=>{alive=false;clearInterval(timer)};
 // eslint-disable-next-line react-hooks/exhaustive-deps
 },[runningId,v26Report]);

 useEffect(()=>{
   const real=job?.state||((v26Report||report?.adaptive?.auto_complete)?'complete':id?'queued':'idle');
   onStatusRef.current?.({state:real,active:real==='queued'||real==='checking_fast'||real==='generating_standard'||real==='checking_standard',
     message:job?.events?.at(-1)?.message||(real==='complete'?'Quality decision recorded':real==='queued'?'Waiting for verification backend':'Waiting'),
     decision:job?.decision||report?.adaptive?.decision,events:job?.events||[]});
 // Status is emitted only when real job state or event count changes.
 // eslint-disable-next-line react-hooks/exhaustive-deps
 },[id,job?.state,job?.events?.length,job?.decision,v26Report,report?.adaptive?.auto_complete]);

 if(!report?.adaptive)return null;
 if(headless)return null;
 const state=job?.state|| (v26Report?'complete':busy?'queued':'queued');
 const decision=v26Report?report.adaptive.decision:job?.decision;
 const outcome=DECISIONS[decision];
 const fastCheck=report.vera_fast;
 const standardCheck=report.vera_standard;
 const stageEvents=job?.events||[];
 const stageSeen=new Set(stageEvents.map(e=>e.stage));
 const totalGPU=report.metrics?.gpu_inference_total_ms;
 const verifyWall=report.metrics?.verifier_total_wall_ms;
 const fullWall=report.metrics?.measured_components_wall_ms;
 const stdRan=report.metrics?.standard_inference_ms!==null&&report.metrics?.standard_inference_ms!==undefined;

 return <section className="nim26Panel" aria-label="VERA autonomous optimization decision">
   <header className="nim26Head"><div><small>VERA / AUTONOMOUS QUALITY AGENT</small><h3>{outcome?.title||STATES[state]||'VERA is preparing'}</h3></div><span className={`nim26Dot ${outcome?.tone||'running'}`}>{state==='complete'?'COMPLETED':state==='error'?'ERROR':'LIVE'}</span></header>
   <p className="nim26Description">VERA uses measured local checks. It only launches Standard after an actionable local failure. Unverifiable scene requirements do not trigger an expensive blind fallback or a false quality pass.</p>
   <div className="nim26Flow" role="status" aria-live="polite">
    {['checking_fast','generating_standard','checking_standard','complete'].map((s,i)=>
     <div key={s} className={`nim26Step ${state===s?'is-current':''} ${stageSeen.has(s==='checking_fast'?'verify':s==='generating_standard'?'execute':s==='checking_standard'?'verify':'deliver')?'observed':''}`}>
      <span>{String(i+1).padStart(2,'0')}</span><b>{STATES[s]}</b>
     </div>)}
   </div>
   {busy&&<p className="nim26Live">Real verification is running on the local CPU. GPU fallback runs only after the Fast decision; this panel will update automatically.</p>}
   {outcome&&<div className={`nim26Outcome ${outcome.tone}`}><strong>{outcome.title}</strong><p>{outcome.description}</p></div>}
   {!busy&&!outcome&&error&&<div className="nim26Outcome fail"><strong>No automatic success recorded</strong><p>{error}</p></div>}
   <div className="nim26Metrics">
    <div><small>Fast GPU inference</small><strong>{fmt(report.metrics?.fast_inference_ms)}</strong></div>
    <div><small>Standard GPU inference</small><strong>{stdRan?fmt(report.metrics?.standard_inference_ms):'Not executed'}</strong></div>
    <div><small>VERA wall time</small><strong>{fmt(verifyWall)}</strong></div>
    <div><small>Executed GPU inference</small><strong>{fmt(totalGPU)}</strong></div>
    <div className="wide"><small>Measured run components (Fast request + agent)</small><strong>{fmt(fullWall)}</strong><span>Frontend handoff excluded; this is not electrical or billing cost.</span></div>
   </div>
   {(fastCheck||standardCheck)&&<div className="nim26Checks">
      <h4>VERA decision evidence</h4>
      {fastCheck&&<div><b>Fast / {fastCheck.verdict}</b><p>{fastCheck.reason}</p><small>Reliable: {fastCheck.boxes_reliable??'—'} · Plausible: {fastCheck.boxes_plausible??'—'} · Detector: {fmt(fastCheck.verification_ms)}</small></div>}
      {standardCheck&&<div><b>Standard / {standardCheck.verdict}</b><p>{standardCheck.reason}</p><small>Reliable: {standardCheck.boxes_reliable??'—'} · Plausible: {standardCheck.boxes_plausible??'—'} · Detector: {fmt(standardCheck.verification_ms)}</small></div>}
   </div>}
   {job?.events?.length>0&&<details className="nim26Details"><summary>Inspect actual backend events ({job.events.length})</summary><ol>{job.events.map((e,i)=><li key={i}><small>{fmt(e.at_ms)}</small> {e.message}</li>)}</ol></details>}
   <p className="nim26Disclaimer">No prompt or PNG is saved in quality evidence. Local OWLv2 checking does not certify complex scene details, exact relationships or independently verified task success. If all requirements cannot be established, the outcome remains unverified rather than asking you to approve it by default.</p>
  </section>;
}
