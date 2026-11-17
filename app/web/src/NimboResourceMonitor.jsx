/** M12.4: genuine viewer-polled job-process telemetry, NOT a benchmark result. */
import React,{useState} from 'react';
import {Activity,ChevronDown,ChevronUp,Cpu,HardDrive,MemoryStick} from 'lucide-react';
import {cpuLabel,mibLabel,percentLabel,fill,points,measured} from './nimboResourceView.mjs';

function Meter({title,value,ratio,detail}){
 return <div className="nimResourceMeter">
  <div className="nimResourceMetricHead"><span>{title}</span><strong>{value}</strong></div>
  <div className="nimResourceTrack"><span style={{width:`${fill(ratio)}%`}}/></div>
  <small>{detail}</small>
 </div>;
}

function Curve({history,keyName,label}){
 const line=points(history,keyName);
 return <div className="nimResourceCurve">
  <small>{label}</small>
  <svg viewBox="0 0 240 52" preserveAspectRatio="none" role="img" aria-label={`${label} history of measured samples`}>
   {line?<polyline fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinejoin="round" strokeLinecap="round" points={line}/>:null}
  </svg>
 </div>;
}

export default function NimboResourceMonitor({jobId,historical,data,history=[],error=''}){
 const [open,setOpen]=useState(true);
 const live=Boolean(jobId&&!historical&&data?.monitor_state==='live');
 const cpu=data?.job_cpu_percent_one_core;
 const rss=data?.job_rss_mib;
 const peak=data?.observed_peak_job_rss_mib;
 const hostCpu=data?.host_cpu_percent;
 const ram=data?.host_ram_used_mib;
 const ramTotal=data?.host_ram_total_mib;
 const ramUsedPct=data?.host_ram_percent;
 const ollama=data?.ollama_rss_mib;
 const ollamaCpu=data?.ollama_cpu_percent_one_core;
 const stage=!jobId?'Awaiting new mission':historical?'RECORDED / NO RESOURCE HISTORY':live?'LIVE · PROCESS TELEMETRY':data?.monitor_state==='finished'?'JOB FINISHED':data?.monitor_state==='unavailable'?'MONITOR UNAVAILABLE':data?.monitor_state==='waiting'?'WAITING FOR WORKER':'AWAITING FIRST SAMPLE';
 return <section className="nimResourcePanel" aria-label="NIMBO live resource monitor">
  <div className="nimResourceHeading">
   <div><Activity size={16}/><strong>LIVE RESOURCE MONITOR</strong><span className={live?'nimResourceLive':''}>{stage}</span></div>
   <button type="button" aria-expanded={open} onClick={()=>setOpen(v=>!v)}>
    {open?<ChevronUp size={15}/>:<ChevronDown size={15}/>} {open?'Collapse':'Expand'}
   </button>
  </div>
  {open&&<div className="nimResourceBody">
   {historical?<p className="nimResourceNotice">This is a recorded Replay. Resource sampling was not stored for this historical job; no CPU/RAM numbers are reconstructed.</p>:
    !jobId?<p className="nimResourceNotice">Start a new real mission to measure its worker process tree. GPU/MPS is not instrumented.</p>:
    <>
     <div className="nimResourceGrid">
      <div className="nimResourceTile"><div className="nimResourceTileTitle"><Cpu size={15}/> Job CPU</div><strong>{cpuLabel(cpu)}</strong><small>100% = one logical CPU core</small></div>
      <div className="nimResourceTile"><div className="nimResourceTileTitle"><MemoryStick size={15}/> Job RAM (RSS)</div><strong>{mibLabel(rss)}</strong><small>Observed peak: {mibLabel(peak)}</small></div>
      <div className="nimResourceTile"><div className="nimResourceTileTitle"><HardDrive size={15}/> Ollama processes</div><strong>{mibLabel(ollama)}</strong><small>CPU: {cpuLabel(ollamaCpu)} · separate from job tree</small></div>
      <div className="nimResourceTile"><div className="nimResourceTileTitle"><Activity size={15}/> GPU / MPS</div><strong>Not measured</strong><small>Apple Unified Memory is not discrete VRAM</small></div>
     </div>
     <div className="nimResourceSecondary">
      <Meter title="Host CPU" value={percentLabel(hostCpu)} ratio={hostCpu} detail="Whole-system CPU usage"/>
      <Meter title="Host RAM" value={measured(ram)&&measured(ramTotal)?`${mibLabel(ram)} / ${mibLabel(ramTotal)}`:'—'} ratio={ramUsedPct} detail="Host memory usage (includes other apps and cache)"/>
     </div>
     <div className="nimResourceGraphs"><Curve history={history} keyName="cpu" label="Job CPU trend"/><Curve history={history} keyName="ram" label="Job RSS memory trend"/></div>
     <small className="nimResourceFootnote">Live samples from job PID and child processes only. CPU readings need two polls. RSS may double-count shared pages; observed peak is not a lifetime peak. Samples are not persisted, not model-quality evidence, and do not measure energy or billing.</small>
     {data?.note&&data?.monitor_state==='unavailable'&&<p className="nimResourceNotice">{data.note}</p>}
     {data?.monitor_state==='finished'&&<p className="nimResourceNotice">This job finished. Peak above reflects only samples observed before completion in this API session.</p>}
     {error&&<p className="nimResourceError" role="status">Resource API: {error}</p>}
    </>}
  </div>}
 </section>;
}
