import React,{useState} from 'react';
import './mission-observatory.css';
const phases=['analyze','route','execute','verify','deliver'];
const names={analyze:'Mission intake',route:'Model routing',execute:'Inference',verify:'Quality check',deliver:'Delivery'};
const who={analyze:'CAPTAIN NIMBO',route:'ROUTA',execute:'MODEL WORKER',verify:'VERA',deliver:'METRI'};
const fmt=n=>typeof n==='number'?`${n.toFixed(2)} ms`:'Not measured';
export default function MissionObservatory({events=[],report=null,busy=false,replaying=false,stage='analyze',mode='classification',onReplay,replaySpeed=1,onSpeedChange}){
 const [expanded,setExpanded]=useState(true);
 const rec=report||{}, metrics=rec.metrics||{},guard=rec.guardrail||{};
 const action=guard.action||'Not independently verified';
 const route=rec.route||'Not selected';
 const actual=events.filter(e=>phases.includes(e.stage));
 const completed=!!report&&!busy;
 const liveStage=busy?stage:null;
 return <section className="nimObs">
   <div className="nimObsHead"><div><span className="nimObsEyebrow">MISSION OBSERVATORY</span><h2>See the decisions, not just the result.</h2><p>Backend-reported events, measured timings, and missing evidence are shown separately.</p></div><div className="nimObsLive">{replaying?'RECORDED REPLAY':busy?'LIVE BACKEND':'NOT RUNNING'}</div></div>
   <div className="nimObsStats">
    <article><small>Selected model path</small><strong>{route}</strong></article>
    <article><small>Measured request</small><strong>{fmt(metrics.end_to_end_ms)}</strong></article>
    <article><small>Guardrail evidence</small><strong>{action}</strong></article>
    <article><small>Compute saving on this request</small><strong>Not established</strong></article>
   </div>
   <div className="nimObsPanel">
     <div className="nimObsSub"><div><strong>Execution timeline</strong><span>{actual.length} recorded events · {busy?'in progress':'after execution'}</span></div><button type="button" onClick={()=>setExpanded(x=>!x)}>{expanded?'Collapse':'Expand'}</button></div>
     <div className="nimObsSteps">{phases.map(p=><div key={p} className={'nimObsStep '+(liveStage===p||replaying&&stage===p?'now':'')+' '+(actual.some(e=>e.stage===p)?'seen':'')}><span className="nimObsDot"/><b>{names[p]}</b><small>{who[p]}</small></div>)}</div>
     {expanded&&<div className="nimObsEvents">{actual.length?actual.map((e,i)=><div key={i} className="nimObsEvent"><span className="nimObsTime">{fmt(e.at_ms)}</span><span className="nimObsEventStage">{e.stage}</span><span>{e.message}</span></div>):<div className="nimObsEmpty">Start a mission to receive real backend events. The scene does not invent progress.</div>}</div>}
   </div>
   <div className="nimObsBottom">
    <div><strong>Why no savings percentage?</strong><p>Latency for one run is not a saving. A matching baseline run, task-success check and comparable measurement conditions are needed before claiming reduced compute.</p></div>
    <div><strong>Readable playback</strong><p>Inference can finish in milliseconds. Replay makes recorded stages understandable without slowing or rerunning the actual model.</p><button disabled={!actual.length||busy||replaying} onClick={onReplay}>Replay recorded flow</button></div>
   </div>
 </section>;
}
