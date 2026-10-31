import React,{useEffect,useState} from 'react';
import './optimization-game.css';

const API='http://127.0.0.1:8000';
export const CHALLENGES=[
 {id:'sentiment',name:'Pick the lightweight route',kind:'classification',level:'Starter',focus:'Model routing',prompt:'The customer support was excellent and I am very happy with the service.',goal:'Observe whether the saved routing policy uses DistilBERT or escalates to BERT.',worker:'ROUTA'},
 {id:'hard_sentiment',name:'Can confidence be trusted?',kind:'classification',level:'Explorer',focus:'Quality guardrail',prompt:'The movie was beautiful but the story was painfully slow.',goal:'Review model confidence, route, and whether quality is independently certified.',worker:'VERA'},
 {id:'context',name:'Can NIMBO trim the prompt?',kind:'generation',level:'Explorer',focus:'Context',prompt:'Explain the difference between model confidence and answer correctness in exactly three short sentences.',goal:'Check whether a context reduction was actually executed; a short answer alone does not prove optimization.',worker:'TOKI'},
 {id:'routing',name:'Small model, real answer',kind:'generation',level:'Operator',focus:'Compute evidence',prompt:'Explain how an AI system can choose between small and large AI models. Give two benefits, one drawback, and one concrete example.',goal:'Inspect the selected model and measured latency; no baseline means no proven savings.',worker:'METRI'},
 {id:'image',name:'Protect image quality',kind:'image',level:'Advanced',focus:'Image verifier',prompt:'A bright orange lifebuoy hanging from a boat railing at sea, realistic photo.',goal:'This live mission requires a connected image gateway. Historical quality results are shown separately.',worker:'VERA'},
];

const formatMs=(v)=>typeof v==='number'?`${v.toLocaleString(undefined,{maximumFractionDigits:2})} ms`:'Not measured';
const fmtPct=(v)=>typeof v==='number'?`${(v*100).toFixed(2)}%`:'Not measured';

export function ChallengePicker({selected,onPick,imageConnected}){
 return <section className="gameChallenge" aria-label="Optimization missions">
  <div className="gameTitle"><div><small>MISSION ACADEMY · LEARN BY DOING</small><h3>Optimization challenges</h3><p>Run a real task, then inspect what NIMBO can actually prove.</p></div><span className="gameIndicator">5 missions</span></div>
  <div className="challengeScroll">{CHALLENGES.map(c=><button key={c.id} type="button" className={'challengeCard '+(selected===c.id?'chosen':'')} onClick={()=>onPick(c)}>
   <span className="challengeLevel">{c.level} · {c.focus}</span><strong>{c.name}</strong><small>{c.goal}</small><span className="challengeWorker">{c.worker}{c.kind==='image'&&!imageConnected?' · gateway offline':''}</span>
  </button>)}</div>
 </section>
}

export function WorkerRoster({mode,stage,report,events}){
 const roles=[
  ['NIMBO','Mission courier','analyze','captain'],['TOKI','Context optimization','context','toki'],
  ['ROUTA','Model routing','route','routa'],['VERA','Quality & guardrails','verify','vera'],
  ['METRI','Evidence analyst','deliver','metri'],['PIXI','Image specialist','image','pixi']
 ];
 const stageNames=(events||[]).map(x=>x.stage);
 return <div className="gameRoster" aria-label="Worker responsibilities">
  {roles.map(([name,title,role,cls])=>{
   const contextUsed=false; // The current live generation endpoint has no context optimizer.
   const active=(role==='analyze'&&stage==='analyze')||(role==='route'&&stage==='route')||(role==='verify'&&stage==='verify')||(role==='deliver'&&stage==='deliver')||(role==='image'&&mode==='image'&&stage==='execute');
   const state=role==='context'?'Not in live pipeline':role==='image'&&mode!=='image'?'Idle':active?'Active':'Waiting';
   return <div key={name} className={'rosterUnit '+cls+(active?' active':'')}><span className="roleIcon">{name.slice(0,1)}</span><div><strong>{name}</strong><small>{title}</small></div><em>{state}</em></div>;
  })}
 </div>
}

export function OptimizationReceipt({report,events,mode,onSeeEvidence}){
 const [bench,setBench]=useState(null);
 const [err,setErr]=useState('');
 useEffect(()=>{
  const controller=new AbortController();
  fetch(`${API}/nimbo/evidence/unified`,{signal:controller.signal}).then(r=>{if(!r.ok)throw Error('Evidence service unavailable');return r.json()}).then(setBench).catch(e=>{if(e.name!=='AbortError')setErr(e.message)});
  return ()=>controller.abort();
 },[report]);
 const route=report?.route||'Not selected';
 const metric=report?.metrics?.end_to_end_ms;
 const hasFallback=route==='small_then_large';
 const anyTrace=(events||[]).some(e=>/fallback|escalat/i.test(e.message||''));
 const guard=report?.guardrail;
 const quality=guard?.quality_certified===true?'Certified by attached checker':guard?.action==='UNVERIFIED'?'UNVERIFIED':'Not independently verified';
 const historical=(bench?.records||[]).filter(r=>r.workload==='image_generation'&&r.evidence_kind==='offline_labeled_benchmark'&&r.criterion==='full_prompt');
 const fast=historical.find(r=>r.route==='always_fast'),nimbo=historical.find(r=>r.route==='nimbo_existing'),standard=historical.find(r=>r.route==='always_standard');
 return <section className="gameReceipt" aria-label="Optimization evidence receipt">
  <div className="gameTitle"><div><small>AFTER-ACTION REPORT</small><h3>Optimization receipt</h3></div><span className="gameIndicator">{report?'Measured request':'Run a mission'}</span></div>
  {report?<>
   <div className="receiptMetrics">
    <div><small>Selected model route</small><strong>{route}</strong></div>
    <div><small>Measured request latency</small><strong>{formatMs(metric)}</strong></div>
    <div><small>Fallback</small><strong>{hasFallback||anyTrace?'Recorded':'Not recorded'}</strong></div>
    <div><small>Quality evidence</small><strong>{quality}</strong></div>
    <div><small>Live input tokens</small><strong>Not measured</strong></div>
    <div><small>Proven compute savings</small><strong>Not established</strong></div>
   </div>
   <div className="receiptReason"><strong>What actually happened?</strong><p>{mode==='classification'?'The saved classification policy selected the displayed route. Without a controlled per-request baseline, model selection alone does not establish a compute reduction.':mode==='generation'?'Qwen ran as a real local model. The separately tested context optimizer is not connected to this live endpoint, so no token saving is claimed.':'Image inference requires a connected gateway; offline results cannot certify a live image.'}</p></div>
  </>:<p className="receiptEmpty">Complete a mission to see route, latency, guardrail status, and where evidence is still missing.</p>}
  <div className="benchPreview">
   <div className="benchHeader"><strong>Historical evidence · Image full-prompt</strong><span>OFFLINE · 16 samples</span></div>
   {historical.length===3?<div className="benchBars">{[['Fast',fast],['NIMBO',nimbo],['Standard',standard]].map(([name,r])=><div className="benchRow" key={name}><span>{name}</span><div><i style={{width:fmtPct(r?.success_rate)}}/></div><b>{fmtPct(r?.success_rate)}</b></div>)}</div>:<p className="receiptEmpty">{err||'No comparable historical records found.'}</p>}
   <p className="benchNote">These historical human-rated results are not live savings and not a held-out validation. Different generation paths cannot be compared as identical quality. No monetary or FLOP savings are inferred.</p>
  </div>
  <button className="evidenceOpen" onClick={onSeeEvidence}>Open full evidence center →</button>
 </section>;
}
