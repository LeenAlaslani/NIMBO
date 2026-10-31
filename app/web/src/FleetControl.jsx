// NIMBO_V17_OPERATIONAL_FLEET
import React,{useEffect,useMemo,useState} from 'react';
import './fleet-control.css';
const API='http://127.0.0.1:8000';
const num=(v,d=1)=>Number.isFinite(v)?v.toLocaleString(undefined,{maximumFractionDigits:d}):'—';
const readable=id=>{const end=String(id||'').split('/').pop();return end.replace(/^distilbert-base-uncased-finetuned-sst-2-english$/i,'DistilBERT · SST-2').replace(/^bert-base-uncased-SST-2$/i,'BERT · SST-2').replace(/^Qwen3-0\.6B$/i,'Qwen3 · 0.6B')};
export default function FleetControl({onBack}){
 const [registry,setRegistry]=useState(null),[data,setData]=useState(null),[error,setError]=useState(''),[loading,setLoading]=useState(false),[tab,setTab]=useState('models'),[search,setSearch]=useState(''),[selected,setSelected]=useState(null);
 async function refresh(){setLoading(true);setError('');try{
  const [r,d]=await Promise.all([fetch(`${API}/api/registry`,{cache:'no-store'}),fetch(`${API}/nimbo/evidence/unified`,{cache:'no-store'})]);
  if(!r.ok||!d.ok)throw Error(`Backend unavailable: registry HTTP ${r.status}, evidence HTTP ${d.status}`);
  const [models,evidence]=await Promise.all([r.json(),d.json()]);setRegistry(models);setData(evidence);
 }catch(e){setError(String(e.message))}finally{setLoading(false)}}
 useEffect(()=>{refresh()},[]);
 const recs=data?.records||[],models=registry?.models||[],runtime=recs.filter(x=>x.evidence_kind==='runtime_measurement'),paired=recs.filter(x=>x.evidence_kind==='local_paired_cpu_benchmark');
 const filtered=useMemo(()=>models.filter(m=>(m.id+' '+m.workload+' '+m.runtime).toLowerCase().includes(search.toLowerCase())),[models,search]);
 const sorted=[...runtime].reverse(),imageConnected=registry?.image_gateway_connected===true;
 const metricPairs=paired.filter(p=>Number.isFinite(p.baseline_mean_inference_ms)&&Number.isFinite(p.candidate_mean_inference_ms));
 const latestPaired=[...paired].reverse();
 const counts=new Map();for(const x of runtime)counts.set(x.workload,(counts.get(x.workload)||0)+1);
 return <section className="fleet17">
  <header className="fleet17Header"><div><span className="fleet17Eyebrow">NIMBO / MODEL OPERATIONS</span><h1>Fleet Control</h1><p>Registered models, connected infrastructure and measured request history.</p>{onBack&&<button className="fleet17Back" onClick={onBack}>← Mission Floor</button>}</div><button className="fleet17Refresh" disabled={loading} onClick={refresh}>{loading?'Refreshing…':'↻ Refresh status'}</button></header>
  {error&&<div className="fleet17Error" role="alert">{error}. Start the NIMBO backend on port 8000.</div>}
  <div className="fleet17Kpis">
   <article><small>Registered models</small><b>{registry?models.length:'—'}</b><span>Configuration · not a health check</span></article>
   <article><small>Recorded API requests</small><b>{data?runtime.length:'—'}</b><span>Completed historical calls</span></article>
   <article><small>Paired comparisons</small><b>{data?paired.length:'—'}</b><span>Offline local CPU measurements</span></article>
   <article><small>GPU image gateway</small><b>{registry?(imageConnected?'Connected':'Disconnected'):'—'}</b><span>{imageConnected?'Configured gateway flag':'No active GPU worker evidence'}</span></article>
  </div>
  <div className="fleet17SectionHeading"><div><h2>Model inventory</h2><p>Availability and resource utilization require live model health checks. Registry status alone is not an uptime measurement.</p></div><label className="fleet17Search">Find model<input value={search} onChange={e=>setSearch(e.target.value)} placeholder="Search registered models"/></label></div>
  <div className="fleet17Models">{filtered.length?filtered.map(m=>{
   const ready=m.status==='ready_on_first_load',disconnected=m.status==='not_connected';
   return <button type="button" key={m.id} className="fleet17Model" onClick={()=>setSelected(m)}>
    <div className="fleet17ModelTop"><span className="fleet17ModelIcon">{m.workload==='classification'?'◈':m.workload==='generation'?'✳':'▧'}</span><span className={'fleet17State '+(disconnected?'offline':'')}>{disconnected?'Not connected':ready?'Configured':'Status unknown'}</span></div>
    <strong>{readable(m.id)}</strong><p>{m.workload||'Unknown workload'} · {m.runtime||'Unknown runtime'}</p><small>{ready?'Loads on first request; health not measured':disconnected?'External connection required':'View model configuration'}</small><span className="fleet17Inspect">Inspect model →</span>
   </button>;
  }):<div className="fleet17Blank">{loading?'Loading model inventory…':'No matching models.'}</div>}</div>
  <div className="fleet17SectionHeading"><div><h2>Recorded operations</h2><p>Historical requests and paired experiments are separate measurements.</p></div><div className="fleet17Tabs" role="tablist"><button role="tab" aria-selected={tab==='models'} className={tab==='models'?'active':''} onClick={()=>setTab('models')}>Request history</button><button role="tab" aria-selected={tab==='experiments'} className={tab==='experiments'?'active':''} onClick={()=>setTab('experiments')}>Comparisons</button></div></div>
  {tab==='models'?<div className="fleet17History">{sorted.length?sorted.slice(0,20).map((e,i)=><div className="fleet17HistoryRow" key={e.id||i}><span className="fleet17HistorySymbol">●</span><div><strong>{e.workload||'Model request'}</strong><small>{e.route||'No route'} · {e.timestamp_utc?new Date(e.timestamp_utc).toLocaleString():'No timestamp'}</small></div><b>{num(e.latency_ms,1)} ms</b><span className="fleet17Trust">{e.quality_certified?'Quality verified':'Quality not independently certified'}</span></div>):<p className="fleet17Blank">No runtime requests recorded.</p>}</div>:<div className="fleet17Experiments">{latestPaired.length?latestPaired.slice(0,10).map(x=><article key={x.id}><div className="fleet17ExperimentHead"><strong>{readable(x.baseline_model)} <span>vs</span> {readable(x.candidate_model)}</strong><small>{x.sample_count} inputs · warm CPU</small></div><div className="fleet17CompareBars"><div><span>Baseline</span><b>{num(x.baseline_mean_inference_ms,2)} ms</b></div><div><span>Candidate</span><b>{num(x.candidate_mean_inference_ms,2)} ms</b></div></div><p>{Number.isFinite(x.mean_latency_reduction_fraction)?`${(100*x.mean_latency_reduction_fraction).toFixed(1)}% lower candidate latency`: 'Latency not comparable'} · {x.all_labels_present?'Accuracy checked on supplied labels':'Accuracy unverified'}</p></article>):<p className="fleet17Blank">No comparable experiments recorded yet.</p>}</div>}
  <footer className="fleet17Foot">This page displays configured assets and recorded activity. It does not imply live GPU usage, live request concurrency, monetary savings or model uptime.</footer>
  {selected&&<div className="fleet17ModalBackdrop" onMouseDown={e=>{if(e.target===e.currentTarget)setSelected(null)}}><section className="fleet17Modal" role="dialog" aria-modal="true" aria-label="Model details"><button className="fleet17Close" onClick={()=>setSelected(null)}>Close ×</button><small>MODEL INVENTORY / REGISTERED ASSET</small><h2>{readable(selected.id)}</h2><p className="fleet17FullId">{selected.id}</p><dl><dt>Workload</dt><dd>{selected.workload||'Unknown'}</dd><dt>Runtime</dt><dd>{selected.runtime||'Unknown'}</dd><dt>Configuration</dt><dd>{selected.status||'Unknown'}</dd><dt>Live health</dt><dd>Not measured</dd><dt>GPU / memory utilization</dt><dd>Not connected</dd></dl><p>Run a mission from Workspace or connect a health-telemetry adapter to monitor live capacity.</p></section></div>}
 </section>;
}
