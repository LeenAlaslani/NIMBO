// NIMBO_V20_SAME_MODEL_TRIAL
import React,{useEffect,useState} from 'react';
import {ArrowRight,Database,GitCompareArrows,Layers3,ShieldCheck,PlugZap,CheckCircle2,FileUp,BriefcaseBusiness} from 'lucide-react';
import './company-workspace-v15.css';
import './nimbo-v20.css';
import './nimbo-image-gateway-v22.css';
import AdoptionConsoleV211 from './AdoptionConsoleV211';
const API='http://127.0.0.1:8000';
const SMALL='distilbert/distilbert-base-uncased-finetuned-sst-2-english';
const LARGE='textattack/bert-base-uncased-SST-2';
const EXTERNAL='lvwerra/distilbert-imdb'; // allowlisted NIMBO v20
const SHORT={classification:'Customer feedback classification',generation:'Text generation',image:'Image generation'};
const SAMPLE={classification:'The movie was beautiful but the story was painfully slow.',generation:'Explain why model confidence is not proof of factual correctness in three sentences.',image:'A bright orange lifebuoy on a boat railing at sea, realistic photograph.'};
const cleanLabel = x => { const val=String(x??'').trim().toUpperCase();return val==='POSITIVE'||val==='NEGATIVE'?val:null; };
function parseCsv(data){
 const rows=[];let row=[],item='',quotes=false;
 for(let i=0;i<data.length;i++){
   const c=data[i];
   if(c==='"'){if(quotes&&data[i+1]==='"'){item+='"';i++;}else quotes=!quotes;}
   else if(c===','&&!quotes){row.push(item);item='';}
   else if((c==='\n'||c==='\r')&&!quotes){if(c==='\r'&&data[i+1]==='\n')i++;row.push(item);if(row.some(x=>x.trim()))rows.push(row);row=[];item='';}
   else item+=c;
 }
 if(quotes)throw Error('Unclosed quotation in CSV');
 row.push(item);if(row.some(x=>x.trim()))rows.push(row);
 if(!rows.length)return [];
 const hdr=rows[0].map(x=>x.trim().toLowerCase());
 const textCol=hdr.indexOf('text'),labelCol=hdr.indexOf('label');
 if(textCol<0)throw Error('CSV must contain a text column (optional label column).');
 return rows.slice(1).map(r=>({text:String(r[textCol]||'').trim(),expected_label:labelCol>=0?cleanLabel(r[labelCol]):null})).filter(x=>x.text);
}
export function parseEvalFile(data,extension){
 const items=extension==='jsonl'?data.split(/\r?\n/).filter(x=>x.trim()).map(x=>{
  const p=JSON.parse(x);return {text:String(p.text||'').trim(),expected_label:cleanLabel(p.label||p.expected_label)};
 }):parseCsv(data);
 if(!items.length)throw Error('File has no evaluation inputs.');
 if(items.length>30)throw Error('The local comparison currently supports a maximum of 30 inputs.');
 if(items.some(x=>!x.text||x.text.length>2000))throw Error('Each input must contain 1–2,000 characters.');
 return items;
}
export default function CompanyWorkspace({registry,onLaunch,onThemeChange,theme}){
 const [catalog,setCatalog]=useState([SMALL,LARGE]);
 const [problem,setProblem]=useState('classification');
 const [imageMode,setImageMode]=useState('adaptive');const [imageSeed,setImageSeed]=useState(42);
 const [operation,setOperation]=useState('compare');
 const [goal,setGoal]=useState('latency');
 const [baseline,setBaseline]=useState(LARGE);
 const [candidate,setCandidate]=useState(SMALL);
 const [request,setRequest]=useState('');
 const [inputMode,setInputMode]=useState('batch');
 const [trialReady,setTrialReady]=useState(false);
 const [onnxReady,setOnnxReady]=useState(false);
 const [validationReady,setValidationReady]=useState(false);
 const [trialPhase,setTrialPhase]=useState('development');
 const [label,setLabel]=useState('');
 const [batch,setBatch]=useState(null);
 const [fileName,setFileName]=useState('');
 const [error,setError]=useState('');
 const [apiConnected,setApiConnected]=useState(false);
 useEffect(()=>{let active=true;fetch(`${API}/nimbo/bench/catalog`).then(r=>{if(!r.ok)throw Error('Catalog unavailable');return r.json()}).then(j=>{if(active){setCatalog((j.models||[]).map(m=>m.id));setApiConnected(true)}}).catch(()=>{if(active)setApiConnected(false)});return ()=>{active=false}},[]);
 useEffect(()=>{fetch(`${API}/nimbo/optimize/v20/catalog`).then(r=>{if(!r.ok)throw Error('Offline');return r.json()}).then(()=>setTrialReady(true)).catch(()=>setTrialReady(false));},[]);
 useEffect(()=>{fetch(`${API}/nimbo/optimize/onnx/catalog`).then(r=>{if(!r.ok)throw Error('Offline');return r.json()}).then(()=>setOnnxReady(true)).catch(()=>setOnnxReady(false));},[]);
 useEffect(()=>{fetch(`${API}/nimbo/validate/v21/catalog`).then(r=>{if(!r.ok)throw Error('Offline');return r.json()}).then(()=>setValidationReady(true)).catch(()=>setValidationReady(false));},[]);
 function chooseProblem(p){setProblem(p);setInputMode(p==='classification'?'batch':'single');setRequest(p==='classification'?'':SAMPLE[p]);setBatch(null);setFileName('');setError('');setOperation(p==='classification'?'compare':'live');}
 async function upload(e){const f=e.target.files?.[0];if(!f)return;setError('');try{
   if(!/\.(csv|jsonl)$/i.test(f.name)||f.size>1024*1024)throw Error('Choose a CSV or JSONL file under 1 MB.');
   const samples=parseEvalFile(await f.text(),f.name.toLowerCase().endsWith('.jsonl')?'jsonl':'csv');
   setBatch(samples);setFileName(f.name);setInputMode('batch');
 }catch(err){setBatch(null);setFileName('');setError(String(err.message))}}
 function submit(){setError('');if(operation==='validate_onnx'){if(!validationReady){setError('Restart the backend to connect the held-out validation endpoint.');return;}onLaunch({problem:'classification',operation:'validate_onnx',goal,baseline:EXTERNAL+' · PyTorch FP32',candidate:EXTERNAL+' · ONNX Runtime FP32',model_id:EXTERNAL,trialPhase:'heldout',examples:[],sampleCount:800,rounds:3,prompt:'IMDb independent held-out evaluation: 800 labeled records · 3 paired runs',inputMode:'benchmark',fileName:'IMDb disjoint test holdout'});return;}if(operation==='optimize'||operation==='optimize_onnx'){if(operation==='optimize_onnx'?!onnxReady:!trialReady){setError('Restart the NIMBO backend to connect this optimization endpoint.');return;}if(inputMode==='batch'&&(!batch?.length||batch.length<10||batch.some(x=>!x.expected_label))){setError('Custom optimization requires at least 10 labeled CSV rows. Choose IMDb 200 instead.');return;}const items=inputMode==='batch'?batch:[];onLaunch({problem:'classification',operation,goal,baseline:EXTERNAL+' · PyTorch FP32',candidate:EXTERNAL+(operation==='optimize_onnx'?' · ONNX Runtime FP32':' · INT8'),model_id:EXTERNAL,trialPhase,examples:items,sampleCount:items.length||200,prompt:`IMDb evaluation: ${items.length||200} labeled records`,inputMode:items.length?'batch':'benchmark',fileName:items.length?fileName:'IMDb test subset'});return;}if(inputMode==='batch'&&!batch?.length){setError('Upload a labeled CSV or JSONL set first, or select Single request.');return;}if(inputMode==='single'&&!request.trim()){setError('Enter a request or upload an evaluation set.');return}
  if(operation==='compare'){
    if(!apiConnected){setError('Local comparison endpoint is not connected. Restart the NIMBO backend.');return}
    if(baseline===candidate){setError('Choose different baseline and candidate models.');return}
  }
  if(problem==='image'&&!registry?.image_gateway_connected){setError('Connect a real image gateway before running this workflow.');return}
  const examples=inputMode==='batch'&&batch?.length?batch:[{text:request.trim(),expected_label:cleanLabel(label)}];
  onLaunch({problem,operation,goal,baseline,candidate,examples,prompt:examples[0].text,inputMode,imageMode,imageSeed,fileName:inputMode==='batch'?fileName:null});
 }
 const compare=problem==='classification'&&operation==='compare';
 const optimizing=problem==='classification'&&(operation==='optimize'||operation==='optimize_onnx'||operation==='validate_onnx');
 const onnxOptimizing=problem==='classification'&&operation==='optimize_onnx';
 const validating=problem==='classification'&&operation==='validate_onnx';
 return <section className="nim15Workspace">
   <div className="nim15Heading"><div><span>OPERATIONS / WORKSPACE</span><h2>Configure your AI workload.</h2><p>Connect a real execution path, select a goal, and send the mission to the floor.</p></div><span className="nim15Env"><PlugZap size={17}/> Local model environment</span></div>
   <div className="nim15Workflow">
    <section className="nim15Block"><div className="nim15BlockTop"><b>01</b><div><h3>Workload</h3><p>What does your application need?</p></div></div>
      <div className="nim15Segments">
       {['classification','generation','image'].map(p=><button type="button" key={p} className={problem===p?'active':''} onClick={()=>chooseProblem(p)}>{SHORT[p]}{p==='image'&&!registry?.image_gateway_connected&&<small>GPU offline</small>}</button>)}
      </div>
      {problem==='classification'&&!optimizing&&<div className="nim16InputMode" role="group" aria-label="Evaluation input type"><button type="button" className={inputMode==='batch'?'active':''} onClick={()=>setInputMode('batch')}>Evaluation dataset</button><button type="button" className={inputMode==='single'?'active':''} onClick={()=>setInputMode('single')}>Single request</button></div>}
      {inputMode==='single'&&!optimizing&&<div className="nim15Field"><label htmlFor="nim15-prompt">Request text <small>One request, not a batch evaluation</small></label><textarea id="nim15-prompt" rows={4} value={request} placeholder="Enter the application's request…" onChange={e=>{setRequest(e.target.value)}} maxLength={2000}/></div>}
      {inputMode==='batch'&&!optimizing&&problem==='classification'&&<div className="nim16DatasetHint">Your dataset is the workload. NIMBO runs each registered model on the same rows; there is no need to retype an example sentence.</div>}
      {problem==='classification'&&inputMode==='batch'&&<div className="nim15Field nim15File"><label htmlFor="nim15-upload"><FileUp size={17}/> Evaluation set — CSV or JSONL</label><input id="nim15-upload" type="file" accept=".csv,.jsonl" onChange={upload}/><p>{batch?`${fileName} · ${batch.length} inputs · ${batch.every(x=>x.expected_label)?'All labels supplied':'No complete quality labels'}`:'Required for dataset comparison. Columns: text,label (labels optional). Only aggregate measurements are recorded.'}</p></div>}
    </section>
    <section className="nim15Block"><div className="nim15BlockTop"><b>02</b><div><h3>Execution path</h3><p>Select what NIMBO will actually run.</p></div></div>
      {problem==='classification'?<div className="nim15SelectGrid"><label><input type="radio" checked={operation==='compare'} onChange={()=>setOperation('compare')}/> Paired baseline comparison <small>Run both models on the same inputs</small></label><label><input type="radio" checked={operation==='live'} onChange={()=>{setOperation('live');setInputMode('single')}}/> Adaptive routing <small>Use the saved NIMBO policy</small></label><label><input type="radio" checked={operation==='optimize'} onChange={()=>{setOperation('optimize');setInputMode('benchmark');setBatch(null);setFileName('')}}/> Dynamic INT8 <small>Same model · smaller weights; previous Mac latency gate failed</small></label><label><input type="radio" checked={onnxOptimizing} onChange={()=>{setOperation('optimize_onnx');setInputMode('benchmark');setBatch(null);setFileName('')}}/> ONNX Runtime FP32 <small>Same model and precision · alternate CPU execution engine</small></label><label><input type="radio" checked={validating} onChange={()=>{setOperation('validate_onnx');setInputMode('benchmark');setBatch(null);setFileName('')}}/> Validate ONNX · 800 held-out <small>3 measured rounds, P95 and local tokenizer+inference gates</small></label></div>:<div className="nim15Info"><CheckCircle2 size={18}/> {problem==='generation'?'Local Qwen3-0.6B is available for single-request inference. No paired baseline currently connected.':'An image gateway must be connected before inference can run.'}</div>}
      {validating&&<div className="nim21Validation"><strong>Independent-of-development test · IMDb</strong><p>800 separate IMDb test rows, 3 alternating CPU rounds, mean/P95, label agreement and tokenizer+inference cost. Only passing variants are staged for manual review — never deployed automatically.</p><small>{validationReady?'Validation endpoint connected':'Validation endpoint offline · restart backend'}</small></div>}
      {optimizing&&!validating&&<div className="nim20Trial"><strong>External model: {EXTERNAL}</strong><p>{onnxOptimizing?'Same weights and same tokenizer; NIMBO exports an ONNX FP32 execution graph and tests it against PyTorch FP32. No speedup is assumed.':'Same model, same tokenizer. FP32 → dynamic INT8. Your previous local CPU result failed the latency requirement.'}</p><label>Evaluation phase <select value={trialPhase} onChange={e=>setTrialPhase(e.target.value)}><option value="development">Development · 200 IMDb examples</option><option value="heldout">Held-out · separate IMDb partition (200 preview)</option></select></label><div className="nim16InputMode"><button type="button" className={inputMode==='benchmark'?'active':''} onClick={()=>setInputMode('benchmark')}>IMDb labeled dataset</button><button type="button" className={inputMode==='batch'?'active':''} onClick={()=>setInputMode('batch')}>My labeled CSV</button></div><small>{(onnxOptimizing?onnxReady:trialReady)?'Optimization endpoint connected':'Optimization endpoint offline · restart backend'} · First run downloads model and data; no savings are assumed.</small></div>}
      {problem==='image'&&<div className="nim22ImageMode" aria-label="Image execution paths"><strong>Real image execution</strong>{/* NIMBO_ADAPTIVE_IMAGE_V24 */}<label className="nim24PrimaryChoice"><input type="radio" checked={imageMode==='adaptive'} onChange={()=>setImageMode('adaptive')}/> Adaptive · Fast first → review → Standard only if needed</label><label><input type="radio" checked={imageMode==='paired'} onChange={()=>setImageMode('paired')}/> Compare fast LCM (4 steps) versus standard SD 1.5 (25 steps)</label><label><input type="radio" checked={imageMode==='fast'} onChange={()=>setImageMode('fast')}/> Fast only · LCM-LoRA</label><label><input type="radio" checked={imageMode==='standard'} onChange={()=>setImageMode('standard')}/> Standard only · baseline</label><label>Reproducible seed <input type="number" min="0" max="2147483647" value={imageSeed} onChange={e=>setImageSeed(Math.max(0,Math.min(2147483647,Number(e.target.value)||0)))}/></label><p>Two real GPU generations on the same seed in comparison mode. Human review decides critical-object presence; NIMBO will not claim image quality automatically. First use downloads the public image model.</p></div>}
      {compare&&<div className="nim15Models"><label>Baseline model<select value={baseline} onChange={e=>setBaseline(e.target.value)}>{catalog.map(m=><option key={m} value={m}>{m}</option>)}</select></label><label>Candidate model<select value={candidate} onChange={e=>setCandidate(e.target.value)}>{catalog.map(m=><option key={m} value={m}>{m}</option>)}</select></label><div className="nim15Hint"><Database size={15}/> {apiConnected?`${catalog.length} registered classifiers`:'Comparison service not connected'}</div></div>}
      {inputMode==='single'&&compare&&<div className="nim15Field"><label htmlFor="nim15-label">Expected sentiment <small>(optional reference label)</small></label><select id="nim15-label" value={label} onChange={e=>setLabel(e.target.value)}><option value="">Not labeled</option><option value="POSITIVE">POSITIVE</option><option value="NEGATIVE">NEGATIVE</option></select></div>}
    </section>
    <section className="nim15Block"><div className="nim15BlockTop"><b>03</b><div><h3>Optimization objective</h3><p>Choose the metric you want to inspect.</p></div></div>
      <div className="nim15Goals"><button type="button" className={goal==='latency'?'active':''} onClick={()=>setGoal('latency')}><GitCompareArrows size={19}/> Lower inference latency</button><button type="button" className={goal==='quality'?'active':''} onClick={()=>setGoal('quality')}><ShieldCheck size={19}/> Preserve accuracy</button></div>
      <div className="nim15Readiness"><p><b>Measured now</b> Model inference latency {compare?'for each registered variant':'for the connected request'}.</p><p><b>Needs evidence</b> Dataset labels for accuracy; real hardware instrumentation for memory, energy or FLOPs.</p><p><b>Connected trials</b> Dynamic INT8 and ONNX Runtime FP32 both compare the same pretrained model on local CPU. Existing INT8 results are preserved.</p></div>
      {error&&<div className="nim15Error" role="alert">{error}</div>}
      <button className="nim15Start" onClick={submit}>Enter Mission Floor <ArrowRight size={18}/></button>
    </section>
   </div>
   <AdoptionConsoleV211/>
 </section>;
}
