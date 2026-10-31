import React,{useEffect,useState} from 'react';
import {Download,FileCheck2,FolderOpen,RefreshCw,ShieldAlert,PlayCircle} from 'lucide-react';
import './nimbo-model-artifact-v1.css';
const ARTIFACT_API='http://127.0.0.1:8000/nimbo/demo/model-artifact';
const COMPANY_API='http://127.0.0.1:8000/nimbo/demo/company-model';
const MAX_BYTES=420*1024*1024;

/** Supported real-model pilot: actually loads the SHA-pinned uploaded ONNX bytes. */
export default function VerifiedModelArtifact({examples=null}){
 const [status,setStatus]=useState(null);
 const [pilotStatus,setPilotStatus]=useState(null);
 const [loading,setLoading]=useState(true);
 const [importing,setImporting]=useState(false);
 const [running,setRunning]=useState(false);
 const [receipt,setReceipt]=useState(null);
 const [pilot,setPilot]=useState(null);
 const [error,setError]=useState('');
 const validExamples=Array.isArray(examples)&&examples.length>=2&&examples.length<=30&&examples.every(x=>x.text?.trim()&&['POSITIVE','NEGATIVE'].includes(x.expected_label));
 async function refresh(){setLoading(true);setError('');try{
  const [a,b]=await Promise.all([fetch(ARTIFACT_API+'/status'),fetch(COMPANY_API+'/status')]);
  if(!a.ok||!b.ok)throw Error('Model pilot service is offline. Restart the NIMBO backend after installing the update.');
  setStatus(await a.json());setPilotStatus(await b.json());
 }catch(e){setStatus(null);setPilotStatus(null);setError(e.message)}finally{setLoading(false)}}
 useEffect(()=>{refresh()},[]);
 async function uploadAndStage(e){const file=e.target.files?.[0];e.target.value='';if(!file)return;
  setReceipt(null);setPilot(null);setError('');
  if(!file.name.toLowerCase().endsWith('.onnx')||file.size>MAX_BYTES){setError('Select the supported .onnx model file under 420 MB.');return}
  setImporting(true);
  try{
   const res=await fetch(COMPANY_API+'/import',{method:'POST',headers:{'Content-Type':'application/octet-stream'},body:file});
   const body=await res.json();if(!res.ok)throw Error(body.detail||'Model import failed');
   setReceipt({...body,fileName:file.name});setPilotStatus(p=>({...p,imported:true}));
  }catch(err){setError(err.message||'Could not import the ONNX model.')}finally{setImporting(false)}
 }
 async function evaluate(){if(!validExamples){setError('Upload a labeled CSV with 2–30 texts (POSITIVE/NEGATIVE) first.');return}
  setError('');setPilot(null);setRunning(true);
  try{const res=await fetch(COMPANY_API+'/evaluate',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({examples})});
   const body=await res.json();if(!res.ok)throw Error(body.detail||'Real inference failed');setPilot(body);
  }catch(err){setError(err.message||'Could not run actual inference.')}finally{setRunning(false)}
 }
 const pct=x=>typeof x==='number'?`${(x*100).toFixed(1)}%`:'Not measured';
 return <section className="nimboArtifact" aria-labelledby="nimbo-artifact-title">
  <div className="nimboArtifactHead"><div><span>COMPANY PILOT · REAL MODEL FILE</span><h3 id="nimbo-artifact-title">Bring a model. Evaluate it on your data.</h3><p>Use a real public pretrained sentiment model as a stand-in for a company's classifier. Import its supported ONNX export, then execute that uploaded file on labeled inputs.</p></div>
    <button type="button" className="nimboArtifactRefresh" disabled={loading||running||importing} onClick={refresh} aria-label="Refresh model availability"><RefreshCw size={17}/></button></div>
  <div className="nimboArtifactActions"><div className="nimboArtifactMeta"><FileCheck2 size={20}/><div><strong>Public DistilBERT IMDb classifier · real ONNX FP32</strong><small>{loading?'Checking local files…':status?.available?'Verified ONNX export available locally':status?'First export the supported ONNX candidate using NIMBO':'Artifact service offline'}</small></div></div>
   {status?.available?<a className="nimboArtifactDownload" href={ARTIFACT_API+'/download'} download><Download size={16}/> Download model (.onnx)</a>:<span className="nimboArtifactMissing">A verified ONNX export must exist before download</span>}
   <label className={'nimboArtifactUpload '+(importing?'isBusy':'')}><FolderOpen size={16}/>{importing?'Importing genuine model…':'Import supported .onnx model'}<input disabled={importing||running} type="file" accept=".onnx" onChange={uploadAndStage}/></label>
  </div>
  {receipt&&<p className="nimboArtifactReceipt" role="status"><FileCheck2 size={17}/> Imported <strong>{receipt.fileName}</strong>. SHA-256 verified; upload is now staged for actual inference.</p>}
  <div className="nimboCompanyPilotRun"><div><b>Run the uploaded model on real labeled data</b><small>{validExamples?`${examples.length} labeled evaluation rows ready`:'Choose Evaluation dataset and upload a real labeled CSV first'}</small></div><button type="button" className="nimboArtifactDownload" disabled={!pilotStatus?.imported||!validExamples||importing||running} onClick={evaluate}><PlayCircle size={16}/>{running?'Running real models…':'Evaluate uploaded ONNX'}</button></div>
  {pilot&&<div className="nimboCompanyPilotResult" role="status"><b>Real local model evaluation completed</b><div><span>Uploaded ONNX accuracy<strong>{pct(pilot.evaluation.candidate.accuracy)}</strong></span><span>PyTorch accuracy<strong>{pct(pilot.evaluation.baseline.accuracy)}</strong></span><span>Prediction agreement<strong>{pct(pilot.evaluation.prediction_agreement)}</strong></span><span>Mean warm CPU latency<strong>{pilot.evaluation.baseline.mean_warm_ms} → {pilot.evaluation.candidate.mean_warm_ms} ms</strong></span></div><small>Actual execution of the uploaded ONNX file; small company-pilot sample, NOT an independently validated speedup or compute-savings claim. Original live model unchanged.</small></div>}
  {error&&<p className="nimboArtifactError" role="alert"><ShieldAlert size={16}/>{error}</p>}
  <p className="nimboArtifactLimit">This pilot accepts only the checksum-pinned public classifier export; it does not accept arbitrary third-party weights. Model source: <a href="https://huggingface.co/lvwerra/distilbert-imdb" target="_blank" rel="noreferrer">Hugging Face</a>. Upload does not deploy or register the model. For official claims use the separate 800-row held-out evidence.</p>
 </section>;
}
