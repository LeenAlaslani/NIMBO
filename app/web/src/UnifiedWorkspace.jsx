/** NIMBO M9: one entry path, supported-model intake -> original Mission Floor. */
import React, { useState } from 'react';
import { ArrowRight, ChevronDown, ChevronUp, Cpu, ShieldCheck } from 'lucide-react';
import './unified-nimbo.css';

const ID = /^[a-zA-Z0-9][a-zA-Z0-9._-]+\/[a-zA-Z0-9][a-zA-Z0-9._-]+$/;
const SHA = /^[0-9a-f]{40}$/;
const FILE = /^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,90}\.(?:csv|jsonl)$/;
export const EXAMPLE = Object.freeze({
  model_id: 'cardiffnlp/twitter-roberta-base-sentiment-latest',
  revision: 'b995b77f45ae7ed76649b47d04952ec8be9f3591',
  pilot_file: 'pilot.csv',
  heldout_file: 'tweeteval_sentiment_test_200.csv',
  preprocess: 'cardiff_tweet',
});
export function validateIntake(form) {
  if (!ID.test(form.model_id.trim())) return 'Enter a supported public Hugging Face model ID in owner/model format.';
  if (!SHA.test(form.revision.trim())) return 'Model revision must be an immutable 40-character lowercase SHA.';
  if (!FILE.test(form.pilot_file.trim()) || !FILE.test(form.heldout_file.trim())) return 'Use staged CSV/JSONL filenames, without folder paths.';
  if (!['identity', 'cardiff_tweet'].includes(form.preprocess)) return 'Choose a supported preprocessing policy.';
  return '';
}

export default function UnifiedWorkspace({ onLaunch, renderAdvanced }) {
  const [form, setForm] = useState({model_id:'',revision:'',pilot_file:'pilot.csv',heldout_file:'tweeteval_sentiment_test_200.csv',preprocess:'identity'});
  const [error, setError] = useState('');
  const [advanced, setAdvanced] = useState(false);
  const edit = (key, val) => setForm(prev => ({...prev, [key]:val}));
  function submit(e) {
    e.preventDefault();
    const msg = validateIntake(form);
    if (msg) { setError(msg); return; }
    setError('');
    onLaunch({
      problem:'classification',operation:'auto_onnx',goal:'speed',
      model_id:form.model_id.trim(),revision:form.revision.trim(),
      pilot_file:form.pilot_file.trim(),heldout_file:form.heldout_file.trim(),
      preprocess:form.preprocess, prompt:'Optimize supported model',
      inputMode:'benchmark',examples:[],sampleCount:200,
      baseline:form.model_id.trim()+' · PyTorch FP32', candidate:form.model_id.trim()+' · ONNX Runtime FP32',
    });
  }
  return <div className="nimuniWorkspace">
    <header className="nimuniIntro"><span>NIMBO / MODEL OPTIMIZATION</span><h2>Bring your model. We find the efficient version.</h2>
      <p>Provide a supported model and labeled evaluation files. NIMBO measures the original, tests an optimized candidate, verifies quality, and delivers only if the gates pass.</p>
    </header>
    <form className="nimuniForm" onSubmit={submit}>
      <div className="nimuniStep"><b>01</b><div><h3>Your model</h3><p>Supported public Hugging Face text classifier at a pinned revision</p></div></div>
      <div className="nimuniGrid2">
        <label>Model ID <input required value={form.model_id} placeholder="owner/model" maxLength={193} onChange={e=>edit('model_id',e.target.value)}/></label>
        <label>Model revision SHA <input required value={form.revision} placeholder="40 lowercase hex characters" maxLength={40} onChange={e=>edit('revision',e.target.value)}/></label>
      </div>
      <button type="button" className="nimuniLink" onClick={()=>{setForm({...EXAMPLE});setError('')}}>Use verified RoBERTa example</button>
      <div className="nimuniStep"><b>02</b><div><h3>Evaluation data</h3><p>Give NIMBO labeled examples to protect accuracy</p></div></div>
      <div className="nimuniGrid2">
        <label>Pilot dataset filename <input required value={form.pilot_file} onChange={e=>edit('pilot_file',e.target.value)}/></label>
        <label>Held-out dataset filename <input required value={form.heldout_file} onChange={e=>edit('heldout_file',e.target.value)}/></label>
      </div>
      <label className="nimuniPreprocess">Text preprocessing <select value={form.preprocess} onChange={e=>edit('preprocess',e.target.value)}><option value="identity">Standard</option><option value="cardiff_tweet">Cardiff Tweet (social text)</option></select></label>
      <p className="nimuniHint">Current local version: files must already exist in <code>results/auto_optimizer_job_inputs_v1/</code> (pilot) and <code>results/auto_optimizer_heldout_v1/</code> (held-out). Browser model-file and dataset upload are not connected yet.</p>
      <div className="nimuniStep"><b>03</b><div><h3>Optimization objective</h3><p>NIMBO chooses the supported execution method; you don't need to choose ONNX or INT8.</p></div></div>
      <div className="nimuniObjective"><Cpu size={19}/><div><strong>Faster inference</strong><span>ONNX FP32 · measured speed, CPU and quality gates</span></div><ShieldCheck size={20}/></div>
      <p className="nimuniHint">Smaller-model optimization is experimental and not available as an automatic approved path yet. Rejected candidates keep the original model.</p>
      {error&&<p className="nimuniError" role="alert">{error}</p>}
      <button className="nimuniPrimary" type="submit">Enter Mission Floor <ArrowRight size={19}/></button>
    </form>
    <section className="nimuniAdvanced"><button type="button" onClick={()=>setAdvanced(v=>!v)} aria-expanded={advanced}>
      Other workloads & advanced tools {advanced?<ChevronUp size={17}/>:<ChevronDown size={17}/>}
    </button>{advanced&&<div className="nimuniLegacy">{renderAdvanced?.()}</div>}</section>
  </div>;
}
