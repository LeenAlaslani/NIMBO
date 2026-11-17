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
  const [form, setForm] = useState({model_id:'',revision:'',pilot_file:'',heldout_file:'',preprocess:'identity'});
  const [error, setError] = useState('');
  const [advanced, setAdvanced] = useState(false);
  const [technical, setTechnical] = useState(false);
  const edit = (key, val) => setForm(prev => ({...prev, [key]:val}));
  function submit(e) {
    e.preventDefault();
    const msg = validateIntake(form);
    if (msg) {
      setError(msg);
      if (/revision|preprocessing/i.test(msg)) setTechnical(true);
      return;
    }
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

  return <div className="nimuniWorkspace nimuniCleanWorkspace">
    <header className="nimuniIntro nimuniCleanIntro">
      <span>NIMBO / MODEL OPTIMIZATION</span>
      <h2>Make your AI model run better.</h2>
      <p>Measure the original. Test an optimization. Keep only what passes quality and resource checks.</p>
    </header>

    <form className="nimuniForm nimuniCleanForm"
          onSubmit={submit} noValidate>

      <div className="nimuniCleanSections">
        <section className="nimuniCleanPanel">
          <div className="nimuniCleanTitle">
            <b>01</b>
            <h3>Your model</h3>
          </div>

          <label>
            Hugging Face model ID
            <input value={form.model_id}
              placeholder="organization/model-name"
              autoComplete="off"
              onChange={e => setForm(prev => ({
                ...prev,
                model_id:e.target.value,
                revision:e.target.value===prev.model_id
                  ?prev.revision:'',
                preprocess:e.target.value===EXAMPLE.model_id
                  ?prev.preprocess:'identity'
              }))}/>
          </label>

          <button type="button" className="nimuniLink"
            onClick={() => {
              setForm({...EXAMPLE});
              setTechnical(false);
              setError('');
            }}>
            Use verified RoBERTa example
          </button>

          <p className="nimuniCleanHelp">
            Supports selected public text classifiers.
            New models require an immutable revision in
            Advanced setup.
          </p>
        </section>

        <section className="nimuniCleanPanel">
          <div className="nimuniCleanTitle">
            <b>02</b>
            <h3>Evaluation data</h3>
          </div>

          <div className="nimuniCleanDatasets">
            <label>
              Pilot dataset
              <input value={form.pilot_file}
                placeholder="pilot.csv"
                onChange={e=>edit('pilot_file',e.target.value)}/>
            </label>

            <label>
              Held-out dataset
              <input value={form.heldout_file}
                placeholder="heldout.csv"
                onChange={e=>edit('heldout_file',e.target.value)}/>
            </label>
          </div>

          <p className="nimuniCleanHelp">
            CSV or JSONL files must already be staged locally.
            Browser uploads are not connected yet.
          </p>
        </section>
      </div>

      <div className="nimuniCleanTechnical">
        <button type="button"
          aria-expanded={technical}
          onClick={()=>setTechnical(v=>!v)}>
          <span>Advanced setup</span>
          {technical?<ChevronUp size={17}/>:<ChevronDown size={17}/>}
        </button>

        {technical&&<div className="nimuniCleanTechnicalBody">
          <label>
            Immutable model revision SHA
            <input value={form.revision}
              placeholder="40 lowercase hexadecimal characters"
              maxLength={40}
              onChange={e=>edit('revision',e.target.value)}/>
          </label>

          <label>
            Text preprocessing
            <select value={form.preprocess}
              onChange={e=>edit('preprocess',e.target.value)}>
              <option value="identity">Standard</option>
              <option value="cardiff_tweet">
                Cardiff Tweet (social text)
              </option>
            </select>
          </label>

          <p className="nimuniCleanHelp">
            Pilot files are read from
            results/auto_optimizer_job_inputs_v1/ and
            held-out files from results/auto_optimizer_heldout_v1/.
            These settings are validated by the backend.
          </p>
        </div>}
      </div>

      <div className="nimuniCleanAction">
        <div>
          <span className="nimuniCleanEyebrow">
            CURRENT OPTIMIZATION OBJECTIVE
          </span>
          <h3><Cpu size={19}/> Faster inference
            <ShieldCheck size={19}/></h3>
          <p>ONNX FP32 · measured performance and quality gates</p>
        </div>

        <button type="submit" className="nimuniPrimary">
          Continue to Mission Floor <ArrowRight size={18}/>
        </button>
      </div>

      {error&&<p className="nimuniError" role="alert">{error}</p>}
    </form>

    <p className="nimuniCleanFootnote">
      Continuing opens the laboratory. Real optimization starts
      only after you confirm the mission there.
      Unsupported models and missing evidence are rejected.
    </p>

    <section className="nimuniAdvanced nimuniCleanLegacy">
      <button type="button" aria-expanded={advanced}
        onClick={()=>setAdvanced(v=>!v)}>
        Other workflows
        {advanced?<ChevronUp size={16}/>:<ChevronDown size={16}/>}
      </button>
      {advanced&&<div className="nimuniLegacy">
        {renderAdvanced?.()}
      </div>}
    </section>
  </div>;
}
