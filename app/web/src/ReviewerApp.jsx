import React, {useEffect, useState} from 'react';
import {ArrowRight, History, Moon, Sun} from 'lucide-react';
import NimboLanding from './NimboLanding';
import UnifiedAutoMission from './UnifiedAutoMission';
import './unified-nimbo.css';
import './reviewer.css';

export default function ReviewerApp() {
  const [view, setView] = useState('home');
  const [theme, setTheme] = useState('day');
  const [cases, setCases] = useState([]);
  const [selected, setSelected] = useState('');
  const [error, setError] = useState('');
  const [mission, setMission] = useState(null);
  const workload = cases.find(item => item.id === selected);
  const toggleTheme = () => setTheme(value => value === 'day' ? 'night' : 'day');

  useEffect(() => {
    document.documentElement.classList.toggle('nimbo-dark', theme === 'night');
    document.documentElement.classList.toggle('nimbo-light', theme === 'day');
    document.documentElement.style.colorScheme = theme === 'night' ? 'dark' : 'light';
  }, [theme]);

  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/reviewer/catalog', {signal: controller.signal}).then(async response => {
      if (!response.ok) throw Error('Could not load supported workloads.');
      const catalog = await response.json();
      setCases(catalog.cases);
      setSelected(catalog.cases[0]?.id ?? '');
    }).catch(err => {
      if (!controller.signal.aborted) setError(err.message);
    });
    return () => controller.abort();
  }, []);

  if (view === 'home') {
    return <NimboLanding reviewer theme={theme} onThemeChange={toggleTheme} onEnter={() => setView('workspace')}/>;
  }
  return <main className={`reviewerShell theme-${theme}`}>
    <header className="gameHeadline">
      <div>
        <span className="gameEyebrow">NIMBO / PUBLIC REVIEWER</span>
        <h1>{mission ? 'Future Garden Mission Floor' : 'NIMBO Workspace'}</h1>
      </div>
      <div className="gameHeadActions">
        <button onClick={toggleTheme}>{theme === 'night' ? <Sun size={15}/> : <Moon size={15}/>} {theme === 'night' ? 'Day Lab' : 'Night Lab'}</button>
      </div>
    </header>
    {mission ? <UnifiedAutoMission reviewer config={mission} theme={theme} onBack={() => setMission(null)}/> :
      <div className="nimuniWorkspace nimuniCleanWorkspace">
        <header className="nimuniIntro nimuniCleanIntro">
          <h2>Compare two runtimes on the same model.</h2>
          <p>Run the PyTorch baseline and ONNX FP32 candidate on prepared labeled data. The deterministic optimizer measures both and applies the quality and performance gates.</p>
        </header>
        <form className="nimuniForm nimuniCleanForm" onSubmit={event => {
          event.preventDefault();
          if (workload) setMission(workload);
        }}>
          <div className="nimuniCleanSections">
            <section className="nimuniCleanPanel">
              <h3>Model and evaluation data</h3>
              <label>Available workload
                <select value={selected} onChange={event => setSelected(event.target.value)} disabled={!workload}>
                  {!cases.length && <option value="">Loading supported workload</option>}
                  {cases.map(item => <option key={item.id} value={item.id}>DistilBERT · SST-2 sentiment</option>)}
                </select>
              </label>
              {workload && <p className="nimuniCleanHelp">{workload.pilot} pilot examples and {workload.heldout} separate held-out examples. One CPU thread per runtime.</p>}
            </section>
            {workload && <section className="nimuniCleanPanel">
              <h3>Pinned inputs</h3>
              <p><a href={`https://huggingface.co/${workload.model}/tree/${workload.revision}`}>DistilBERT model revision</a></p>
              <p><a href={`https://huggingface.co/datasets/${workload.dataset}/tree/${workload.dataset_revision}`}>Public SST-2 dataset revision</a></p>
            </section>}
          </div>
          {error && <p role="alert">{error}</p>}
          <div className="nimuniCleanAction">
            <button type="button" onClick={() => setMission({replay: true})}><History size={16}/> Watch recorded replay</button>
            <button className="nimuniPrimary" type="submit" disabled={!workload}>Continue to Mission Floor <ArrowRight size={18}/></button>
          </div>
        </form>
        <p className="nimuniCleanFootnote">Hosted Qwen agents are unavailable. A live job can run for up to six minutes. Results are available in this browser for one hour.</p>
      </div>}
  </main>;
}
