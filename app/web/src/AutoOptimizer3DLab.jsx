/** M8b: existing NIMBO Mission Floor 3D scene bound ONLY to M7's real events. */
import React, { useEffect, useMemo, useState } from 'react';
import { ArrowRight, Eye, RadioTower } from 'lucide-react';
import { LabScene } from './GameMissionFloor';
import { eventWorker, realEvents, sceneFromTimeline } from './autoOptimizer3DState.mjs';
import './auto-optimizer-3d.css';

const WORKER_NAMES = { captain: 'CAPTAIN', model: 'MAKO', vera: 'VERA', metri: 'METRI' };

export default function AutoOptimizer3DLab({ job, events, connected, theme = 'night' }) {
  const [selectedSeq, setSelectedSeq] = useState(null);
  const [follow, setFollow] = useState(true);
  const [visible, setVisible] = useState(false);
  useEffect(() => { setSelectedSeq(null); setVisible(false); }, [job?.job_id]);
  const timeline = useMemo(() => realEvents(events), [events]);
  const scene = sceneFromTimeline(job, timeline, selectedSeq);
  const current = scene.event;
  const worker = scene.worker || 'captain';
  const label = scene.worker ? WORKER_NAMES[scene.worker] : '—';
  const active = Boolean(connected && current && job && scene.worker);
  function inspectWorker(id) {
    const matching = [...timeline].reverse().find(e => eventWorker(e, timeline) === id);
    if (matching) setSelectedSeq(matching.seq);
  }
  const status = !connected ? 'API DISCONNECTED · LAST KNOWN DATA' :
                 scene.historical ? 'RECORDED EVENT · MANUAL INSPECTION' :
                 scene.running ? 'LIVE BACKEND EVENT' :
                 scene.finished ? 'COMPLETED JOB · RECORDED EVENT' : 'WAITING FOR BACKEND';
  return <section className="nim3dPanel" aria-label="Real optimization events inside the 3D Mission Floor">
    <div className="nim3dHeading">
      <div><span className="nim3dEyebrow"><RadioTower size={15}/> MISSION FLOOR / AUTO-OPTIMIZER</span>
        <h2>3D Lab — Evidence-bound robots</h2>
        <p>Same NIMBO workstations and characters. The selected worker is derived from an actual backend event, not a timer or fabricated agent action.</p>
      </div>
      <button type="button" className="nim3dToggle" onClick={() => setVisible(x => !x)}>
        <Eye size={16}/>{visible ? 'Hide 3D Lab' : 'Open 3D Lab'}
      </button>
    </div>
    {visible && <>
      <div className="nim3dControls">
        <span className={scene.running ? 'nim3dLive' : 'nim3dRecorded'}>{status}</span>
        <div>
          <button type="button" onClick={() => setSelectedSeq(null)} disabled={selectedSeq === null}>Latest actual event</button>
          <button type="button" onClick={() => setFollow(x => !x)}>{follow ? 'Camera: Follow worker' : 'Camera: Overview'}</button>
        </div>
      </div>
      <div className="nim3dScene">
        {active ? <LabScene
          veraLive={null} target={worker} onSelect={inspectWorker}
          busy={scene.running} route="" comparison={null} imageComparison={null}
          theme={theme} offline={false} liveExecute={scene.running} started={true}
          onArrive={() => {}} missionSeed={job.job_id}
          explainingId={worker} thinkingId={null}
          speech={{ title: current.message.slice(0, 135) }}
          celebrating={scene.verifiedDelivery} metrics={{}}
          events={timeline.map(e => ({stage:e.stage, message:e.message, at_ms:null}))}
          quality={scene.verifiedDelivery ? 'VERIFIED_OFFLINE_RELOAD' : 'NOT_APPROVED'}
          flow={['captain','model','vera','metri']} followWorker={follow}
        /> : <div className="nim3dPlaceholder">{connected ? 'Load an actual Job ID to activate the lab.' : 'M7 API is not connected. No worker activity is claimed.'}</div>}
      </div>
      <div className="nim3dEvidence" aria-live="polite">
        {current ? <>
          <b>EVENT #{current.seq} · {String(current.stage).toUpperCase()} · {label}</b>
          <span>{current.message}</span>
          <small>Backend state: {current.state} · Job: {scene.outcome}</small>
        </> : <span>No recorded stage to display.</span>}
      </div>
      {timeline.length > 0 && <div className="nim3dHistory" aria-label="Select one recorded event">
        {timeline.slice(-25).map(e => <button type="button" key={e.seq}
          className={e.seq === current?.seq ? 'active' : ''}
          onClick={() => setSelectedSeq(e.seq)}
          title={e.message}>#{e.seq} {e.stage} <ArrowRight size={11}/></button>)}
      </div>}
      <p className="nim3dFoot">Historic events do not mean the model is running again. Animations represent recorded backend stages. Failed and rejected jobs are never shown as successful delivery.</p>
    </>}
  </section>;
}
