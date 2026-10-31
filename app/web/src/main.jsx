import React,{useState,useEffect} from 'react';
import {createRoot} from 'react-dom/client';
import FleetControl from './FleetControl';
import EvidenceDashboard from './EvidenceDashboard';
import GameMissionFloor from './GameMissionFloor';
import './style.css';
import './game-mission-floor.css';

// NIMBO_CONNECTED_EVIDENCE_V16
function App(){
 const [view,setView]=useState('mission');
 const [registry,setRegistry]=useState(null);
 const [registryError,setRegistryError]=useState('');
 const [theme,setTheme]=useState('night');
 useEffect(()=>{fetch('http://127.0.0.1:8000/api/registry').then(async r=>{if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json()}).then(setRegistry).catch(e=>setRegistryError(e.message))},[]);
 const nav=[['mission','◈','Mission Control'],['fleet','▦','Fleet Control'],['registry','⬡','Model Registry'],['evidence','▤','Evidence Center']];
 return <div className={'shell theme-'+theme}>
   <aside><div className="logo"><div className="logoIcon">☁</div><div><b>NIMBO</b><small>AI ORCHESTRATION</small></div></div>
    {nav.map(([key,icon,name])=><div key={key} role="button" className={'nav '+(view===key?'active':'')} onClick={()=>setView(key)}>{icon} &nbsp; {name}</div>)}
    <p className="sidebarFoot">LOCAL DEVELOPMENT BUILD<br/>Model comparison · connected evidence</p>
   </aside>
   <main>
    {/* Keep the mission mounted while browsing evidence or fleet. */}
    <div style={{display:view==='mission'?'block':'none'}}><GameMissionFloor theme={theme} onThemeChange={()=>setTheme(t=>t==='night'?'day':'night')} registry={registry} active={view==='mission'} onSeeEvidence={()=>setView('evidence')}/></div>
    {view!=='mission'&&<div style={{padding:'28px clamp(16px,3vw,40px)',minHeight:'100vh'}}>
      {view==='fleet'?<FleetControl onBack={()=>setView('mission')}/>:view==='evidence'?<EvidenceDashboard onBack={()=>setView('mission')}/>:<section className="panel registry"><div className="panelTitle">MODEL REGISTRY <span>REAL CONFIGURATION</span></div><p>Configured models and connection state, not imaginary running agents.</p><div className="modelList">{registry?.models?.map(m=><div className="modelItem" key={m.id}><div><b>{m.id}</b><small>{m.workload} · {m.runtime}</small></div><span>{m.status}</span></div>)||registryError||'Loading registry…'}</div></section>}
     </div>}
   </main>
 </div>
}
createRoot(document.getElementById('root')).render(<App/>);
