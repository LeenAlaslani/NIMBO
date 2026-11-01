import React,{useState,useEffect} from 'react';
import {createRoot} from 'react-dom/client';
import FleetControl from './FleetControl';
import EvidenceDashboard from './EvidenceDashboard';
import GameMissionFloor from './GameMissionFloor';
import AutoOptimizerLive from './AutoOptimizerLive';
import NimboLanding from './NimboLanding';
import './style.css';
import './game-mission-floor.css';
import './nimbo-design-system-v1.css';
import {LayoutDashboard,Layers3,Database,Boxes,Sun,Moon,Cpu} from 'lucide-react';

// NIMBO_CONNECTED_EVIDENCE_V16
function App(){
 const [view,setView]=useState('home');
 const [registry,setRegistry]=useState(null);
 const [registryError,setRegistryError]=useState('');
 const [theme,setTheme]=useState(()=>{try{return localStorage.getItem('nimbo-ui-theme-v1')==='night'?'night':'day'}catch{return 'day'}});
 useEffect(()=>{try{localStorage.setItem('nimbo-ui-theme-v1',theme)}catch{};document.documentElement.style.colorScheme=theme==='night'?'dark':'light';document.documentElement.classList.toggle('nimbo-dark',theme==='night');document.documentElement.classList.toggle('nimbo-light',theme==='day')},[theme]);
 useEffect(()=>{fetch('http://127.0.0.1:8000/api/registry').then(async r=>{if(!r.ok)throw Error(`HTTP ${r.status}`);return r.json()}).then(setRegistry).catch(e=>setRegistryError(e.message))},[]);
 const nav=[['home',LayoutDashboard,'Overview'],['mission',LayoutDashboard,'Workspace'],['auto',Cpu,'Auto-Optimizer'],['fleet',Layers3,'Fleet Control'],['registry',Boxes,'Model Registry'],['evidence',Database,'Evidence Center']];
 if(view==='home')return <NimboLanding theme={theme} onEnter={()=>setView('mission')} onThemeChange={()=>setTheme(t=>t==='night'?'day':'night')}/>;
 return <div className={'shell theme-'+theme}>
   <aside className="nimboSidebar" aria-label="Primary navigation">
    <button type="button" className="logo nimboFullBrand nimboBrandHomeButton" onClick={()=>setView('home')} aria-label="Go to NIMBO home"><img className="nimboOfficialLogo" src="/nimbo-brand-mark.png" alt="NIMBO" title="Official NIMBO logo"/></button>
    <div className="nimboNavLabel">PLATFORM</div>
    <nav className="nimboNavList" aria-label="Main sections">{nav.map(([key,Icon,name])=><button type="button" key={key} className={'nav '+(view===key?'active':'')} onClick={()=>setView(key)} aria-current={view===key?'page':undefined}><Icon size={18} strokeWidth={1.8}/><span>{name}</span></button>)}</nav>
    <div className="nimboSidebarBottom"><div className="nimboModeLabel"><span className="nimboStatusIndicator"/> Local environment <span className="nimboStatusContext">configured</span></div><button className="nimboThemeToggle" type="button" onClick={()=>setTheme(t=>t==='night'?'day':'night')} aria-label={theme==='night'?'Switch to light mode':'Switch to dark mode'}>{theme==='night'?<Sun size={17}/>:<Moon size={17}/>}<span>{theme==='night'?'Light appearance':'Dark appearance'}</span></button><p className="sidebarFoot">Performance you can account for.<br/>Evidence you can inspect.</p></div>
   </aside>
   <main>
    {/* Keep the mission mounted while browsing evidence or fleet. */}
    <div style={{display:view==='mission'?'block':'none'}}><GameMissionFloor theme={theme} onThemeChange={()=>setTheme(t=>t==='night'?'day':'night')} registry={registry} active={view==='mission'} onSeeEvidence={()=>setView('evidence')}/></div>
    {view!=='mission'&&<div style={{padding:'28px clamp(16px,3vw,40px)',minHeight:'100vh'}}>
      {view==='auto'?<AutoOptimizerLive/>:view==='fleet'?<FleetControl onBack={()=>setView('mission')}/>:view==='evidence'?<EvidenceDashboard onBack={()=>setView('mission')}/>:<section className="panel registry"><div className="panelTitle">MODEL REGISTRY <span>REAL CONFIGURATION</span></div><p>Configured models and connection state, not imaginary running agents.</p><div className="modelList">{registry?.models?.map(m=><div className="modelItem" key={m.id}><div><b>{m.id}</b><small>{m.workload} · {m.runtime}</small></div><span>{m.status}</span></div>)||registryError||'Loading registry…'}</div></section>}
     </div>}
   </main>
 </div>
}
createRoot(document.getElementById('root')).render(<App/>);
