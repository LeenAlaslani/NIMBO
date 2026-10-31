import React from 'react';
import {ArrowRight, Moon, Sun, CheckCircle2, Cpu, ShieldCheck} from 'lucide-react';
import './nimbo-landing-v1.css';

/** A real home screen: no request is sent and no model is selected. */
export default function NimboLanding({theme,onEnter,onThemeChange}){
 return <div className={'nimboIntro theme-'+theme}>
   <header className="nimboIntroHeader">
     <span className="nimboIntroIdentity">NIMBO <span className="nimboIntroSeparator"/> AI OPTIMIZATION LAB</span>
     <button className="nimboIntroTheme" type="button" onClick={onThemeChange} aria-label={theme==='night'?'Use light appearance':'Use dark appearance'}>
       {theme==='night'?<Sun size={17}/>:<Moon size={17}/>}<span>{theme==='night'?'Light':'Dark'} mode</span>
     </button>
   </header>
   <main className="nimboIntroMain">
     <div className="nimboIntroAsset"><img src="/nimbo-brand-mark.png" alt="NIMBO — Neural Intelligence for Multimodal Budget Optimization"/></div>
     <div className="nimboIntroContent">
       <span className="nimboIntroKicker">DELIVER AI WITH LESS COMPUTE</span>
       <h1>Optimize AI.<br/><strong>Prove the decision.</strong></h1>
       <p>Compare real execution paths. Preserve measurable quality. Make resource-aware choices with evidence you can inspect.</p>
       <button type="button" className="nimboIntroEnter" onClick={onEnter}>Enter workspace <ArrowRight size={19}/></button>
       <div className="nimboIntroFootnote">LOCAL DEMONSTRATOR <span aria-hidden="true">·</span> NO AUTOMATIC DEPLOYMENT</div>
     </div>
   </main>
   <footer className="nimboIntroFooter"><span><Cpu size={16}/> Measured execution</span><span><ShieldCheck size={16}/> Quality-aware gates</span><span><CheckCircle2 size={16}/> Auditable decisions</span></footer>
 </div>
}
