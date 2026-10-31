// NIMBO_V20_SAME_MODEL_TRIAL
// NIMBO_V20_2_ONNX_TRIAL
// NIMBO_V21_INDEPENDENT_VALIDATION
// NIMBO_V19_ANIMATED_OPERATIONS_LAB
// NIMBO_V18_ACTUAL_UI_UPGRADE
// NIMBO_V17_OPERATIONAL_MISSION
// NIMBO_CONNECTED_EVIDENCE_V16
// NIMBO_COMPANY_WORKSPACE_V15 — company configuration + real paired model evidence.
import React,{useRef,useState,useEffect,useMemo} from 'react';
import {createPortal} from 'react-dom';
import CompanyWorkspace from './CompanyWorkspace';
import AdaptiveImagePanelV24 from './AdaptiveImagePanelV24';
import './nimbo-vera-v27.css';
// NIMBO_V27_VERA_STAGE
import './nimbo-adaptive-image-v24.css';
// NIMBO_ADAPTIVE_IMAGE_V24
import {Canvas,useFrame,useThree} from '@react-three/fiber';
import {Html,Text,ContactShadows,RoundedBox} from '@react-three/drei';
import OperationsEnvironmentV19 from './OperationsEnvironmentV19';
import LivingWorkerV19 from './LivingWorkerV19';
import WorkstationV19 from './WorkstationV19';
import MissionWallV19 from './MissionWallV19';
import * as THREE from 'three';
import {Play,RotateCcw,Sun,Moon,ChevronRight,ChevronLeft,Activity,ShieldAlert,Check,BookOpen,Zap,ArrowUpRight,Info,Pause,Expand,Minimize2,MapPin,MoveRight} from 'lucide-react';
import './game-mission-floor.css';
import './company-workspace-v15.css';
import {readableRoute} from './evidencePresentation.mjs';

const API='http://127.0.0.1:8000';
const STEPS=['captain','toki','routa','model','vera','metri'];
const ROUTES={classification:['captain','routa','model','vera','metri'],generation:['captain','routa','model','vera','metri'],context:['captain','toki','routa','model','vera','metri'],image:['captain','routa','model','vera','metri']};
const PEOPLE={
 captain:{name:'CAPTAIN',role:'Mission intake',x:-4.7,z:2.2,color:'#8cc9ff',icon:'✦'},
 toki:{name:'TOKI',role:'Context review',x:-3.25,z:-1.25,color:'#bd9aff',icon:'✂'},
 routa:{name:'ROUTA',role:'Model router',x:-.25,z:-1.65,color:'#7fc4ff',icon:'◇'},
 model:{name:'MAKO',role:'Model execution',x:3.05,z:-1.35,color:'#60e3d5',icon:'⚡'},
 vera:{name:'VERA',role:'Quality guardrail',x:4.7,z:2.0,color:'#ffcd7e',icon:'✓'},
 metri:{name:'METRI',role:'Evidence report',x:.55,z:3.05,color:'#8bedad',icon:'▥'},
};
const MISSIONS=[
 {id:'route',title:'Smart Routing',short:'Use an adaptive model path',mode:'classification',prompt:'The movie was beautiful but the story was painfully slow.',hint:'Compare the recorded model route and warm inference latency.'},
 {id:'qwen',title:'Text Generation',short:'Run the connected generator',mode:'generation',prompt:'Explain why model confidence is not proof of factual correctness in three sentences.',hint:'Inspect real Qwen inference events and any available guardrails.'},
 {id:'guard',title:'Quality Guardrails',short:'Inspect quality evidence',mode:'classification',prompt:'I absolutely loved the experience.',hint:'Compare model confidence with independent verification availability.'},
 {id:'context',title:'Context Optimization',short:'Review token policy',mode:'generation',prompt:'Describe three ways an AI service can reduce unnecessary compute without losing response quality.',hint:'Token optimization is only reported if the live backend applies it.'},
 {id:'image',title:'Image Pipeline',short:'Check the GPU worker',mode:'image',prompt:'A bright orange lifebuoy hanging on a boat railing at sea, realistic photograph.',hint:'Requires a real connected image gateway.'},
 {id:'quant',title:'Quantization',short:'Compare precision variants',mode:'not_connected',locked:true,prompt:'',hint:'Register full-precision and quantized variants to evaluate.'},
 {id:'distill',title:'Distillation',short:'Compare model variants',mode:'not_connected',locked:true,prompt:'',hint:'Connect a teacher and student model with comparable quality evaluations.'},
];
const COLORS={night:{sky:'#081a33',floor:'#152e4b',wall:'#244361',line:'#375878',desk:'#54718d',metal:'#6d90aa',text:'#d9ebff',screen:'#0f2742',ambient:1.7,accent:'#7dd9e7'},day:{sky:'#deedf6',floor:'#adc5d8',wall:'#8daac2',line:'#648aab',desk:'#63819a',metal:'#b7d0e1',text:'#15354f',screen:'#143b5c',ambient:1.45,accent:'#147dba'}};
function Mat({color,emissive,intensity=0,roughness=.43}){return <meshStandardMaterial color={color} emissive={emissive||'#000000'} emissiveIntensity={intensity} metalness={.18} roughness={roughness}/>}
function GuidePath({target,flow=STEPS}){
 const to=PEOPLE[target],idx=flow.indexOf(target);
 const source=idx>0?PEOPLE[flow[idx-1]]:null;
 const start=source?[source.x,source.z+.7]:[-6.15,4.75];
 const end=[to.x,to.z+.7];const dx=end[0]-start[0],dz=end[1]-start[1],len=Math.hypot(dx,dz);
 const mid=[(start[0]+end[0])/2,.05,(start[1]+end[1])/2];
 const angle=Math.atan2(dx,dz),ref=useRef();
 useFrame(({clock})=>{if(ref.current)ref.current.opacity=.40+.35*Math.sin(clock.elapsedTime*4)**2});
 return <group>
  <mesh position={mid} rotation={[-Math.PI/2,0,angle]}><planeGeometry args={[.14,len]}/><meshBasicMaterial ref={ref} color={to.color} transparent opacity={.7} depthWrite={false}/></mesh>
  <mesh position={[end[0],.055,end[1]+.1]} rotation={[-Math.PI/2,0,0]}><ringGeometry args={[.58,.7,32]}/><meshBasicMaterial color={to.color} transparent opacity={.72}/></mesh>
 </group>
}
function CourierCapsule({target,theme,enabled,onArrive,compactRatio=1}){
 const g=useRef(),leg1=useRef(),leg2=useRef(),lastArrival=useRef(null);
 useEffect(()=>{lastArrival.current=null},[target]);
 useFrame(({clock},delta)=>{
  if(!g.current||!enabled)return;const d=Math.min(delta,.05),p=PEOPLE[target];
  const dx=p.x-g.current.position.x,dz=p.z+1.3-g.current.position.z,dist=Math.hypot(dx,dz);
  if(dist>.055){const speed=Math.min(dist,d*1.65);g.current.position.x+=dx/dist*speed;g.current.position.z+=dz/dist*speed;g.current.rotation.y=THREE.MathUtils.damp(g.current.rotation.y,Math.atan2(dx,dz),7,d)}
  else if(lastArrival.current!==target){lastArrival.current=target;onArrive?.(target)}
  g.current.position.y=Math.sin(clock.elapsedTime*5)*.04;
  if(leg1.current)leg1.current.rotation.x=dist>.1?Math.sin(clock.elapsedTime*10)*.4:0;
  if(leg2.current)leg2.current.rotation.x=dist>.1?-Math.sin(clock.elapsedTime*10)*.4:0;
 });
 const cloudColor=theme==='day'?'#fffef8':'#e2f1ff';
 return <group ref={g} position={[-6.15,0,4.75]} scale={.59}>
  <mesh castShadow position={[0,1.1,0]}><sphereGeometry args={[.56,20,16]}/><Mat color={cloudColor}/></mesh>
  <mesh position={[0,1.42,-.05]}><sphereGeometry args={[.35,18,12]}/><Mat color={cloudColor}/></mesh>
  <mesh position={[0,1.08,.51]}><boxGeometry args={[.65,.39,.09]}/><Mat color="#173250"/></mesh>
  {[-.16,.16].map(x=><mesh key={x} position={[x,1.1,.57]}><sphereGeometry args={[.07,12,10]}/><meshBasicMaterial color="#6ae5da"/></mesh>)}
  <mesh position={[0,.60,0]}><sphereGeometry args={[.31,14,12]}/><Mat color={cloudColor}/></mesh>
  {[-.2,.2].map((x,i)=><group key={x} ref={i===0?leg1:leg2} position={[x,.40,.04]}><mesh position={[0,-.20,0]}><capsuleGeometry args={[.12,.16,4,8]}/><Mat color="#c6d9eb"/></mesh></group>)}
  <mesh position={[0,1.92,0]} castShadow><boxGeometry args={[.65*compactRatio,.43,.48*compactRatio]}/><Mat color="#b57a3f"/></mesh>
  <mesh position={[0,2.15,0]}><boxGeometry args={[.68*compactRatio,.05,.5*compactRatio]}/><Mat color="#e2b56d"/></mesh>
  <pointLight position={[0,.4,.1]} color="#f9c477" intensity={.8} distance={1.4}/>
 </group>;
}
const stageLabels={analyze:'INTAKE',route:'ROUTING',execute:'INFERENCE',guardrail:'GUARDRAIL',verify:'QUALITY',deliver:'DELIVERY'};
function qualityEvidenceLabel(report){
 const c=report?.comparison;
 const b=c?.baseline?.accuracy_on_reference_labels??c?.baseline?.accuracy_on_supplied_labels;
 const a=c?.candidate?.accuracy_on_reference_labels??c?.candidate?.accuracy_on_supplied_labels;
 if(c?.all_labels_present&&typeof b==='number'&&Number.isFinite(b)&&typeof a==='number'&&Number.isFinite(a)){
   return report?.optimization?.validation==='heldout_repeated_v21'?'HELD-OUT ACCURACY MEASURED':'LABEL ACCURACY MEASURED';
 }
 return report?.guardrail?.action||'UNVERIFIED';
}
function qualityEvidenceDetail(report){
 if(report?.comparison?.all_labels_present){
   return report?.optimization?.validation==='heldout_repeated_v21'
    ?'IMDb held-out reference-label accuracy evaluated; not independently audited or production certified.'
    :'Reference-label accuracy evaluated for this local comparison; labels not independently audited.';
 }
 return 'No complete reference labels for this request. Accuracy cannot be evaluated.';
}
function numOrNull(v){return typeof v==='number'&&Number.isFinite(v)?v:null}
function getTokenPair(metrics){
 const before=numOrNull(metrics?.input_tokens_before??metrics?.original_input_tokens);
 const after=numOrNull(metrics?.input_tokens_after??metrics?.optimized_input_tokens);
 return before!==null&&after!==null&&before>0&&after>=0?[before,after]:null;
}
function CameraFollow({target,boardFocus=false}){const {camera}=useThree();const dest=useMemo(()=>new THREE.Vector3(),[]);const look=useRef(new THREE.Vector3(0,1.25,-.9));useFrame((_,delta)=>{const p=PEOPLE[target];dest.set(...(boardFocus?[0.6,3.9,5.85]:[3.15,7.35,15.25]));camera.position.lerp(dest,Math.min(1,delta*1.6));const desired=new THREE.Vector3(boardFocus?0:p.x*.12,boardFocus?2.65:1.25,boardFocus?-4.45:-.9);look.current.lerp(desired,Math.min(1,delta*1.75));camera.lookAt(look.current);});return null}
function LabScene({veraLive,target,onSelect,busy,route,comparison,imageComparison,theme,offline,liveExecute,started,onArrive,missionSeed,explainingId,thinkingId,speech,celebrating,metrics,events=[],guideStep=0,quality='UNVERIFIED',flow=STEPS,boardFocus=false}){const p=COLORS[theme];return <Canvas camera={{position:[3.15,7.35,15.25],fov:47}} shadows dpr={[1,1.5]} gl={{antialias:true}}>
 <color attach="background" args={[p.sky]}/><ambientLight intensity={p.ambient}/><hemisphereLight args={['#ffffff',theme==='day'?'#7396ad':'#102f4c',theme==='day'?.75:.6]}/><directionalLight castShadow position={[2,9,5]} intensity={theme==='day'?2:3} shadow-mapSize={[512,512]}/>
 <OperationsEnvironmentV19 theme={theme}/><MissionWallV19 route={route} reported={Boolean(route)} theme={theme} metrics={metrics} comparison={comparison} imageComparison={imageComparison} events={events} guideStep={guideStep} quality={quality} busy={busy}/><GuidePath target={target} flow={flow}/><CourierCapsule key={missionSeed} target={target} theme={theme} enabled={started} onArrive={onArrive} compactRatio={getTokenPair(metrics)?Math.max(.48,Math.min(1,getTokenPair(metrics)[1]/getTokenPair(metrics)[0])):1}/>
 {STEPS.map(id=><React.Fragment key={id}><WorkstationV19 id={id} person={PEOPLE[id]} active={id===target} busy={busy&&liveExecute} explaining={explainingId===id} theme={theme} route={route} offline={id==='model'&&offline}/><LivingWorkerV19 hideLabels={boardFocus} id={id} person={PEOPLE[id]} active={id===target||(id==='vera'&&veraLive?.active)} busy={(busy&&liveExecute)||(id==='vera'&&veraLive?.active)} explaining={explainingId===id} thinking={thinkingId===id||(id==='vera'&&veraLive?.active)} speech={id==='vera'&&veraLive?.active?{title:({queued:'Queued for verification',checking_fast:'Checking Fast image',generating_standard:'Running fallback',checking_standard:'Checking Standard'})[veraLive.state]||'Checking image'}:explainingId===id?speech:null} celebrating={celebrating && id==='metri'} theme={theme} offline={id==='model'&&offline} onSelect={onSelect}/></React.Fragment>)}
 <ContactShadows position={[0,.018,0]} scale={17} opacity={theme==='day'?.25:.35} blur={3} far={4}/><CameraFollow target={target} boardFocus={boardFocus}/>
 </Canvas>}

const DONE_HINTS={captain:'Mission accepted. Follow the glowing path to TOKI.',toki:'Context reviewed. No compression is applied unless the backend reports it.',routa:'The actual model route will come from the backend. Starting real inference.',model:'The model run has finished. You can inspect the measured result.',vera:'Quality status comes from the real result, not a fake check.',metri:'Comparison evidence recorded. Open METRI to review model timings and supplied-label quality.'};
const shortEvent=(e)=>String(e?.message||'').slice(0,140);
function prettyOutput(answer){if(!answer)return null;const parts=String(answer).split(/(\*\*[^*]+\*\*)/g);return parts.map((p,i)=>p.startsWith('**')&&p.endsWith('**')?<strong key={i}>{p.slice(2,-2)}</strong>:p)}

export default function GameMissionFloor({theme,onThemeChange,registry,onSeeEvidence,active=true}){
 // NIMBO_IMAGE_MISSION_V22
 const [fastRating,setFastRating]=useState('');const [standardRating,setStandardRating]=useState('');const [imageReviewNote,setImageReviewNote]=useState('');
 const [showLiveProgress,setShowLiveProgress]=useState(false);const [veraLive,setVeraLive]=useState(null);const [boardFocus,setBoardFocus]=useState(false);const [workspaceOpen,setWorkspaceOpen]=useState(true);const [workConfig,setWorkConfig]=useState(null);const [inspect,setInspect]=useState(null);const [modelOps,setModelOps]=useState(false);const [selectedEval,setSelectedEval]=useState(null);const [uploadPreview,setUploadPreview]=useState(null);const [mission,setMission]=useState(MISSIONS[0]);const [arrived,setArrived]=useState(false);const [story,setStory]=useState(null);const [thinkingId,setThinkingId]=useState(null);const [dialogReady,setDialogReady]=useState(false);const [missionSeed,setMissionSeed]=useState(0);const [fullScreen,setFullScreen]=useState(false);const shellRef=useRef(null);const worldRef=useRef(null);const [input,setInput]=useState(MISSIONS[0].prompt);const [target,setTarget]=useState('captain');const [started,setStarted]=useState(false);const [liveEvents,setLiveEvents]=useState([]);const [report,setReport]=useState(null);const [busy,setBusy]=useState(false);const [error,setError]=useState('');const [route,setRoute]=useState('');const [drawer,setDrawer]=useState(false);const [inspected,setInspected]=useState(false);const [guide,setGuide]=useState('Choose a mission, then talk to CAPTAIN.');const [requestDone,setRequestDone]=useState(false);const [liveExecute,setLiveExecute]=useState(false);const [replay,setReplay]=useState(false);const [playSpeed,setPlaySpeed]=useState(1);const [eventIndex,setEventIndex]=useState(-1);const [readable,setReadable]=useState([]);
 const session=useRef(0),abort=useRef(null),replaySeq=useRef(0);const timers=useRef([]);const currentReport=useRef(null);useEffect(()=>{currentReport.current=report},[report]);
 const offline=mission.mode==='image'&&!registry?.image_gateway_connected;
 function stopTimers(){timers.current.forEach(clearTimeout);timers.current=[];replaySeq.current++}
 function reset(all=true){setFastRating('');setStandardRating('');setImageReviewNote('');setShowLiveProgress(false);setVeraLive(null);setBoardFocus(false);setMissionSeed(n=>n+1);setArrived(false);setStory(null);setThinkingId(null);setDialogReady(false);currentReport.current=null;session.current++;abort.current?.abort();abort.current=null;stopTimers();setLiveEvents([]);setReport(null);setBusy(false);setError('');setRoute('');setDrawer(false);setInspect(null);setInspected(false);setRequestDone(false);setLiveExecute(false);setReplay(false);setReadable([]);setEventIndex(-1);setTarget('captain');setStarted(false);setGuide('Click CAPTAIN to receive the mission.');}
 useEffect(()=>()=>{session.current++;abort.current?.abort();stopTimers()},[]);
 useEffect(()=>{const update=()=>setFullScreen(document.fullscreenElement===worldRef.current);document.addEventListener('fullscreenchange',update);return()=>document.removeEventListener('fullscreenchange',update)},[]);
 useEffect(()=>{if(!active){setDrawer(false);setInspect(null);setModelOps(false);if(document.fullscreenElement===worldRef.current)document.exitFullscreen().catch(()=>{});}},[active]);
 async function toggleFullScreen(){try{if(document.fullscreenElement)await document.exitFullscreen();else await worldRef.current?.requestFullscreen?.()}catch(e){setGuide('Fullscreen unavailable in this browser. Use browser full-screen mode instead.')}}
 function storyFor(id){
   if(id==='captain')return {speaker:'CAPTAIN',title:'Mission received',body:'Task received. We can route it to the right station.',type:'guided'};
   if(id==='toki')return {speaker:'TOKI',title:'Context review',body:'Context optimization is not active in this run. I will only report reductions measured by the backend.',type:'guided'};
   if(id==='routa')return currentReport.current?{speaker:'ROUTA',title:currentReport.current.comparison?'Model pair compared':'Route recorded',body:currentReport.current.comparison?`Both registered models completed on ${currentReport.current.comparison.sample_count} identical inputs. Open the board for measured latency.`:`Backend selected ${currentReport.current.route||'an unreported route'}. The real model request has finished.`,type:'measured'}:{speaker:'ROUTA',title:'Request submitted',body:'The model service is processing the task. Its reported route and timing will appear in the live signals.',type:'live'};
   if(id==='model')return {speaker:'MAKO',title:'Model execution reviewed',body:currentReport.current?.comparison?`Candidate average: ${currentReport.current.comparison.candidate.mean_inference_ms} ms; baseline: ${currentReport.current.comparison.baseline.mean_inference_ms} ms. Warm CPU inference only.`:currentReport.current?`Actual route: ${currentReport.current.route||'not reported'}. Measured request: ${currentReport.current.metrics?.end_to_end_ms??'unavailable'} ms. Animation time is separate from execution time.`:'Inference has not finished.',type:'measured'};
   if(id==='vera'&&currentReport.current?.adaptive)return {speaker:'VERA',title:'Autonomous quality decision',body:`${currentReport.current.adaptive.decision||'UNVERIFIED'}: ${currentReport.current.adaptive.verdict_reason||'No complete quality certification.'} Model success is not assumed.`,type:'evidence'};
   if(id==='vera')return {speaker:'VERA',title:'Quality evidence',body:currentReport.current?.comparison?.all_labels_present?`Candidate matched ${Math.round(currentReport.current.comparison.candidate.accuracy_on_supplied_labels*100)}% of supplied labels. Labels were not independently audited.`:`${currentReport.current?.guardrail?.action||'UNVERIFIED'}: A result exists, but independent accuracy certification is unavailable.`,type:'evidence'};
   return {speaker:'METRI',title:'Evidence receipt',body:'Results recorded. Compare a baseline before claiming savings.',type:'evidence'};
 }
 const onArrive=id=>{setArrived(prev=>id===target?true:prev)};
 function pick(m){if(m.locked){setModelOps(true);setSelectedEval(m.id);return;} reset();setMission(m);setInput(m.prompt);setGuide(`Selected: ${m.title}. Click ENTER LAB, then CAPTAIN.`)}
 function configureCompany(c){
   reset();
   setWorkConfig(c);
   const selected=MISSIONS.find(m=>m.mode===c.problem&&!m.locked) || MISSIONS[0];
   setMission({...selected,title:(c.operation==='optimize'||c.operation==='optimize_onnx'||c.operation==='validate_onnx')?'Same-model optimization':c.operation==='compare'?'Baseline comparison':selected.title});
   setInput(c.prompt);
   setWorkspaceOpen(false);
   setGuide('Your workload is configured. Follow PICO through the stations required for this operation.');
 }
 function backToWorkspace(){reset();setWorkspaceOpen(true)}
 function enter(){reset();setStarted(true);setGuide('Follow the glowing path and click CAPTAIN to accept the task.');}
 function nextFrom(id){setArrived(false);const i=missionFlow.indexOf(id);const next=missionFlow[i+1];if(next){setTarget(next);setGuide(`Follow the illuminated path to ${PEOPLE[next].name}. ${next==='toki'?'Review context, without claiming optimization.':next==='routa'?'Start the real backend call here.':next==='model'?'Watch real model activity, then click the worker.':next==='vera'?'Inspect genuine quality evidence.':'Open the evidence receipt.'}`)}else{setDrawer(true);setGuide('Mission tour completed. Inspect the evidence and decide what NIMBO proved.');}}
 async function runBackend(){const requestId=++session.current;const controller=new AbortController();abort.current=controller;setBusy(true);setError('');setRequestDone(false);setLiveExecute(true);setGuide('Real inference started. Live backend events are appearing in the mission feed.');
 try{
  const paired=mission.mode==='classification'&&workConfig?.operation==='compare';
  const optimizing=mission.mode==='classification'&&(workConfig?.operation==='optimize'||workConfig?.operation==='optimize_onnx'||workConfig?.operation==='validate_onnx');
  const endpoint=workConfig?.operation==='validate_onnx'?'/nimbo/validate/v21/run':workConfig?.operation==='optimize_onnx'?'/nimbo/optimize/onnx/run':optimizing?'/nimbo/optimize/v20/run':paired?'/nimbo/bench/compare':mission.mode==='image'?(workConfig?.imageMode==='adaptive'?'/nimbo/image/adaptive/v24/start':'/nimbo/image/v22/generate'):'/api/run/stream';
  const res=await fetch(API+endpoint,{method:'POST',signal:controller.signal,headers:{'Content-Type':'application/json'},body:JSON.stringify(optimizing?{model_id:workConfig.model_id,phase:workConfig.trialPhase,sample_count:workConfig.sampleCount,rounds:workConfig.rounds||undefined,export_variant:workConfig.operation==='validate_onnx'?true:undefined,examples:workConfig.examples?.length?workConfig.examples:undefined}:{...(paired?{baseline_model:workConfig.baseline,candidate_model:workConfig.candidate,examples:workConfig.examples}:mission.mode==='image'?{prompt:input,mode:workConfig?.imageMode||'paired',seed:workConfig?.imageSeed??42}:{text:input,workload:mission.mode})})});
  if(!res.ok){const body=await res.json().catch(()=>({}));throw Error(body.detail||`API HTTP ${res.status}`)}
  if(mission.mode==='image'){
    const body=await res.json();if(requestId!==session.current)return;setReport(body);setRoute(body.route||'gateway');const trace=Array.isArray(body.trace)?body.trace:[];setLiveEvents(trace.map(x=>({stage:x.stage||x.step||'deliver',message:x.message||'Image received',at_ms:x.at_ms??null})));return;
  }
  if(!res.body)throw Error('Backend did not return an event stream');
  const reader=res.body.getReader(),decoder=new TextDecoder();let buffer='';
  function take(line){if(requestId!==session.current||!line.trim())return;const p=JSON.parse(line);if(p.type==='error')throw Error(p.message||'Model failed');if(p.type==='event'){
   const event={stage:p.stage,message:p.message,at_ms:p.at_ms};setLiveEvents(prev=>[...prev,event]);setReadable(prev=>[...prev,event]);
   if(p.stage==='route'&&(p.policy||p.model))setRoute(p.policy||p.model);
  }if(p.type==='result'){setReport(p.report);setRoute(p.report?.route||'');}}
  while(true){const {done,value}=await reader.read();if(done)break;buffer+=decoder.decode(value,{stream:true});let lines=buffer.split('\n');buffer=lines.pop();for(const line of lines){if(requestId!==session.current)return;take(line)}}
  buffer+=decoder.decode();if(buffer.trim())take(buffer);
 }catch(e){if(requestId===session.current&&e.name!=='AbortError'){setError(String(e.message));setGuide('Backend error: '+e.message)}}finally{if(requestId===session.current){setBusy(false);setRequestDone(true);setLiveExecute(false);setGuide('Model request finished. You can inspect its actual result at MAKO.')}}
 }
 function onBot(id){
  if(!started){setInspect(id);setGuide('Inspect this station, or start a mission.');return}
  if(id!==target){setInspect(id);return;}
  if(replay){setGuide('Recorded replay is active. Stop it before interacting.');return}
  if(!arrived){setGuide(`NIMBO is on its way to ${PEOPLE[target].name}.`);return}
  if(thinkingId||dialogReady)return;
  if(id==='model'&&busy){setShowLiveProgress(true);return}
  if(id==='model'&&!requestDone){setGuide('Waiting for a completed model request.');return}
  if(id==='vera'&&!currentReport.current){setGuide('No model output was returned, so there is no quality result to inspect.');return}
  if(id==='vera'&&currentReport.current?.adaptive&&!currentReport.current?.adaptive?.auto_complete&&veraLive?.state!=='error'){setGuide('VERA is running a real verification. The mission cannot reach METRI until it finishes.');return}
  if(id==='routa'&&offline){setGuide('The image gateway is offline; no inference is submitted.');setStory({speaker:'ROUTA',title:'Image route unavailable',body:'No image GPU worker is connected. Try another workload or connect a real gateway.',type:'evidence'});setDialogReady(true);return}
  if(id==='routa'){runBackend()}
  // Deliberate *presentation* beat: this is not additional inference time.
  setThinkingId(id);
  setDialogReady(false);
  setStory({speaker:PEOPLE[id].name,title:'Reviewing the mission',body:id==='routa'?'Submitting the real request to the model service.':'Reading the current stage and preparing the handoff.',type:'presentation'});
  setGuide(`${PEOPLE[id].name} is reviewing this part of the mission. The model runs independently of this animation.`);
  const timer=setTimeout(()=>{
    setThinkingId(null);
    setStory(storyFor(id));
    setDialogReady(true);
    setGuide(`Read ${PEOPLE[id].name}'s message. Continue when ready.`);
  },2000);
  timers.current.push(timer);
 }
 function continueStory(){
  if(!dialogReady||thinkingId)return;
  const id=target;
  if(id==='model')setInspected(true);
  if(id==='vera')setGuide('Review the actual guardrail state.');
  if(id==='routa'&&offline){setDialogReady(false);setStory(null);setArrived(false);setTarget('metri');setGuide('Image worker unavailable. Inspect this mission without an inference result.');return}
  setDialogReady(false);setStory(null);
  if(id==='metri'){setDrawer(true);nextFrom(id);return}
  nextFrom(id);
 }
 function playRecorded(){if(!liveEvents.length)return;stopTimers();const seq=replaySeq.current;setReplay(true);setDialogReady(false);setThinkingId(null);setStory(null);setDrawer(false);setEventIndex(-1);setReadable([]);setGuide('REPLAY MODE — recorded backend events, not new inference.');const entries=liveEvents.filter(e=>e.stage);const stages=['analyze','route','execute','verify','deliver'];
  entries.forEach((e,i)=>{const timer=setTimeout(()=>{if(seq!==replaySeq.current)return;setEventIndex(i);setReadable(v=>[...v,e]);const m={analyze:'captain',route:'routa',execute:'model',verify:'vera',deliver:'metri'};setTarget(m[e.stage]||'captain');if(i===entries.length-1){timers.current.push(setTimeout(()=>{if(seq===replaySeq.current){setReplay(false);setGuide('Replay finished. It did not call the model.');}},1000/playSpeed))}},(i+1)*1200/playSpeed);timers.current.push(timer)})
 }
 function stopReplay(){stopTimers();setReplay(false);setTarget('metri');setGuide('Replay stopped. Recorded events remain available.');}
 const WORKER_PROFILES={
  captain:{job:'Mission coordinator',description:'Receives the company workload and confirms the evaluation plan before execution.',proof:'Input count, workload and mission requirements'},
  toki:{job:'Context engineer',description:'Reviews prompt and context use. Applies token reduction only when a connected policy actually executes.',proof:'Before/after token counts when available'},
  routa:{job:'Model routing specialist',description:'Chooses a route or launches paired evaluation on the same dataset.',proof:'Selected models and requested evaluation path'},
  model:{job:'Inference operator',description:'Runs the registered models and captures warm inference times separately from loading.',proof:'Measured latency and execution status'},
  vera:{job:'Quality reviewer',description:'Checks accuracy against available labels and exposes missing or unaudited quality evidence.',proof:'Quality checks and their trust level'},
  metri:{job:'Evidence analyst',description:'Combines comparable measurements into an evidence receipt without guessing energy or cost savings.',proof:'Comparison receipt and provenance'}
 };
 const inspectInformation=(id)=>{
  const modelId=report?.route||(route||'Not selected');
  const latency=numOrNull(report?.metrics?.end_to_end_ms);
  const tokens=getTokenPair(report?.metrics);
  const quality=qualityEvidenceLabel(report);
  if(id==='captain')return {role:'Mission intake',facts:[['Workload',mission.mode],['Inputs',workConfig?.inputMode==='batch'?`${workConfig.examples.length} dataset rows`:`${input?.length||0} characters`],['Request started',requestDone||busy?'Yes':'Not yet']],note:'Input and requirements enter the workflow here. No model inference happens at this station.'};
  if(id==='toki'&&mission.id!=='context')return {role:'Context optimization · not required',facts:[['Workflow step','Skipped — this comparison does not optimize tokens'],['Token reduction','Not applicable']],note:'The current evaluation compares model inference latency and quality on identical inputs; context trimming is not part of this experiment.'};
  if(id==='toki')return {role:'Context optimization',facts:[['Live token counts',tokens?`${tokens[0]} → ${tokens[1]}`:'Not measured'],['Context reduction',tokens?(tokens[1]<tokens[0]?'Recorded':'No reduction recorded'):'Not connected to this run'],['Token policy', 'Experimental; no live claim']],note:'TOKI can explain context efficiency. It only reports an applied reduction when the runtime returns actual before/after token counts.'};
  if(id==='routa')return {role:'Model routing',facts:report?.comparison?[['Baseline',report.comparison.baseline_model],['Candidate',report.comparison.candidate_model],['Evaluation size',String(report.comparison.sample_count)]]:[['Requested workload',mission.mode],['Chosen route',modelId],['Fallback', 'No fallback recorded in this request']],note:report?.comparison?'Both registered models were actually executed on identical inputs. This is a paired local CPU latency measurement.':'The chosen route is read from the actual backend report. A smaller route alone does not establish compute savings.'};
  if(id==='model')return {role:'Model execution',facts:report?.comparison?[['Baseline avg',`${report.comparison.baseline.mean_inference_ms} ms`],['Candidate avg',`${report.comparison.candidate.mean_inference_ms} ms`],['Load+warmup',`${report.metrics.load_warmup_ms} ms`]]:[['Route',modelId],['Request duration',latency!==null?`${latency.toFixed(1)} ms`:'Not measured'],['Output',report?'Available':'Not yet returned']],note:'Guided animation is separate from measured local CPU inference.'};
  if(id==='vera')return {role:'Quality guardrails',facts:report?.comparison?[['Expected labels',report.comparison.all_labels_present?(report?.optimization?.validation==='heldout_repeated_v21'?'IMDb held-out; not externally audited':'Provided; not audited'):'Unavailable'],['Baseline accuracy',report.comparison.all_labels_present?`${(report.comparison.baseline.accuracy_on_supplied_labels*100).toFixed(1)}%`:'Not established'],['Candidate accuracy',report.comparison.all_labels_present?`${(report.comparison.candidate.accuracy_on_supplied_labels*100).toFixed(1)}%`:'Not established']]:[['Evidence status',quality],['Independent task quality','Not certified'],['Confidence',typeof report?.result?.confidence==='number'?`${(report.result.confidence*100).toFixed(1)}%`:'Not available']],note:'Supplied labels enable local accuracy checks, but are not independent certification. Confidence alone is not ground-truth accuracy.'};
  return {role:'Evidence & outcomes',facts:report?.comparison?[['Paired inputs',String(report.comparison.sample_count)],['Mean latency difference',`${((report.comparison.mean_latency_reduction_fraction??0)*100).toFixed(1)}% (candidate vs baseline)`],['Accuracy check',report.comparison.all_labels_present?'Supplied labels, not independently audited':'No complete labels']]:[['Measured request',latency!==null?`${latency.toFixed(1)} ms`:'Not measured'],['Comparable baseline','Not available for this run'],['Compute savings','Not established']],note:report?.comparison?'Measured latency comparison is real, but it is not a measurement of power, FLOPs or monetary cost. Accuracy must be checked with a representative dataset.':'Savings need a comparable baseline, measured resource use and quality checks.'};
 };
 function localUpload(ev){
  const file=ev.target.files?.[0];if(!file)return;
  const extension=file.name.split('.').pop().toLowerCase();
  if(!['csv','jsonl'].includes(extension)){setUploadPreview({error:'Use a CSV or JSONL file.'});return}
  if(file.size>3*1024*1024){setUploadPreview({error:'Preview limit is 3 MB. No upload was sent.'});return}
  file.text().then(text=>{
   const lines=text.split(/\r?\n/).filter(Boolean);
   setUploadPreview({name:file.name,records:Math.max(0,lines.length-(extension==='csv'?1:0)),sample:lines.slice(0,3).join('\n').slice(0,850),extension});
  }).catch(()=>setUploadPreview({error:'Could not read file.'}));
 }
 const missionFlow=mission.id==='context'?ROUTES.context:ROUTES[mission.mode]||STEPS;
 const stepNumber=missionFlow.indexOf(target)+1;
 const receiptTitle=report?'Optimization receipt':'No measured result yet';
 const quality=qualityEvidenceLabel(report);
 return <div className="gameShell" ref={shellRef}>
  <div className="gameHeadline"><div><span className="gameEyebrow">NIMBO / MISSION CONTROL</span><h1>{workspaceOpen?'NIMBO Workspace':'Mission Floor'}</h1><p>{workspaceOpen?'Configure a real workload and connect the evidence.':'Watch the operation, inspect each worker and compare measured outcomes.'}</p></div><div className="gameHeadActions"><button onClick={onThemeChange}>{theme==='night'?<Sun size={15}/>:<Moon size={15}/>} {theme==='night'?'Day Lab':'Night Lab'}</button><span className="gameConnected">● LOCAL BACKEND</span></div></div>
  {workspaceOpen ? <CompanyWorkspace registry={registry} theme={theme} onLaunch={configureCompany}/> : <>
  <div className="nim15TopNav"><button type="button" onClick={backToWorkspace}>← Configure workload</button><div><strong>{workConfig?.operation==='validate_onnx'?'ONNX · independent holdout validation':workConfig?.operation==='optimize_onnx'?'Same-model PyTorch FP32 → ONNX FP32':workConfig?.operation==='optimize'?'Same-model FP32 → INT8':workConfig?.operation==='compare'?'Baseline comparison':mission.title}</strong><span>{(workConfig?.operation==='optimize'||workConfig?.operation==='optimize_onnx'||workConfig?.operation==='validate_onnx')?`${workConfig.sampleCount} records · same-model CPU trial`:workConfig?.operation==='compare'?`${workConfig.examples.length} input(s) · ${workConfig.goal} objective`:workConfig?.goal+' objective'}</span></div><span>CONNECTED MODEL ENVIRONMENT</span></div>
  <div className="gameWorldFrame" ref={worldRef}>
   <div className="gameWorldTop"><span><span className="gameLiveDot"/> NIMBO MISSION FLOOR</span><div className="gameWorldTopActions"><span>{replay?'REPLAY':busy?'RUNNING':started?'ACTIVE':'READY'}</span><button type="button" onClick={()=>setBoardFocus(v=>!v)} title="Zoom camera to the measured results">{boardFocus?'Full Lab':'Focus Board'}</button><button type="button" onClick={toggleFullScreen} title="Toggle fullscreen for the lab only">{fullScreen?<Minimize2 size={16}/>:<Expand size={16}/>} {fullScreen?'Exit Lab View':'Expand Lab'}</button></div></div>
   <div className="gameWorldStage">
     <AdaptiveImagePanelV24 headless report={report} prompt={input} onStatus={(v)=>{setVeraLive(v);setLiveEvents(prev=>{const others=prev.filter(e=>!e.vera_v27);const recorded=(v?.events||[]).map(e=>({stage:e.stage,message:'VERA · '+e.message,at_ms:e.at_ms,vera_v27:true}));return prev.filter(e=>e.vera_v27).length===recorded.length?prev:[...others,...recorded];});}} onUpdate={(next)=>{setReport(next);setLiveEvents((next.trace||[]).map(x=>({stage:x.stage||'deliver',message:x.message||'Adaptive stage',at_ms:x.at_ms??null})));setRoute(next.route||'sd15_adaptive_autonomous_vera_v26');}}/>
     <LabScene veraLive={veraLive} target={target} onSelect={onBot} busy={busy} route={route} comparison={report?.comparison||null} imageComparison={report?.image_comparison||null} theme={theme} offline={offline} liveExecute={liveExecute} started={started} missionSeed={missionSeed} onArrive={onArrive} explainingId={story&&arrived?(story.speaker.toLowerCase()==="mako"?"model":story.speaker.toLowerCase()):null} thinkingId={thinkingId} speech={story&&arrived?story:null} celebrating={Boolean((report?.comparison?.quality_audited && report?.comparison?.mean_latency_reduction_fraction>0 && report?.comparison?.candidate?.accuracy_on_supplied_labels>=report?.comparison?.baseline?.accuracy_on_supplied_labels) || (report?.optimization?.quality_verified && report?.optimization?.baseline_comparable && report?.optimization?.compute_reduction_measured))} metrics={report?.metrics||{}} events={liveEvents} guideStep={started?stepNumber-1:0} quality={qualityEvidenceLabel(report)} flow={missionFlow} boardFocus={boardFocus}/>
    <div className="gameMissionHUD">
     <span className="gameHudKicker">{started?`WORKER TOUR ${stepNumber} / ${missionFlow.length}`:'YOUR MISSION'}</span>
     <h2>{mission.title}</h2><p>{started?guide:mission.hint}</p>
     {((workConfig?.operation==='optimize'||workConfig?.operation==='optimize_onnx'||workConfig?.operation==='validate_onnx')||(workConfig?.inputMode==='batch'&&workConfig?.operation==='compare'))?<div className="nim16BatchMission"><strong>Evaluation dataset</strong><span>{workConfig.fileName||'Uploaded evaluation set'} · {workConfig.sampleCount||workConfig.examples.length} rows</span><small>Baseline: {workConfig.baseline.split('/').pop()}</small><small>Candidate: {workConfig.candidate.split('/').pop()}</small><em>{workConfig?.operation==='validate_onnx'?'Heldout IMDb test records, repeated paired CPU runs; request text stays out of evidence.':workConfig?.operation==='optimize'?'Both FP32 and INT8 come from the same base model; no text is saved in evidence.':'Both models process identical inputs; request text is kept out of the HUD.'}</em></div>:<textarea aria-label="Mission input" value={input} onChange={e=>{setInput(e.target.value);if(started)reset();}} maxLength={2000} disabled={busy||replay}/>}
     {!started?<button className="gamePrimary" onClick={enter} disabled={!input.trim()}><Play size={17}/> Start Mission</button>:!story?<button className="gamePrimary" disabled={replay||Boolean(thinkingId)||(!arrived&&!(busy&&target==='model'))} onClick={()=>{if(busy&&target==='model')setShowLiveProgress(true);else onBot(target)}}>{busy&&target==='model'?'View live model progress':`Speak to ${PEOPLE[target].name}`} <ChevronRight size={17}/></button>:thinkingId?<div className="gamePending"><span className="gamePulse"/> {PEOPLE[thinkingId].name} is preparing a response</div>:<div className="gamePending">{PEOPLE[target].name} is ready · Continue below</div>}
     <div className="gameGuideRow"><span style={{background:PEOPLE[target].color}}/> {started?(arrived?'WORKER READY':'COURIER MOVING'):'READY'}: <b>{PEOPLE[target].name}</b></div><button type="button" className="gameInspectButton" onClick={()=>setInspect(target)}>Inspect {PEOPLE[target].name} <ArrowUpRight size={14}/></button>
     {error&&<p className="gameError">{error}</p>}
    </div>
    {mission.mode==='image'&&veraLive?.active&&<div className="nim27VerificationStrip" role="status"><strong>VERA · LIVE CHECK</strong><span>{veraLive.message}</span></div>}
    <div className="gameTelemetryHUD" aria-live="polite"><div className="gameTelemetryTitle"><Activity size={15}/> <b>BACKEND SIGNALS</b><span>{busy?'LIVE':report?'COMPLETED':error?'ERROR':'IDLE'}</span></div>
     {liveEvents.length?liveEvents.slice(-4).map((e,i)=><div className="gameTelemetryEvent" key={`${e.stage}-${liveEvents.length-4+i}`}><small>{typeof e.at_ms==='number'?e.at_ms.toFixed(1)+'ms':'EVENT'}</small><strong>{String(e.stage).toUpperCase()}</strong><span>{shortEvent(e)}</span></div>):<p>Live measurements appear here when ROUTA launches the model. No fabricated progress.</p>}
     {report&&<div className="gameTelemetryResult"><Check size={14}/> Inference finished · {report.metrics?.end_to_end_ms??'—'} ms</div>}
    </div>
    {story&&started&&!drawer&&<div className="gameStoryPanel" aria-live="polite"><div className="gameStoryTop"><span className="gameStoryAvatar" style={{background:PEOPLE[story.speaker==='MAKO'?'model':story.speaker.toLowerCase()]?.color||'#76dcea'}}/><strong>{story.speaker}</strong><small>{thinkingId?'Thinking · visual explanation':story.type==='measured'?'Actual measurement':'Mission dialogue'}</small></div><h3>{thinkingId?'Let me take a look…':story.title}</h3><p>{story.body}</p>{!thinkingId&&dialogReady&&<button className="gameStoryContinue" onClick={continueStory}>{target==='metri'?'View Mission Receipt':`Continue ${stepNumber} / ${missionFlow.length}`} <ChevronRight size={16}/></button>}{thinkingId&&<div className="gameThoughtDots"><i/><i/><i/></div>}</div>}
    {inspected&&report&&!drawer&&<div className="gameArtifactPreview"><small>MODEL RESULT · {report.route||'route pending'}</small><p>{String(report?.result?.answer||report?.result?.label||'Image received').slice(0,170)}{String(report?.result?.answer||'').length>170?'…':''}</p><button onClick={()=>setDrawer(true)}>Inspect full result <ArrowUpRight size={12}/></button></div>}
    {report?.comparison&&<div className="nim15EvalBadge" role="status"><strong>Paired CPU evaluation</strong><span>Baseline {report.comparison.baseline.mean_inference_ms.toFixed(1)}ms · Candidate {report.comparison.candidate.mean_inference_ms.toFixed(1)}ms</span><small>{qualityEvidenceDetail(report)}</small></div>}
    {started&&<div className="gameFloatingStep"><span>{arrived?'✦':'➜'}</span><div><small>{arrived?'CLICK THE WORKER':'FOLLOW THE MOVING CAPSULE'}</small><strong>{PEOPLE[target].name}</strong></div></div>}
   </div>
   <div className="gameWorldFooter"><div><span className="gameSquare"/> {liveEvents.length} recorded backend events · {report?'Result received':'Not yet completed'} · {offline?'GPU offline':report?.comparison?'Paired baseline measured':'Evidence: baseline not measured'}</div><div className="gameFooterBtns"><select aria-label="Replay speed" value={playSpeed} onChange={e=>setPlaySpeed(Number(e.target.value))}><option value={.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option></select><button disabled={!liveEvents.length||busy} onClick={replay?stopReplay:playRecorded}>{replay?<><Pause size={14}/> Stop Replay</>:<><RotateCcw size={14}/> Replay Story</>}</button><button disabled={!report} onClick={()=>setDrawer(true)}>Open Proof <ArrowUpRight size={14}/></button></div></div>
  </div>

  {/* NIMBO_V20_MODEL_PROGRESS_FIX_2: click-to-view real streamed events */}
  {showLiveProgress&&createPortal(<div className="gameV14Overlay" onClick={()=>setShowLiveProgress(false)}>
    <section className="gameV14Drawer nim20LiveProgress" onClick={e=>e.stopPropagation()} role="dialog" aria-modal="true" aria-label="Live model execution progress">
      <div className="gameV14DrawerHeading"><div><small>MAKO · MODEL EXECUTION</small><h2>Live model progress</h2><p>Measured backend stages, not simulated percentage complete.</p></div><button onClick={()=>setShowLiveProgress(false)}>Close ×</button></div>
      <div className="nim20LiveStatus" role="status"><span className={busy?'nim20LiveDot nim20Running':'nim20LiveDot'}/><strong>{busy?'Model execution in progress':error?'Execution failed':report?'Execution completed':'No finished result yet'}</strong></div>
      <p className="nim20LiveHint">{busy?'On the first run, downloading IMDb data and model weights can take time. Updates only appear when the backend reports a stage.':'You can return to MAKO to inspect the result or continue the mission.'}</p>
      <div className="nim20LiveTimeline" aria-live="polite">
        {liveEvents.length===0?<p>No stage has been returned yet. The request may still be loading the dataset or model.</p>:
          liveEvents.slice(-30).map((e,i)=><div className="nim20LiveEvent" key={`${i}-${e.stage}-${e.at_ms??'pending'}`}><span>{typeof e.at_ms==='number'?`${(e.at_ms/1000).toFixed(1)}s`:'LIVE'}</span><strong>{String(e.stage||'event').toUpperCase()}</strong><p>{String(e.message||'Stage recorded')}</p></div>)}
      </div>
      {error&&<p className="nim20LiveError">{error}</p>}
      <button type="button" className="gameV14Primary" onClick={()=>setShowLiveProgress(false)}>{busy?'Return to lab — keep model running':'Close and continue mission'}</button>
    </section>
  </div>,fullScreen&&worldRef.current?worldRef.current:document.body)}
  {inspect&&createPortal(<div className="gameV14Overlay" onClick={()=>setInspect(null)}><section className="gameV14Drawer" onClick={e=>e.stopPropagation()} role="dialog" aria-modal="true" aria-label={`${PEOPLE[inspect].name} station details`}>
    <div className="gameV14DrawerHeading"><div><small>STATION INTELLIGENCE</small><h2>{PEOPLE[inspect].name}</h2><p>{inspectInformation(inspect).role}</p></div><button onClick={()=>setInspect(null)}>Close ×</button></div>
    <div className="nim18WorkerProfile"><span>WORKER PROFILE</span><h3>{WORKER_PROFILES[inspect].job}</h3><p>{WORKER_PROFILES[inspect].description}</p><small>Evidence owned: {WORKER_PROFILES[inspect].proof}</small></div><p className="gameV14Status">{missionFlow.includes(inspect)?inspect===target?'Current station':report?'Latest measurement available':'Waiting for mission':'Not part of this workflow'} · {busy?'Execution in progress':report?'Result recorded':'No request submitted'}</p>
    <div className="gameV14Facts">{inspectInformation(inspect).facts.map(([k,v])=><div key={k}><small>{k}</small><strong>{v}</strong></div>)}</div>
    <p className="gameV14Note">{inspectInformation(inspect).note}</p>
    <button className="gameV14Primary" onClick={()=>setInspect(null)}>Return to mission</button>
  </section></div>, fullScreen&&worldRef.current?worldRef.current:document.body)}
  {modelOps&&createPortal(<div className="gameV14Overlay" onClick={()=>setModelOps(false)}><section className="gameV14Drawer" onClick={e=>e.stopPropagation()} role="dialog" aria-modal="true" aria-label="Model operations">
     <div className="gameV14DrawerHeading"><div><small>MODEL OPERATIONS</small><h2>Optimization workspace</h2><p>Connected pathways and evaluation readiness.</p></div><button onClick={()=>setModelOps(false)}>Close ×</button></div>
     <div className="gameV14Capability"><strong>Smart routing</strong><span>Connected · local classification policy</span></div>
     <div className="gameV14Capability"><strong>Text generation</strong><span>Connected · local Qwen request</span></div>
     <div className="gameV14Capability"><strong>Context optimization</strong><span>Development experiments only · not wired to this live run</span></div>
     <div className="gameV14Capability"><strong>Quantization</strong><span>Not connected · register comparable precision variants</span></div>
     <div className="gameV14Capability"><strong>Distillation</strong><span>Not connected · provide teacher/student models and quality labels</span></div>
     <div className="gameV14Capability"><strong>Batch evaluation</strong><span>Local file preview available · execution service not connected</span></div>
     <div className="gameV14Import"><label htmlFor="v14dataset">Preview evaluation input <small>CSV or JSONL · max 3 MB</small></label><input id="v14dataset" type="file" accept=".csv,.jsonl" onChange={localUpload}/>
      {uploadPreview&&<div className="gameV14FileInfo">{uploadPreview.error?uploadPreview.error:<><b>{uploadPreview.name} · {uploadPreview.records} lines</b><pre>{uploadPreview.sample}</pre><small>Preview only. No model inference or server upload was performed.</small></>}</div>}
     </div>
     <button className="gameV14Primary" onClick={()=>setModelOps(false)}>Back to Mission Floor</button>
  </section></div>, fullScreen&&worldRef.current?worldRef.current:document.body)}
  {drawer&&createPortal(<div className="gameProofBackdrop" onClick={()=>setDrawer(false)}><aside className="gameProofDrawer" onClick={e=>e.stopPropagation()}><div className="gameProofHeading"><div><small>METRI · EVIDENCE ANALYST</small><h2>{receiptTitle}</h2></div><button onClick={()=>setDrawer(false)}>Close ×</button></div><div className="gameProofKPIs"><div><small>Workload</small><strong>{mission.mode}</strong></div><div><small>Selected route</small><strong className="gameProofReadableRoute">{readableRoute(report?.route)}</strong></div><div><small>{report?.adaptive?'Executed GPU inference (excludes review time)':'Full request including warmup'}</small><strong>{typeof report?.metrics?.end_to_end_ms==='number'?`${report.metrics.end_to_end_ms} ms`:'Not measured'}</strong></div><div><small>Quality evidence</small><strong>{quality}</strong></div><div><small>Input tokens</small><strong>{report?.metrics?.input_tokens??'Not measured'}</strong></div><div><small>{report?.comparison?'Measured latency difference':'Proven compute savings'}</small><strong>{report?.comparison?`${((report.comparison.mean_latency_reduction_fraction??0)*100).toFixed(1)}% candidate vs baseline`:'Not established'}</strong></div></div>{report?.optimization&&<section className="nim20GateSummary" aria-label="Same-model optimization decision">
  <div className="nim20GateHead"><span>{report.optimization.kind==='onnx_runtime_fp32'?'ONNX / EXECUTION OPTIMIZATION':'INT8 / QUANTIZATION'}</span><strong>{report.optimization.gate?.eligible?'PASSED LOCAL GATES':'NOT ACCEPTED'}</strong></div>
  <p className="nim20GateReason">{report.optimization.gate?.reason||'No decision returned'}</p>
  <div className="nim20GateMetrics">
    <div><small>FP32 accuracy</small><b>{((report.comparison?.baseline?.accuracy_on_supplied_labels??0)*100).toFixed(1)}%</b></div>
    <div><small>{report.optimization.kind==='onnx_runtime_fp32'?'ONNX accuracy':'INT8 accuracy'}</small><b>{((report.comparison?.candidate?.accuracy_on_supplied_labels??0)*100).toFixed(1)}%</b></div>
    <div><small>{report.optimization.kind==='onnx_runtime_fp32'?'Serialized files (different formats)':'Serialized weight reduction'}</small><b>{report.optimization.kind==='onnx_runtime_fp32'?`${(report.comparison.baseline_state_dict_bytes/1048576).toFixed(1)} MB state · ${(report.comparison.onnx_artifact_bytes/1048576).toFixed(1)} MB ONNX`:typeof report.comparison?.serialized_weight_reduction_fraction==='number'?`${(report.comparison.serialized_weight_reduction_fraction*100).toFixed(1)}%`:'Not measured'}</b></div>
    <div><small>Warm CPU latency reduction</small><b>{typeof report.comparison?.mean_latency_reduction_fraction==='number'?`${(report.comparison.mean_latency_reduction_fraction*100).toFixed(1)}%`:'Not measured'}</b></div>
    <div><small>FP32 P95</small><b>{report.comparison?.baseline?.p95_ms??'—'} ms</b></div>
    <div><small>{report.optimization.kind==='onnx_runtime_fp32'?'ONNX P95':'INT8 P95'}</small><b>{report.comparison?.candidate?.p95_ms??'—'} ms</b></div>
  </div>
  <p className="nim20GateDisclaimer">{report.optimization.approval==='eligible_for_review'?'Held-out trial: eligible for human review, not deployed.':'Development trial: diagnostic only, not approved for deployment.'} {report.optimization.kind==='onnx_runtime_fp32'?'ONNX includes a computation graph, so its file size is not directly comparable to a PyTorch state_dict.':'Size is serialized weights, not RAM.'} Timings are warm CPU model inference, not full request time.</p>
</section>}{report?.optimization?.validation==='heldout_repeated_v21'&&<section className="nim21GateSummary" aria-label="Independent heldout ONNX decision">
      <div className="nim21Decision"><span>v21 · HELDOUT VALIDATION</span><strong>{report.optimization.gate?.eligible?'ELIGIBLE FOR REVIEW':'REJECTED'}</strong></div>
      <p>800 separate IMDb test records · {report.comparison?.rounds||3} real paired rounds · deployment unchanged</p>
      <div className="nim21MetricGrid">
        <div><small>Warm CPU speedup</small><b>{((report.comparison?.mean_latency_reduction_fraction??0)*100).toFixed(1)}%</b></div>
        <div><small>Tokenizer + inference speedup</small><b>{((report.comparison?.local_pipeline_speedup_fraction??0)*100).toFixed(1)}%</b></div>
        <div><small>Prediction agreement</small><b>{((report.comparison?.prediction_agreement_fraction??0)*100).toFixed(2)}%</b></div>
        <div><small>Repeatability</small><b>{report.optimization.gate?.repeatability_pass?'PASS':'FAIL'}</b></div>
      </div><p>Candidate staged: {report.optimization.export?'Yes · manual review only':'No'}. These are local CPU measurements, not an independent quality audit or a production deployment.</p>
    </section>}{report?.comparison&&<div className="nim15ComparisonProof"><strong>Measured baseline vs candidate</strong><p>{report.comparison.baseline_model}</p><b>{report.comparison.baseline.mean_inference_ms} ms avg</b><p>{report.comparison.candidate_model}</p><b>{report.comparison.candidate.mean_inference_ms} ms avg</b><p>{report.comparison.all_labels_present?`${report?.optimization?.validation==='heldout_repeated_v21'?'Held-out IMDb-label':'Supplied-label'} accuracy: baseline ${(report.comparison.baseline.accuracy_on_supplied_labels*100).toFixed(1)}% / candidate ${(report.comparison.candidate.accuracy_on_supplied_labels*100).toFixed(1)}% · not externally audited`:'Accuracy not evaluated — no complete reference labels'}</p></div>}{report?.image_comparison&&<section className="nim22ImageEvidence" aria-label="Real generated image comparison">
      <h3>Measured image generation</h3><p className="nim22Sub">SD 1.5 · {report.metrics?.device?.toUpperCase()||'GPU'} · identical prompt and seed {report.image_comparison.seed}. Images are not guaranteed to contain the requested objects.</p>
      <div className="nim22ImagePair">
        {report.result?.fast_image_base64&&<figure><img alt="Actual fast LCM generated result" src={'data:image/png;base64,'+report.result.fast_image_base64}/><figcaption><b>Fast · 4-step LCM-LoRA</b><small>{Number(report.image_comparison.fast_inference_ms).toFixed(1)} ms GPU inference</small></figcaption></figure>}
        {report.result?.standard_image_base64&&<figure><img alt="Actual standard SD 1.5 generated result" src={'data:image/png;base64,'+report.result.standard_image_base64}/><figcaption><b>Baseline · 25-step SD 1.5</b><small>{Number(report.image_comparison.standard_inference_ms).toFixed(1)} ms GPU inference</small></figcaption></figure>}
      </div>
      {report?.adaptive&&<div className="nim27MetriReceipt"><strong>VERA result: {report?.adaptive?.decision||'verification pending'}</strong><span>{report?.adaptive?.verdict_reason||'The autonomous quality check is running at VERA.'}</span><small>Independent full-prompt quality: not certified. No financial or energy savings inferred.</small></div>}
      <div className="nim22Times">{report.image_comparison.measured_fast_latency_reduction_fraction!=null?`${(100*report.image_comparison.measured_fast_latency_reduction_fraction).toFixed(1)}% ${report.image_comparison.measured_fast_latency_reduction_fraction>=0?'lower fast-path latency':'fast path was slower'}`:'Single path measured'} · Quality still requires human assessment. No energy or billing estimate.</div>
      <p className="nim22Sub">Does each image contain the specific critical object requested in the prompt? This is not an overall beauty/quality rating.</p>
      <div className="nim22ImageReview">
        {report.result?.fast_image_base64&&<label>Fast result<select value={fastRating} onChange={e=>setFastRating(e.target.value)}><option value="">Choose an assessment</option><option value="yes">Critical object present</option><option value="no">Critical object missing</option></select></label>}
        {report.result?.standard_image_base64&&<label>Standard result<select value={standardRating} onChange={e=>setStandardRating(e.target.value)}><option value="">Choose an assessment</option><option value="yes">Critical object present</option><option value="no">Critical object missing</option></select></label>}
        <button type="button" disabled={(Boolean(report.result?.fast_image_base64)&&!fastRating)||(Boolean(report.result?.standard_image_base64)&&!standardRating)} onClick={async()=>{setImageReviewNote('Saving review…');try{const r=await fetch(API+'/nimbo/image/v22/review',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({run_id:report.run_id,fast_success:fastRating?fastRating==='yes':null,standard_success:standardRating?standardRating==='yes':null})});if(!r.ok)throw Error('Review could not be saved');setImageReviewNote('Human review recorded in the local image evidence catalog.')}catch(e){setImageReviewNote(e.message)}}}>Save human review</button>
      </div>{imageReviewNote&&<p className="nim22ReviewFeedback" role="status">{imageReviewNote}</p>}
      </section>}
      <div className="gameProofResult"><small>MODEL RESULT</small><div>{report?.image_comparison?'Image comparison displayed above':report?.result?.image_base64?<img alt="Generated image" src={'data:image/png;base64,'+report.result.image_base64}/>:prettyOutput(report?.result?.answer||report?.result?.label||'No output')}</div></div><div className="gameProofNote"><ShieldAlert size={19}/><p>{report?.comparison?`Both paths ran on identical inputs with warm CPU inference timings. ${qualityEvidenceDetail(report)} ${report?.optimization?.validation==='heldout_repeated_v21'?'This v21 trial used 800 disjoint IMDb records and three paired rounds.':''} Inference-time reduction does not measure FLOPs, electricity or dollars.`:'The response was produced by the selected model. Comparable baseline cost and independently verified quality are not available for this request. UNVERIFIED indicates missing quality certification, not a failed response.'}</p></div><button className="gamePrimary" onClick={()=>{setDrawer(false);setInspect(null);setModelOps(false);onSeeEvidence();}}>Open Evidence Center <ArrowUpRight size={14}/></button></aside></div>, fullScreen&&worldRef.current?worldRef.current:document.body)}
  </>}
 </div>
}
export {MISSIONS,STEPS,PEOPLE};
