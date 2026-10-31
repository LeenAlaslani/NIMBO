// NIMBO_V20_SAME_MODEL_TRIAL
import React,{useEffect,useRef,useState} from 'react';
import {useFrame} from '@react-three/fiber';
import {Text,RoundedBox} from '@react-three/drei';
import * as THREE from 'three';
import {latencyPercent} from './evidencePresentation.mjs';

const friendlyModel=(s)=>{if(/· FP32/.test(String(s)))return 'ORIGINAL FP32';if(/· INT8/.test(String(s)))return 'OPTIMIZED INT8';const id=String(s||'').split('/').pop();if(/distilbert/i.test(id))return 'DistilBERT';if(/bert/i.test(id))return 'BERT';return cut(id,16)};
const cut=(s,n=20)=>{const v=String(s||'');return v.length>n?v.slice(0,n-1)+'…':v};
const valid=n=>typeof n==='number'&&Number.isFinite(n);
function AccentPanel({x,title,value,sub,color='#fff',active=false}){const frame=useRef();
 useFrame(({clock})=>{if(frame.current&&active)frame.current.emissiveIntensity=.12+.12*Math.sin(clock.elapsedTime*2.5)**2});
 return <group position={[x,0,.20]}>
  <RoundedBox args={[3.25,2.10,.025]} radius={.12} smoothness={4} position={[0,.13,0]}><meshStandardMaterial ref={frame} color="#102f55" emissive={color} emissiveIntensity={active?.14:.035} roughness={.44}/></RoundedBox>
  <Text anchorX="left" position={[-1.43,.82,.024]} fontSize={.25} letterSpacing={.045} color="#aacbe5">{title}</Text>
  <Text anchorX="left" position={[-1.43,.04,.03]} fontSize={.78} color={color}>{value}</Text>
  <Text anchorX="left" position={[-1.43,-.58,.04]} fontSize={.23} color="#c0dced">{sub}</Text>
 </group>}
export default function MissionWallV19({route,reported=false,theme='night',metrics={},comparison=null,imageComparison=null,events=[],guideStep=0,quality='UNVERIFIED',busy=false}){
 const frame=useRef(),progress=useRef(),glow=useRef();const [latest,setLatest]=useState(null);
 const numEvents=events.filter(e=>/Evaluated input\s+(\d+)\/(\d+)/i.test(String(e?.message||'')));
 const lastProgress=numEvents.length?/(\d+)\/(\d+)/.exec(String(numEvents.at(-1).message||'')):null;
 // NIMBO_ADAPTIVE_BOARD_V24: no fake fast-vs-standard savings when both paths executed.
 const adaptiveReady=Boolean(imageComparison?.mode==='adaptive'&&valid(imageComparison?.fast_inference_ms));
 const adaptiveFallback=adaptiveReady&&valid(imageComparison?.standard_inference_ms);
 const adaptiveAccepted=adaptiveReady&&imageComparison?.adaptive_phase==='accepted_fast';
 const adaptiveReviewed=adaptiveReady&&imageComparison?.adaptive_phase==='fallback_reviewed';
 const imageReady=Boolean(!adaptiveReady&&imageComparison&&valid(imageComparison?.fast_inference_ms)&&valid(imageComparison?.standard_inference_ms));
 const finished=Boolean((comparison&&valid(comparison?.baseline?.mean_inference_ms)&&valid(comparison?.candidate?.mean_inference_ms))||imageReady);
 const count=lastProgress?Number(lastProgress[1]):finished?(comparison?.sample_count||0):0;
 const total=lastProgress?Number(lastProgress[2]):comparison?.sample_count||0;
 const progressRatio=adaptiveReady?1:imageReady?1:total?THREE.MathUtils.clamp(count/total,0,1):0;
 const baseline=imageReady?imageComparison.standard_inference_ms:comparison?.baseline?.mean_inference_ms,candidate=imageReady?imageComparison.fast_inference_ms:comparison?.candidate?.mean_inference_ms;
 const difference=imageReady?imageComparison.measured_fast_latency_reduction_fraction:comparison?.mean_latency_reduction_fraction;
 const before=metrics?.input_tokens_before??metrics?.original_input_tokens,after=metrics?.input_tokens_after??metrics?.optimized_input_tokens;
 const tokenStatus=valid(before)&&valid(after)?`TOKENS ${before} → ${after}`:'TOKENS NOT MEASURED';
 const labelCheck=finished&&!imageReady&&comparison?.all_labels_present&&valid(comparison?.baseline?.accuracy_on_supplied_labels)&&valid(comparison?.candidate?.accuracy_on_supplied_labels)
  ?`LABEL ACCURACY ${(comparison.baseline.accuracy_on_supplied_labels*100).toFixed(0)} / ${(comparison.candidate.accuracy_on_supplied_labels*100).toFixed(0)}% · UNAUDITED`
  :imageReady?'IMAGE QUALITY · HUMAN REVIEW PENDING':finished?'LABEL ACCURACY NOT MEASURED':'QUALITY CHECK PENDING';
 const latestEvent=events.length?events[events.length-1]:null;
 useEffect(()=>{
  if(!latestEvent)return;
  const label=String(latestEvent?.message||'').trim();setLatest(label);
  const timer=setTimeout(()=>setLatest(null),2600);return()=>clearTimeout(timer);
 },[events.length]);
 useFrame(({clock},delta)=>{
  if(frame.current)frame.current.emissiveIntensity=.07+.022*Math.sin(clock.elapsedTime*1.3);
  if(glow.current)glow.current.opacity=.35+.25*Math.sin(clock.elapsedTime*1.8)**2;
  if(progress.current){const n=THREE.MathUtils.damp(progress.current.scale.x,Math.max(.006,progressRatio),4,Math.min(delta,.05));progress.current.scale.x=n;progress.current.position.x=-5.05+n*5.05;}
 });
 const header=adaptiveReady?(adaptiveAccepted?'FAST ACCEPTED · NO FALLBACK':adaptiveFallback?(adaptiveReviewed?'FALLBACK REVIEWED':'STANDARD FALLBACK EXECUTED'):'FAST IMAGE · REVIEW AND DECIDE'):finished?'MODEL COMPARISON COMPLETE':busy?'RUNNING THE EVALUATION':'READY FOR AN EVALUATION';
 return <group position={[0,2.68,-4.51]} rotation={[0,.017,0]}>
  <RoundedBox args={[11.9,4.38,.24]} radius={.16} smoothness={5} castShadow><meshStandardMaterial color={theme==='day'?'#258cce':'#194a79'} metalness={.53} roughness={.21} emissive="#286bc4" emissiveIntensity={.15}/></RoundedBox>
  <mesh position={[0,0,.133]}><planeGeometry args={[11.53,4.06]}/><meshStandardMaterial ref={frame} color="#061831" emissive="#0b3564" emissiveIntensity={.11} roughness={.39}/></mesh>
  <mesh position={[0,2.14,.15]}><boxGeometry args={[11.55,.055,.055]}/><meshBasicMaterial ref={glow} color="#35c9ff" transparent opacity={.5}/></mesh>
  <Text anchorX="left" position={[-5.34,1.81,.18]} fontSize={.26} color="#a0ecff" letterSpacing={.055}>NIMBO  /  LIVE OPTIMIZATION</Text>
  <Text anchorX="right" position={[5.34,1.81,.18]} fontSize={.25} color={busy?'#7df1de':'#a9d4fa'}>{adaptiveReady?'ADAPTIVE IMAGE READY':finished?'RESULTS READY':busy?'MEASURING':'STANDBY'}</Text>
  <mesh position={[0,1.59,.18]}><planeGeometry args={[10.6,.018]}/><meshBasicMaterial color="#276d9d"/></mesh>
  {finished?<>
    <AccentPanel x={-3.63} title={imageReady?'BASELINE / SD 1.5':`BASELINE / ${friendlyModel(comparison.baseline_model)}`} value={imageReady?`${(baseline/1000).toFixed(1)} s`:`${baseline.toFixed(1)} ms`} sub={imageReady?'GPU inference · 25 steps':'Mean CPU inference'} color="#f5faff"/>
    <AccentPanel x={0} title={imageReady?'FAST / LCM-LoRA':`CANDIDATE / ${friendlyModel(comparison.candidate_model)}`} value={imageReady?`${(candidate/1000).toFixed(1)} s`:`${candidate.toFixed(1)} ms`} sub={imageReady?'GPU inference · 4 steps':'Mean CPU inference'} color="#82e9ff"/>
    <AccentPanel x={3.63} title="LATENCY CHANGE" value={latencyPercent(difference)} sub={typeof difference!=='number'||!Number.isFinite(difference)?'No paired latency measurement':difference>0?'Lower candidate latency':difference<0?'Candidate slower':'Same measured latency'} color={difference>0?'#6af6ce':'#ffb991'} active/>
  </>:<>
    <Text anchorX="left" position={[-5.20,.73,.20]} fontSize={.65} color="#f2fbff">{header}</Text>
    <Text anchorX="left" position={[-5.20,.04,.20]} fontSize={.34} color="#b9e4f7">{busy?`${total?`${count} / ${total} matched inputs measured`:'MODEL REQUEST IN PROGRESS'}`:adaptiveReady?adaptiveFallback?`FAST ${(imageComparison.fast_inference_ms/1000).toFixed(1)}s + STANDARD ${(imageComparison.standard_inference_ms/1000).toFixed(1)}s GPU`:`FAST ${(imageComparison.fast_inference_ms/1000).toFixed(1)}s · STANDARD NOT EXECUTED`:reported?`ROUTE: ${cut(route,39)}`:'SELECT A WORKLOAD · START THE MISSION'}</Text>
    <Text anchorX="left" position={[-5.20,-.62,.20]} fontSize={.27} color="#a8c8df">{adaptiveReady?`GPU INFERENCE ${metrics.gpu_inference_total_ms?.toFixed(0)||'—'} ms · HUMAN WAIT EXCLUDED`:valid(metrics?.end_to_end_ms)?`REQUEST TOTAL ${metrics.end_to_end_ms.toFixed(0)} ms (includes loading)`:'AWAITING BACKEND MEASUREMENTS'}</Text>
  </>}
  <mesh position={[0,-1.2,.20]}><planeGeometry args={[10.8,.018]}/><meshBasicMaterial color="#2a5d8b"/></mesh>
  <Text anchorX="left" position={[-5.28,-1.46,.21]} fontSize={.24} color="#c1dcf4">{adaptiveReady?'SEQUENTIAL IMAGE EXECUTION':imageReady?'SAME PROMPT · SAME SEED':finished?`${comparison.sample_count} MATCHED INPUTS`:`${count} / ${total||'—'} INPUTS`}</Text>
  <Text anchorX="right" position={[5.28,-1.46,.21]} fontSize={.215} color={finished?'#8ef8d4':'#b2d4e9'}>{adaptiveReady?'CRITICAL OBJECT · HUMAN-REVIEWED, NOT AUDITED':finished?labelCheck:tokenStatus}</Text>
  <mesh position={[0,-1.86,.20]}><planeGeometry args={[10.1,.075]}/><meshBasicMaterial color="#173b65"/></mesh>
  <mesh ref={progress} position={[-5.05,-1.86,.215]} scale={[.006,1,1]}><planeGeometry args={[10.1,.075]}/><meshBasicMaterial color="#46dffa" toneMapped={false}/></mesh>
  {latest&&busy&&<group position={[0,1.10,.28]}><RoundedBox args={[5.5,.50,.03]} radius={.09} smoothness={3}><meshBasicMaterial color="#136a9c"/></RoundedBox><Text fontSize={.19} color="#f1fbff" maxWidth={5.08}>{cut(latest,49)}</Text></group>}
 </group>;
}
