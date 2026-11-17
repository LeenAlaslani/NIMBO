/** Mission Floor wall: true M7/M11 backend event checkpoints, never simulated progress. */
import React from 'react';
import {RoundedBox,Text} from '@react-three/drei';
import {recordedCheckpoints,jobHeadline} from './nimboAutoCheckpoints.mjs';

const cut=(value,max=56)=>{
 const text=String(value||'');return text.length>max?text.slice(0,max-1)+'…':text;
};
const statusColor={RECORDED:'#73edc6',ACTIVE:'#93d6ff',REJECTED:'#ffad97',PENDING:'#99b4c9'};
export default function NimboAutoWall({mission,theme='night'}){
 const events=mission?.events||[];
 const checks=recordedCheckpoints(events);
 const latest=events.length?events.at(-1):null;
 const status=jobHeadline(mission?.job,mission?.mode);
 return <group position={[0,2.68,-4.51]} rotation={[0,.017,0]}>
  <RoundedBox args={[11.9,4.38,.24]} radius={.16} smoothness={5} castShadow><meshStandardMaterial color={theme==='day'?'#258cce':'#194a79'} roughness={.22} metalness={.5} emissive="#144a77" emissiveIntensity={.12}/></RoundedBox>
  <mesh position={[0,0,.133]}><planeGeometry args={[11.53,4.06]}/><meshStandardMaterial color="#071e34" roughness={.45}/></mesh>
  <Text position={[-5.21,1.78,.18]} anchorX="left" fontSize={.27} color="#9ceaff">NIMBO  /  REAL MODEL OPTIMIZATION</Text>
  <Text position={[5.16,1.78,.18]} anchorX="right" fontSize={.22} color="#a9d8f4">{mission?.mode==='historical'?'REPLAY · NOT LIVE':mission?.job?.state==='RUNNING'?'LIVE BACKEND':cut(mission?.job?.state||'IDLE',22)}</Text>
  <mesh position={[0,1.55,.18]}><planeGeometry args={[10.6,.018]}/><meshBasicMaterial color="#2b658c"/></mesh>
  <Text position={[-5.12,1.16,.19]} anchorX="left" fontSize={.43} color="#effaff">{cut(status,33)}</Text>
  <Text position={[5.16,1.16,.19]} anchorX="right" fontSize={.23} color="#b6d8eb">{events.length} REAL EVENTS</Text>
  {checks.map((step,i)=>{
   const x=-3.91+i*2.62;
   return <group key={step.key} position={[x,.13,.18]}>
    <RoundedBox args={[2.41,1.30,.025]} radius={.1} smoothness={3} position={[0,0,0]}><meshStandardMaterial color="#173951" roughness={.65}/></RoundedBox>
    <Text anchorX="center" position={[0,.31,.028]} fontSize={.22} color="#c5e0ef" maxWidth={2.15}>{step.title}</Text>
    <Text anchorX="center" position={[0,-.19,.028]} fontSize={.33} color={statusColor[step.state]}>{step.state}</Text>
   </group>;
  })}
  <mesh position={[0,-.80,.19]}><planeGeometry args={[10.6,.016]}/><meshBasicMaterial color="#2b658c"/></mesh>
  <Text anchorX="left" position={[-5.17,-1.12,.20]} fontSize={.23} color="#b2d6e9">{latest?cut(latest.message,95):'No backend event received yet'}</Text>
  <Text anchorX="left" position={[-5.17,-1.67,.20]} fontSize={.19} color="#8daec5">RECORDED GATES ONLY  ·  NO SYNTHETIC COMPUTE  ·  REVIEW ONLY</Text>
 </group>;
}
