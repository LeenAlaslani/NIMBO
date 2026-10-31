import React,{useRef} from 'react';
import {useFrame} from '@react-three/fiber';
import {Html,RoundedBox} from '@react-three/drei';
import * as THREE from 'three';

/** Forward-kinematic, individually articulated mascot. This is a procedural rig, NOT a GLB/skinned artist asset. */
export default function LivingWorkerV19({person,id,active,busy,explaining,thinking,speech,celebrating,theme='night',onSelect,offline}){
 const rig=useRef(),head=useRef(),body=useRef(),upperL=useRef(),upperR=useRef(),lowerL=useRef(),lowerR=useRef(),footL=useRef(),footR=useRef(),eyeL=useRef(),eyeR=useRef(),mouth=useRef(),badge=useRef();
 const seed=(person.x+7.4)*.81, c=offline?'#98b8c5':person.color;
 useFrame(({clock},delta)=>{
  if(!rig.current)return;const d=Math.min(delta,.05),t=clock.elapsedTime+seed,working=active&&(busy||explaining||thinking),joy=active&&celebrating;
  const idleGesture=id==='toki'?Math.sin(t*.77)*.075:id==='routa'?Math.sin(t*.86)*.04:Math.sin(t*.96)*.058;
  const typing=working&&!thinking&&(id==='model'||id==='toki'||id==='routa');
  rig.current.position.y=THREE.MathUtils.damp(rig.current.position.y,Math.sin(t*1.8)*.032+(joy?Math.max(0,Math.sin(t*5.8))*.21:0),6,d);
  rig.current.rotation.y=THREE.MathUtils.damp(rig.current.rotation.y,typing?.26:thinking?-.20:Math.sin(t*.45)*.09,5,d);
  if(body.current)body.current.scale.y=THREE.MathUtils.damp(body.current.scale.y,1+.022*Math.sin(t*1.85),5,d);
  if(head.current){
   head.current.rotation.z=THREE.MathUtils.damp(head.current.rotation.z,joy?Math.sin(t*6)*.13:thinking?.19:Math.sin(t*.63)*.045,5,d);
   head.current.rotation.x=THREE.MathUtils.damp(head.current.rotation.x,thinking?-.13:working?.085:Math.sin(t*.75)*.025,5,d);
   head.current.rotation.y=THREE.MathUtils.damp(head.current.rotation.y,joy?Math.sin(t*3)*.12:working?Math.sin(t*1.3)*.09:Math.sin(t*.43)*.15,5,d);
  }
  const alphaL=joy?-2.35:thinking?-.58:typing?-.94+Math.sin(t*11)*.22:id==='vera'?-.23+idleGesture:idleGesture;
  const alphaR=joy?-2.45:thinking?-.92:typing?-.95-Math.sin(t*10)*.23:id==='metri'?-.42+idleGesture:-idleGesture;
  if(upperL.current)upperL.current.rotation.x=THREE.MathUtils.damp(upperL.current.rotation.x,alphaL,7,d);
  if(upperR.current)upperR.current.rotation.x=THREE.MathUtils.damp(upperR.current.rotation.x,alphaR,7,d);
  if(lowerL.current)lowerL.current.rotation.x=THREE.MathUtils.damp(lowerL.current.rotation.x,typing?-.55+Math.sin(t*13)*.13:joy?-.26:thinking?-.36:.09,8,d);
  if(lowerR.current)lowerR.current.rotation.x=THREE.MathUtils.damp(lowerR.current.rotation.x,typing?-.51+Math.sin(t*12+1)*.12:joy?-.25:thinking?-.55:.09,8,d);
  if(footL.current)footL.current.rotation.x=THREE.MathUtils.damp(footL.current.rotation.x,joy?-.38:Math.sin(t*.9)*.045,5,d);
  if(footR.current)footR.current.rotation.x=THREE.MathUtils.damp(footR.current.rotation.x,joy?.38:-Math.sin(t*.9)*.045,5,d);
  const blink=Math.sin(t*.94)>.987?.065:joy?.64:thinking?.82:1;
  if(eyeL.current){eyeL.current.scale.y=THREE.MathUtils.damp(eyeL.current.scale.y,blink,17,d);eyeL.current.position.x=-.24+(working?.012*Math.sin(t*2.8):0)}
  if(eyeR.current){eyeR.current.scale.y=THREE.MathUtils.damp(eyeR.current.scale.y,blink,17,d);eyeR.current.position.x=.24+(working?.012*Math.sin(t*2.8):0)}
  if(mouth.current)mouth.current.visible=Boolean(joy||explaining);
  if(badge.current)badge.current.emissiveIntensity=THREE.MathUtils.damp(badge.current.emissiveIntensity,working?.8:joy?1.6:.25,6,d);
 });
 const cloud=theme==='day'?'#fffdf8':'#ebf5ff',shoe=theme==='day'?'#bdd0e0':'#c0d7e9';
 const bits=id==='captain'?<mesh position={[0,.64,0]}><cylinderGeometry args={[.24,.27,.15,16]}/><meshStandardMaterial color="#efbd71" metalness={.26}/></mesh>:id==='toki'?<group position={[.55,.40,0]} rotation={[0,0,.36]}><mesh><boxGeometry args={[.14,.34,.12]}/><meshStandardMaterial color="#c4a1fc"/></mesh></group>:id==='routa'?<mesh position={[0,.68,0]}><octahedronGeometry args={[.17,0]}/><meshStandardMaterial color="#74c8ff" emissive="#74c8ff" emissiveIntensity={.24}/></mesh>:id==='model'?<mesh position={[0,.63,0]}><boxGeometry args={[.50,.10,.34]}/><meshStandardMaterial color="#66e4d6" metalness={.32}/></mesh>:id==='vera'?<mesh position={[.56,-.05,.05]} rotation={[0,.3,0]}><torusGeometry args={[.16,.045,9,22]}/><meshStandardMaterial color="#f5c56b" metalness={.3}/></mesh>:<group position={[.55,.02,0]}>{[0,.15,.30].map((h,i)=><mesh key={i} position={[0,h,0]}><boxGeometry args={[.09,.11+i*.035,.08]}/><meshStandardMaterial color="#73e8ad"/></mesh>)}</group>;
 return <group position={[person.x,0,person.z+.73]}>
  <group ref={rig} scale={id==='captain'?.92:id==='model'?.84:.79} onClick={e=>{e.stopPropagation();onSelect(id)}} onPointerOver={e=>{e.stopPropagation();document.body.style.cursor='pointer'}} onPointerOut={()=>{document.body.style.cursor='auto'}}>
   <group ref={body} position={[0,.68,0]}><mesh castShadow><sphereGeometry args={[.46,24,16]}/><meshStandardMaterial color={cloud} roughness={.42}/></mesh>
     <mesh position={[0,0,.42]}><sphereGeometry args={[.16,14,12]}/><meshStandardMaterial ref={badge} color={c} emissive={c} emissiveIntensity={.20}/></mesh></group>
   <group ref={head} position={[0,1.23,0]}>
     <mesh castShadow><sphereGeometry args={[.63,26,20]}/><meshStandardMaterial color={cloud} roughness={.34}/></mesh>
     {[[-.43,.28,-.08,.35],[0,.50,-.06,.44],[.40,.30,-.07,.32]].map(([x,y,z,r],i)=><mesh key={i} position={[x,y,z]} castShadow><sphereGeometry args={[r,18,12]}/><meshStandardMaterial color={cloud} roughness={.36}/></mesh>)}
     <RoundedBox args={[1.12,.68,.12]} radius={.22} smoothness={4} position={[0,.04,.51]}><meshStandardMaterial color="#091b35" metalness={.32} roughness={.22}/></RoundedBox>
     <mesh ref={eyeL} position={[-.24,.08,.58]}><sphereGeometry args={[.13,18,12]}/><meshBasicMaterial color={c}/></mesh>
     <mesh ref={eyeR} position={[.24,.08,.58]}><sphereGeometry args={[.13,18,12]}/><meshBasicMaterial color={c}/></mesh>
     <mesh ref={mouth} position={[0,-.15,.583]} visible={false}><torusGeometry args={[.09,.016,8,18,Math.PI]}/><meshBasicMaterial color={c}/></mesh>
     {bits}
   </group>
   {[-1,1].map(side=><group key={side} ref={side<0?upperL:upperR} position={[side*.48,.87,.01]}>
     <mesh position={[side*.02,-.18,0]} castShadow><capsuleGeometry args={[.155,.22,5,12]}/><meshStandardMaterial color={cloud} roughness={.45}/></mesh>
     <group ref={side<0?lowerL:lowerR} position={[side*.015,-.33,.015]}><mesh position={[0,-.17,.035]} castShadow><capsuleGeometry args={[.14,.10,5,10]}/><meshStandardMaterial color={cloud} roughness={.38}/></mesh></group>
   </group>)}
   {[-1,1].map(side=><group key={'leg'+side} ref={side<0?footL:footR} position={[side*.22,.27,.04]}><mesh position={[0,-.09,.07]} castShadow><capsuleGeometry args={[.165,.18,5,10]}/><meshStandardMaterial color={shoe} roughness={.42}/></mesh></group>)}
  </group>
  <Html position={[0,2.25,.14]} center style={{pointerEvents:'none'}} zIndexRange={[9,1]}><div className={'nim19Name '+(active?'isActive':'')} style={{'--worker-accent':c}}><strong>{person.name}</strong><span>{person.role}</span></div></Html>
  {active&&speech&&<Html position={[0,3.02,.30]} center style={{pointerEvents:'none'}} zIndexRange={[12,2]}><div className={'nim19Speech '+(thinking?'isThinking':'')} style={{'--worker-accent':c}}><strong>{person.name}</strong><span>{thinking?(id==='vera'&&speech?.title?speech.title:'Checking…'):speech.title}</span><i/></div></Html>}
 </group>;
}
