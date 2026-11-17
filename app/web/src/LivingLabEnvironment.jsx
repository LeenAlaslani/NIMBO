/** Three adjoining rooms in ONE scene. Decorative geometry never represents compute progress. */
import React from 'react';
import {Text, RoundedBox} from '@react-three/drei';
import {ROOM_ORDER, ROOMS} from './livingLabPlan.mjs';
const roomColors={execution:'#49b9e9',forge:'#a28dea',vision:'#55caaa'};
function Pillar({x,z,color}){return <group position={[x,0,z]}>
 <RoundedBox args={[.24,4.5,.27]} radius={.07} position={[0,2.25,0]}><meshStandardMaterial color="#426886" metalness={.55} roughness={.32}/></RoundedBox>
 <mesh position={[0,4.3,0]}><boxGeometry args={[.5,.12,.44]}/><meshBasicMaterial color={color}/></mesh>
 </group>}
function Room({id,active,theme,onSelect}){
 const r=ROOMS[id],color=roomColors[id],day=theme==='day';return <group position={[r.x,0,0]}>
  <mesh rotation={[-Math.PI/2,0,0]} receiveShadow position={[0,.008,0]}><planeGeometry args={[18,16]}/><meshStandardMaterial color={day?'#c8dcea':'#09243c'} roughness={.52}/></mesh>
  {Array.from({length:9},(_,i)=><group key={i}>
   <mesh position={[-8+i*2,.012,0]}><boxGeometry args={[.02,.017,15]}/><meshBasicMaterial color={color} transparent opacity={.20}/></mesh>
   <mesh position={[0,.012,-7+i*1.8]}><boxGeometry args={[17,.017,.02]}/><meshBasicMaterial color={color} transparent opacity={.20}/></mesh>
  </group>)}
  <mesh position={[0,2.35,-7.2]}><boxGeometry args={[18,4.7,.22]}/><meshStandardMaterial color={day?'#9cbed8':'#173752'} roughness={.6}/></mesh>
  <mesh position={[0,4.8,-7.1]}><boxGeometry args={[17.9,.18,.35]}/><meshStandardMaterial color={color} emissive={color} emissiveIntensity={active?.6:.18}/></mesh>
  <RoundedBox args={[12.4,1.18,.2]} radius={.12} position={[0,3.7,-6.92]}><meshStandardMaterial color={day?'#264764':'#071b33'} metalness={.2}/></RoundedBox>
  <Text anchorX="center" position={[0,3.88,-6.76]} fontSize={.47} color={color} letterSpacing={.04}>{r.title}</Text>
  <Text anchorX="center" position={[0,3.39,-6.74]} fontSize={.22} color="#c7e4ef">{r.subtitle}</Text>
  <Text anchorX="center" rotation={[-Math.PI/2,0,0]} position={[0,.034,-5.4]} fontSize={.32} color={color}>{active?'MISSION ROOM':'AVAILABLE LAB'}</Text>
  {[-8.5,8.5].map(x=>[-6.9,5.7].map(z=><Pillar key={`${x}-${z}`} x={x} z={z} color={color}/>))}
  <group position={[0,0,6.4]} onClick={e=>{e.stopPropagation();onSelect?.(id)}}>
   <RoundedBox args={[5,.15,1.4]} radius={.13} position={[0,.09,0]}><meshStandardMaterial color={active?color:'#33516d'} emissive={color} emissiveIntensity={active?.27:.04}/></RoundedBox>
   <Text position={[0,.21,.1]} rotation={[-Math.PI/2,0,0]} fontSize={.31} color="#eafaff">{active?'ACTIVE ROOM':'EXPLORE ROOM'}</Text>
  </group>
  <pointLight position={[0,4,1]} intensity={active?1.55:.65} color={color} distance={12}/>
 </group>
}
export default function LivingLabEnvironment({room='execution',theme='night',onRoomSelect}){return <group>
 <mesh position={[0,-.16,8]} rotation={[-Math.PI/2,0,0]} receiveShadow><planeGeometry args={[61,20]}/><meshStandardMaterial color={theme==='day'?'#a6c5d8':'#051c32'}/></mesh>
 {ROOM_ORDER.map(id=><Room key={id} id={id} active={room===id} theme={theme} onSelect={onRoomSelect}/>)}
 {[-10,10].map(x=><group key={x} position={[x,0,0]}><mesh position={[0,2.3,-3.6]}><boxGeometry args={[.15,4.6,7.5]}/><meshStandardMaterial color="#315a76" transparent opacity={.25} depthWrite={false}/></mesh></group>)}
 <Text position={[0,.035,8.6]} rotation={[-Math.PI/2,0,0]} fontSize={.38} color="#9feafa">NIMBO  ·  LIVING LABORATORY  ·  PICO MISSION PATH</Text>
 </group>}
