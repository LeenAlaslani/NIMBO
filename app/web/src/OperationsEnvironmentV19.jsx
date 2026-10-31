import React from 'react';
import {Text, RoundedBox} from '@react-three/drei';

const palettes = {
  day: {floor:'#d9e8f2', seam:'#9fbfd1', wall:'#c4dce9', beams:'#eaf8ff', metal:'#7598b0', glow:'#1bbee4', glass:'#b7efff', rack:'#234d70'},
  night:{floor:'#0b203b', seam:'#245375', wall:'#122d4a', beams:'#284563', metal:'#4c7997', glow:'#56dcff', glass:'#0d5f8f', rack:'#07192e'}
};
function Trim({x,z,theme}){const p=palettes[theme];return <>
  <mesh position={[x,.028,z]}><boxGeometry args={[.054,.022,14.2]}/><meshBasicMaterial color={p.glow}/></mesh>
  <pointLight position={[x,.55,z]} color={p.glow} intensity={theme==='day'?.32:.48} distance={2.4}/>
</>}
function ServerRack({x,theme}){const p=palettes[theme];return <group position={[x,0,-2.75]}>
  <RoundedBox args={[.85,3.15,.82]} radius={.12} smoothness={3} position={[0,1.6,0]}><meshStandardMaterial color={p.rack} roughness={.42} metalness={.54}/></RoundedBox>
  {Array.from({length:7},(_,i)=><group key={i} position={[0,.42+i*.39,.44]}>
    <mesh><boxGeometry args={[.65,.26,.025]}/><meshStandardMaterial color="#061a31" roughness={.44}/></mesh>
    <mesh position={[-.20,0,.023]}><boxGeometry args={[.18,.035,.01]}/><meshBasicMaterial color={i%2?'#71daf6':'#65f1c9'}/></mesh>
    <mesh position={[.23,0,.02]}><sphereGeometry args={[.04,10,10]}/><meshBasicMaterial color="#93dcff"/></mesh>
  </group>)}
  <pointLight color={p.glow} intensity={theme==='day'?.5:1.1} distance={2.5} position={[0,2,.7]}/>
</group>}
function GlassPane({x,theme}){const p=palettes[theme];return <group position={[x,2.65,-.3]}>
  <mesh><boxGeometry args={[.08,4.95,8.8]}/><meshPhysicalMaterial color={p.glass} transparent opacity={theme==='day'?.10:.065} roughness={.17} metalness={.12} depthWrite={false}/></mesh>
  {[-3.9,0,3.9].map((z,i)=><mesh key={i} position={[0,0,z]}><boxGeometry args={[.17,5.1,.11]}/><meshStandardMaterial color={p.metal} metalness={.58} roughness={.29}/></mesh>)}
</group>}
function Planter({x,z,theme}){return <group position={[x,0,z]}>
 <RoundedBox args={[.65,.68,.65]} radius={.10} position={[0,.35,0]}><meshStandardMaterial color={theme==='day'?'#f7fcff':'#264d67'} metalness={.2} roughness={.42}/></RoundedBox>
 {[[-.2,0],[.12,.15],[.2,-.15],[-.1,-.2]].map(([a,b],i)=><group key={i} position={[a,.75,b]} rotation={[.16*i,0,.15*i]}>
   <mesh position={[0,.2,0]}><cylinderGeometry args={[.025,.035,.48,6]}/><meshStandardMaterial color="#317b64"/></mesh>
   <mesh position={[0,.40,.05]} rotation={[.25,0,i%2?-.35:.35]} scale={[.65,1.50,.50]}><sphereGeometry args={[.15,12,10]}/><meshStandardMaterial color={i%2?'#40aa91':'#4eb89c'} roughness={.6}/></mesh>
 </group>)}
 </group>}
/** Procedural 3D environment, deliberately separate from AI compute. */
export default function OperationsEnvironmentV19({theme='night'}){
 const p=palettes[theme];return <group>
  <mesh rotation={[-Math.PI/2,0,0]} receiveShadow><planeGeometry args={[19,16]}/><meshStandardMaterial color={p.floor} roughness={.42} metalness={.06}/></mesh>
  {Array.from({length:17},(_,i)=>{const x=-8+i;return <mesh key={'x'+i} position={[x,.008,0]}><boxGeometry args={[.012,.012,15]}/><meshBasicMaterial color={p.seam} transparent opacity={theme==='day'?.53:.34}/></mesh>})}
  {Array.from({length:15},(_,i)=>{const z=-7+i;return <mesh key={'z'+i} position={[0,.008,z]}><boxGeometry args={[18,.012,.012]}/><meshBasicMaterial color={p.seam} transparent opacity={theme==='day'?.53:.34}/></mesh>})}
  <mesh position={[0,2.45,-4.9]} receiveShadow><boxGeometry args={[16.8,5.1,.17]}/><meshStandardMaterial color={p.wall} roughness={.62}/></mesh>
  <mesh position={[0,5.03,-4.83]}><boxGeometry args={[16.8,.18,.32]}/><meshStandardMaterial color={p.beams} metalness={.38}/></mesh>
  <GlassPane x={-8.05} theme={theme}/><GlassPane x={8.05} theme={theme}/>
  {[-7.2,7.2].map(x=><group key={x}>
   <mesh position={[x,5.12,1.3]}><boxGeometry args={[2.5,.15,.52]}/><meshStandardMaterial color={p.beams} emissive={theme==='day'?'#effaff':'#4188bb'} emissiveIntensity={theme==='day'?.18:.48}/></mesh>
   <mesh position={[x,5.00,1.3]}><boxGeometry args={[2.15,.025,.34]}/><meshBasicMaterial color={theme==='day'?'#ffffff':'#b3eaff'}/></mesh>
   <pointLight position={[x,4.62,1.3]} intensity={theme==='day'?1.1:2.0} distance={6} color={theme==='day'?'#fffefd':'#9adfff'} />
  </group>)}
  <Trim x={-6.1} z={0} theme={theme}/><Trim x={6.1} z={0} theme={theme}/>
  <ServerRack x={-7.02} theme={theme}/><ServerRack x={7.02} theme={theme}/>
  <Planter x={-6.8} z={3.9} theme={theme}/><Planter x={6.8} z={3.9} theme={theme}/>
  <mesh position={[0,.02,5.7]}><boxGeometry args={[12,.021,.035]}/><meshBasicMaterial color={p.glow}/></mesh>
  <Text position={[0,5.05,-4.46]} fontSize={.29} color={theme==='day'?'#376a8b':'#9be6fa'} letterSpacing={.085}>NIMBO  ·  OPERATIONS LAB</Text>
 </group>
}
