import React,{useRef} from 'react';
import {useFrame} from '@react-three/fiber';
import {RoundedBox,Text} from '@react-three/drei';
import * as THREE from 'three';

export default function WorkstationV19({person,id,active,busy,explaining,theme='night',route,offline}){
 const beacon=useRef();const color=offline?'#8a9bab':person.color;
 useFrame(({clock})=>{if(beacon.current)beacon.current.opacity=active?.48+.30*Math.sin(clock.elapsedTime*3.1)**2:.11});
 const status=offline?'OFFLINE':active&&busy&&id==='model'?'RUNNING':active&&explaining?'REVIEWING':active?'ACTIVE':'IDLE';
 return <group position={[person.x,0,person.z]}>
  <RoundedBox args={[1.88,.12,1.00]} radius={.07} smoothness={3} position={[0,.54,-.53]} castShadow receiveShadow><meshStandardMaterial color={theme==='day'?'#d9eafb':'#4c7191'} metalness={.38} roughness={.32}/></RoundedBox>
  {[-.78,.78].map(x=><mesh key={x} position={[x,.25,-.52]} castShadow><cylinderGeometry args={[.07,.07,.50,12]}/><meshStandardMaterial color={theme==='day'?'#91b0c6':'#456784'} metalness={.55} roughness={.38}/></mesh>)}
  <RoundedBox args={[1.62,1.10,.15]} radius={.06} smoothness={3} position={[0,1.22,-.85]} castShadow><meshStandardMaterial color={theme==='day'?'#32556c':'#08172d'} metalness={.36} roughness={.27}/></RoundedBox>
  <mesh position={[0,1.22,-.766]}><planeGeometry args={[1.42,.89]}/><meshStandardMaterial color="#071a34" emissive={active?person.color:'#13345b'} emissiveIntensity={active?.28:.13}/></mesh>
  <mesh position={[0,1.70,-.746]}><boxGeometry args={[1.48,.045,.02]}/><meshBasicMaterial color={color}/></mesh>
  <Text position={[0,1.43,-.741]} fontSize={.16} color="#e9f8ff" anchorX="center">{person.name}</Text>
  <Text position={[0,1.16,-.740]} fontSize={.115} color={offline?'#ffbda3':active?'#8ff4f4':'#a1bed7'} anchorX="center">{status}</Text>
  <mesh position={[0,.83,-.74]}><planeGeometry args={[1.08,.035]}/><meshBasicMaterial color={active?person.color:'#355c79'}/></mesh>
  <mesh position={[0,.60,-.12]}><boxGeometry args={[.80,.02,.34]}/><meshStandardMaterial color={theme==='day'?'#b6d1e2':'#27485f'} metalness={.27}/></mesh>
  {Array.from({length:9},(_,i)=><mesh key={i} position={[-.30+(i%5)*.15,.613,-.24+Math.floor(i/5)*.14]}><boxGeometry args={[.105,.011,.077]}/><meshBasicMaterial color={active?color:'#80a5ba'}/></mesh>)}
  <mesh rotation={[-Math.PI/2,0,0]} position={[0,.025,.42]}><ringGeometry args={[.58,.68,40]}/><meshBasicMaterial ref={beacon} color={color} transparent opacity={.12} depthWrite={false}/></mesh>
  {active&&<pointLight color={color} intensity={theme==='day'?.75:1.55} distance={2.7} position={[0,1.3,.3]}/>}
 </group>
}
