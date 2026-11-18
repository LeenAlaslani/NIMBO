/**
 * NIMBO M13.3 — Living Garden Motion & Honest Replay Evidence
 * A procedural 3D lab, preserving the existing model-job contract.
 * Movement and idle expressions are visual presentation only. They do not
 * assert that inference, gate verification, or LLM reasoning has occurred.
 */
import React, {createContext, useContext, useEffect, useMemo, useRef, useState} from 'react';
import {Canvas, useFrame, useThree} from '@react-three/fiber';
import {Html, OrbitControls} from '@react-three/drei';
import * as THREE from 'three';
import {gardenEvidence, gardenHandoffs} from './nimboGardenEvidence.mjs';
import {walkStep, dampYaw, stationFacingYaw} from './nimboGardenMotion.mjs';
import {replayEventsUntil, isAtRecordedEnd} from './nimboGardenReplay.mjs';
import './nimbo-garden-world.css';

const GardenLabelPortal = createContext(null);

function GardenLabel(props) {
  const portal = useContext(GardenLabelPortal);
  return <Html {...props} portal={portal}/>;
}

const STATIONS = Object.freeze({
  toki:    {p:[-6.9, -3.25], color:'#c4a6ff', label:'TOKI',    sub:'Context studio',   symbol:'◈'},
  routa:   {p:[-3.85,-3.55], color:'#74cfff', label:'ROUTA',   sub:'Routing desk',    symbol:'◇'},
  forge:   {p:[-.88, -3.55], color:'#ffb77c', label:'FORGE',   sub:'Optimization forge',symbol:'⚙'},
  model:   {p:[2.55, -3.55], color:'#68e8ff', label:'MAKO',    sub:'Execution studio',symbol:'ϟ'},
  captain: {p:[-5.0,  2.2 ], color:'#8fd4ff', label:'CAPTAIN', sub:'Mission control',symbol:'⚑'},
  bundle:  {p:[-1.55, 3.5 ], color:'#ffd083', label:'BUNDLE',  sub:'Package dock',   symbol:'▣'},
  metri:   {p:[2.2,   3.5 ], color:'#95d9ff', label:'METRI',   sub:'Evidence garden',symbol:'▥'},
  vera:    {p:[5.45,  2.2 ], color:'#98efc0', label:'VERA',    sub:'Validation grove',symbol:'✓'}
});

const MINT = '#62ead7';

function Pillar({position, height=4.65}) {
  return <group position={position}>
    <mesh position={[0,height/2,0]} castShadow>
      <cylinderGeometry args={[.13,.2,height,10]}/>
      <meshStandardMaterial color="#a0ccd1" metalness={.72} roughness={.18}/>
    </mesh>
    <mesh position={[0,height,0]}>
      <sphereGeometry args={[.19,12,10]}/>
      <meshStandardMaterial color="#71ffdf" emissive="#35dfc8" emissiveIntensity={.85}/>
    </mesh>
  </group>;
}

function CurvedRail({points,color='#5acaca',radius=.055}) {
  const curve=useMemo(()=>new THREE.CatmullRomCurve3(points.map(p=>new THREE.Vector3(...p))),[points]);
  return <mesh>
    <tubeGeometry args={[curve,56,radius,7,false]}/>
    <meshStandardMaterial color={color} emissive={color} emissiveIntensity={.2} metalness={.4} roughness={.26}/>
  </mesh>;
}

function GlowCircle({radius=1.0,color=MINT,y=.13}) {
  return <mesh rotation={[-Math.PI/2,0,0]} position={[0,y,0]}>
    <torusGeometry args={[radius,.038,7,70]}/>
    <meshBasicMaterial color={color}/>
  </mesh>;
}

function Planter({x,z,s=.8,color='#367f78'}) {
  return <group position={[x,0,z]} scale={s}>
    <mesh position={[0,.2,0]} castShadow>
      <cylinderGeometry args={[.54,.48,.42,14]}/>
      <meshStandardMaterial color="#c8e4d9" metalness={.15} roughness={.45}/>
    </mesh>
    <mesh position={[0,.45,0]}>
      <cylinderGeometry args={[.48,.48,.06,14]}/>
      <meshStandardMaterial color="#265e51"/>
    </mesh>
    {[[-.2,.7,.08],[.17,.63,-.1],[.02,.84,.13]].map(([a,b,c],i)=><mesh
      key={i} position={[a,b,c]} rotation={[0,i*1.1,i%2?.3:-.2]}>
      <icosahedronGeometry args={[.26+i*.05,1]}/>
      <meshStandardMaterial color={i===1?'#69bd89':color} roughness={.85}/>
    </mesh>)}
    <mesh position={[.12,.88,.18]}>
      <sphereGeometry args={[.075,12,8]}/>
      <meshStandardMaterial color="#b9b0ff" emissive="#946ae8" emissiveIntensity={.65}/>
    </mesh>
  </group>;
}

function Tree({x,z,s=1}) {
  return <group position={[x,0,z]} scale={s}>
    <mesh position={[0,1.08,0]} castShadow>
      <cylinderGeometry args={[.13,.27,2.15,9]}/>
      <meshStandardMaterial color="#95775a" roughness={.92}/>
    </mesh>
    <mesh position={[0,2.44,0]} castShadow>
      <icosahedronGeometry args={[.93,1]}/>
      <meshStandardMaterial color="#3c9e7f" roughness={.8}/>
    </mesh>
    <mesh position={[-.52,1.92,.12]} castShadow>
      <icosahedronGeometry args={[.63,1]}/>
      <meshStandardMaterial color="#6fbc85" roughness={.86}/>
    </mesh>
    <mesh position={[.48,2.18,.1]} castShadow>
      <icosahedronGeometry args={[.6,1]}/>
      <meshStandardMaterial color="#4eaf9b" roughness={.86}/>
    </mesh>
    {[[-.56,2.07,.5],[.59,2.2,.45],[0,2.85,.76]].map((p,i)=><mesh key={i} position={p}>
      <sphereGeometry args={[.065,10,8]}/>
      <meshStandardMaterial color="#c0b5ff" emissive="#a792ff" emissiveIntensity={.7}/>
    </mesh>)}
  </group>;
}

function WaterFeature() {
  return <group position={[0,0,-7.15]}>
    <mesh position={[0,.38,0]}>
      <cylinderGeometry args={[2.7,2.7,.7,40]}/>
      <meshStandardMaterial color="#6fadb0" metalness={.65} roughness={.22}/>
    </mesh>
    <mesh position={[0,.77,0]}>
      <cylinderGeometry args={[2.55,2.55,.045,40]}/>
      <meshPhysicalMaterial color="#3eacb0" metalness={.65} roughness={.05}
        transparent opacity={.68}/>
    </mesh>
    {[0,1,2,3,4,5].map(i=><mesh key={i} position={[Math.cos(i*1.047)*1.8,1.35,Math.sin(i*1.047)*1.8]}>
      <sphereGeometry args={[.075,10,7]}/>
      <meshStandardMaterial color="#6fffea" emissive="#4deedc" emissiveIntensity={.75}/>
    </mesh>)}
    <mesh position={[0,1.2,0]}>
      <cylinderGeometry args={[.16,.26,.9,14]}/>
      <meshStandardMaterial color="#d8e7dc" metalness={.6}/>
    </mesh>
    <mesh position={[0,1.75,0]}>
      <sphereGeometry args={[.44,24,16]}/>
      <meshPhysicalMaterial color="#74e4db" roughness={.1} transmission={.38}
        thickness={.2} emissive="#20a7c2" emissiveIntensity={.4}/>
    </mesh>
  </group>;
}

function Atrium({day}) {
  const ribs=useMemo(()=>[-7.6,-4.6,-1.6,1.6,4.6,7.6].map(z=>
    Array.from({length:29},(_,i)=>{
      const a=Math.PI*i/28;
      return [10.3*Math.cos(a),.6+8.6*Math.sin(a),z];
    })
  ),[]);
  return <group>
    <mesh position={[0,-.42,0]} receiveShadow>
      <boxGeometry args={[23,.58,18]}/>
      <meshStandardMaterial color={day?'#8bb6a7':'#113644'} roughness={.4}/>
    </mesh>
    <mesh position={[0,-.085,0]} receiveShadow>
      <boxGeometry args={[22.1,.085,17.1]}/>
      <meshStandardMaterial color={day?'#b7dacf':'#224e5d'} roughness={.35} metalness={.12}/>
    </mesh>
    <mesh position={[0,-.03,0]} rotation={[-Math.PI/2,0,0]}>
      <ringGeometry args={[5.0,5.14,80]}/>
      <meshBasicMaterial color="#3bb8bd" transparent opacity={.85} side={THREE.DoubleSide}/>
    </mesh>
    {ribs.map((pts,i)=><CurvedRail key={i} points={pts} color={day?'#7ab4bd':'#64b0b7'} radius={.04}/>)}
    {[-8.8,8.8].map(x=>[-6.9,6.9].map(z=><Pillar key={`${x}:${z}`} position={[x,0,z]}/>))}
    <mesh position={[0,7.4,-7.2]} rotation={[Math.PI/2,0,0]}>
      <torusGeometry args={[2.3,.08,8,64]}/>
      <meshStandardMaterial color="#77f9e6" emissive="#39d7c4" emissiveIntensity={.9}/>
    </mesh>
    <mesh position={[0,.05,0]} receiveShadow>
      <cylinderGeometry args={[2.25,2.42,.38,48]}/>
      <meshStandardMaterial color={day?'#d3e9df':'#356275'} roughness={.22} metalness={.58}/>
    </mesh>
    <group position={[0,.25,0]}><GlowCircle radius={2.12} color="#59eddf"/></group>
    <WaterFeature/>
    <Tree x={-8.9} z={-6} s={.82}/><Tree x={8.9} z={-6} s={.82}/>
    <Tree x={-9} z={5.8} s={.95}/><Tree x={9} z={5.8} s={.95}/>
    {[
      [-7.9,.55],[-9.0,2.8],[-7.9,4.9],[-7.8,-4.7],
      [7.9,.6],[9.0,2.8],[7.9,4.9],[8,-4.7]
    ].map(([x,z],i)=><Planter key={i} x={x} z={z} s={i%3===0?.75:.95}/>)}
    <GardenLabel position={[0,7.95,-7.1]} center distanceFactor={13} style={{pointerEvents:'none'}}>
      <div className="gardenArchitectureLogo">✦ NIMBO <small>LIVING OPERATIONS GARDEN</small></div>
    </GardenLabel>
  </group>;
}

function ZonePod({x,z,w=4.2,d=2.8,accent='#65f4dd',title,subtitle}) {
  return <group position={[x,0,z]}>
    <mesh position={[0,-.08,0]} receiveShadow>
      <boxGeometry args={[w,.16,d]}/>
      <meshStandardMaterial color="#2b5363" metalness={.28} roughness={.35}/>
    </mesh>
    <mesh position={[0,.015,0]} receiveShadow>
      <boxGeometry args={[w-.12,.038,d-.12]}/>
      <meshStandardMaterial color="#456878" roughness={.3} metalness={.45}/>
    </mesh>
    {[-1,1].map(i=><mesh key={i} position={[i*(w/2-.15),.05,0]}>
      <boxGeometry args={[.045,.04,d-.18]}/>
      <meshBasicMaterial color={accent}/>
    </mesh>)}
    <GardenLabel position={[0,2.0,-d/2]} center distanceFactor={12} style={{pointerEvents:'none'}}>
      <div className="gardenZoneTitle"><strong>{title}</strong><span>{subtitle}</span></div>
    </GardenLabel>
  </group>;
}

function WorkBench({accent='#65e8e1',type='code'}) {
  return <group>
    <mesh position={[0,.48,-.55]} castShadow>
      <boxGeometry args={[1.35,.14,.65]}/>
      <meshStandardMaterial color="#d6ebeb" metalness={.45} roughness={.27}/>
    </mesh>
    {[-.47,.47].map(x=><mesh key={x} position={[x,.21,-.55]}>
      <boxGeometry args={[.09,.38,.11]}/>
      <meshStandardMaterial color="#5b929c" metalness={.65}/>
    </mesh>)}
    <mesh position={[0,.96,-.69]} rotation={[-.16,0,0]}>
      <boxGeometry args={[1.0,.61,.055]}/>
      <meshStandardMaterial color="#092f42" emissive="#15556a" emissiveIntensity={.35}/>
    </mesh>
    <mesh position={[0,.96,-.65]} rotation={[-.16,0,0]}>
      <planeGeometry args={[.86,.46]}/>
      <meshBasicMaterial color={accent} transparent opacity={.22}/>
    </mesh>
    {type==='forge'&&<mesh position={[.55,.77,.07]} rotation={[.4,.55,.2]}>
      <octahedronGeometry args={[.32,0]}/>
      <meshStandardMaterial color="#ffbf87" emissive="#fd8d3b" emissiveIntensity={.7}/>
    </mesh>}
    {type==='shield'&&<mesh position={[.52,.8,0]}>
      <icosahedronGeometry args={[.27,0]}/>
      <meshStandardMaterial color="#b4ffc1" emissive="#2aa078" emissiveIntensity={.65}/>
    </mesh>}
    {type==='package'&&<mesh position={[.52,.67,0]}>
      <boxGeometry args={[.38,.38,.38]}/>
      <meshStandardMaterial color="#ffd78a" emissive="#b4893c" emissiveIntensity={.4}/>
    </mesh>}
    {type==='data'&&[0,1,2].map(i=><mesh key={i} position={[.35+i*.17,.7+i*.1,.07]}>
      <boxGeometry args={[.09,.23+i*.1,.09]}/>
      <meshStandardMaterial color="#90dbff" emissive="#348fc1" emissiveIntensity={.4}/>
    </mesh>)}
  </group>;
}

function Character({id,color,active=false,warning=false,faceYaw=0,locomotionRef=null,reacting=false}) {
  // The model is rigged procedurally: head, arms and feet have independent pivots.
  const actor=useRef(),head=useRef(),leftArm=useRef(),rightArm=useRef();
  const leftLeg=useRef(),rightLeg=useRef();
  const phase=useRef(id.length*.61);
  useFrame(({clock},dt)=>{
    const seconds=Math.min(.08,Math.max(0,dt));
    const walking=Boolean(locomotionRef?.current?.moving);
    const stride=walking?1:0;
    phase.current+=seconds*(walking?11:1.6);
    const wave=Math.sin(phase.current),opposite=Math.sin(phase.current+Math.PI);
    if(actor.current){
      actor.current.rotation.y=dampYaw(actor.current.rotation.y,faceYaw,seconds,5);
      actor.current.rotation.z=walking?wave*.045:Math.sin(clock.elapsedTime*.85+id.length)*.024;
      actor.current.position.y=.25+(walking?Math.abs(wave)*.06:Math.sin(clock.elapsedTime*1.7+id.length)*.027);
    }
    if(head.current){
      head.current.rotation.y=Math.sin(clock.elapsedTime*.62+id.length)*.1;
      head.current.rotation.z=(warning ? .1 : active ? -.07 : 0)+Math.sin(clock.elapsedTime*.85)*.035;
    }
    if(leftLeg.current)leftLeg.current.rotation.x=stride*wave*.65;
    if(rightLeg.current)rightLeg.current.rotation.x=stride*opposite*.65;
    if(leftArm.current){
      leftArm.current.rotation.x=walking?opposite*.52:active?Math.sin(phase.current*2)*.28:Math.sin(phase.current)*.07;
      leftArm.current.rotation.z=-.13+(reacting?.24:0);
    }
    if(rightArm.current){
      rightArm.current.rotation.x=walking?wave*.52:active?Math.sin(phase.current*2+1)*.25:Math.sin(phase.current+1)*.075;
      rightArm.current.rotation.z=.13+(reacting?-.17:0);
    }
  });
  const eye=warning?'#ffb67e':color;
  const pleased=id==='bundle'||id==='toki'||id==='pico';
  return <group ref={actor}>
    <mesh position={[0,.41,0]} castShadow>
      <capsuleGeometry args={[.34,.23,6,16]}/>
      <meshStandardMaterial color="#e9f7fc" metalness={.2} roughness={.25}/>
    </mesh>
    {[-.24,.24].map((x,i)=><group key={i} ref={i===0?leftLeg:rightLeg} position={[x,.25,.12]}>
      <mesh position={[0,-.14,0]} castShadow>
        <capsuleGeometry args={[.12,.16,4,10]}/>
        <meshStandardMaterial color="#c7dfe7"/>
      </mesh>
    </group>)}
    <group ref={head} position={[0,1.05,0]}>
      <mesh castShadow>
        <sphereGeometry args={[.49,22,16]}/>
        <meshStandardMaterial color="#f3fbff" metalness={.12} roughness={.19}/>
      </mesh>
      <mesh position={[0,0,.365]} scale={[.41,.3,.14]}>
        <sphereGeometry args={[1,20,12]}/>
        <meshStandardMaterial color="#071d2f" metalness={.36} roughness={.12}/>
      </mesh>
      {[-.16,.16].map((x,i)=><mesh key={i} position={[x,.035,.49]}
        scale={[1,pleased ? .65 : warning ? .48 : 1,1]}>
        <sphereGeometry args={[.075,10,8]}/>
        <meshBasicMaterial color={eye}/>
      </mesh>)}
      <mesh position={[0,-.125,.49]} scale={[.115,pleased?.035:.027,.015]}>
        <sphereGeometry args={[1,12,8]}/><meshBasicMaterial color={eye}/>
      </mesh>
      {[-.47,.47].map(x=><mesh key={x} position={[x,0,0]}>
        <sphereGeometry args={[.12,12,10]}/>
        <meshStandardMaterial color={color} emissive={color} emissiveIntensity={active?.45:.15}/>
      </mesh>)}
      {id==='captain'&&<group position={[0,.49,0]}>
        <mesh><cylinderGeometry args={[.34,.38,.14,20]}/>
          <meshStandardMaterial color="#215481"/></mesh>
        <mesh position={[0,.12,0]}>
          <boxGeometry args={[.49,.2,.48]}/><meshStandardMaterial color="#26496d"/>
        </mesh>
      </group>}
      {id==='forge'&&<mesh position={[0,.44,0]}>
        <boxGeometry args={[.59,.14,.54]}/>
        <meshStandardMaterial color="#b46c36" metalness={.55}/>
      </mesh>}
      {id==='vera'&&<group position={[.23,.47,-.08]} rotation={[0,0,-.25]}>
        <mesh><coneGeometry args={[.18,.38,6]}/>
          <meshStandardMaterial color="#5fc68a"/></mesh>
      </group>}
      {id==='metri'&&[-.16,.16].map(x=><mesh key={x} position={[x,.035,.5]} rotation={[Math.PI/2,0,0]}>
        <torusGeometry args={[.13,.02,6,24]}/><meshBasicMaterial color="#d9f8ff"/>
      </mesh>)}
      {id==='toki'&&<mesh position={[.25,.43,-.06]} rotation={[0,0,-.3]}>
        <coneGeometry args={[.13,.33,6]}/><meshStandardMaterial color="#aa8ef6"/>
      </mesh>}
      {id==='model'&&<mesh position={[0,.5,0]}>
        <octahedronGeometry args={[.2]}/><meshStandardMaterial color="#7feafd"/>
      </mesh>}
      {id==='pico'&&<mesh position={[0,.52,0]} rotation={[0,0,.25]}>
        <coneGeometry args={[.13,.38,7]}/><meshStandardMaterial color="#71d698"/>
      </mesh>}
    </group>
    <group ref={rightArm} position={[.43,.58,0]}>
      <mesh position={[.06,-.14,.05]}>
        <capsuleGeometry args={[.12,.22,4,10]}/>
        <meshStandardMaterial color="#e2f4fa"/>
      </mesh>
    </group>
    <group ref={leftArm} position={[-.43,.58,0]}>
      <mesh position={[-.06,-.14,.05]}>
        <capsuleGeometry args={[.12,.22,4,10]}/>
        <meshStandardMaterial color="#e2f4fa"/>
      </mesh>
    </group>
    <mesh position={[0,.47,.32]}>
      <sphereGeometry args={[.105,12,9]}/>
      <meshStandardMaterial color={color} emissive={color} emissiveIntensity={.5}/>
    </mesh>
    {active&&<pointLight color={color} position={[0,1,.65]} distance={2.6} intensity={.95}/>}</group>;
}

function Station({id,isActive,onSelect,warning,peer}) {
  const s=STATIONS[id];
  const [x,z]=s.p;
  const workType=id==='forge'?'forge':id==='vera'?'shield':id==='bundle'?'package':id==='metri'?'data':'code';
  // Different default orientations: some focus on their own bench, some observe the floor.
  const workYaw={toki:2.55,routa:-2.5,forge:3.0,model:-2.7,
    captain:-2.4,bundle:1.9,metri:-2.2,vera:2.45}[id]||0;
  const faceYaw=stationFacingYaw(workYaw,x,z,peer?{x:STATIONS[peer].p[0],z:STATIONS[peer].p[1]}:null);
  return <group position={[x,0,z]}>
    <mesh position={[0,.04,0]} receiveShadow>
      <cylinderGeometry args={[1.18,1.24,.2,30]}/>
      <meshStandardMaterial color="#3b6275" metalness={.56} roughness={.25}/>
    </mesh>
    <GlowCircle radius={1.12} color={isActive?'#ffdf91':s.color} y={.16}/>
    <group position={[0,.09,0]}>
      <WorkBench accent={s.color} type={workType}/>
      <group position={[0,0,.37]} onClick={e=>{e.stopPropagation();onSelect?.(id)}}>
        <Character id={id} color={s.color} active={isActive} warning={warning}
          faceYaw={faceYaw} reacting={Boolean(peer)}/>
      </group>
    </group>
    <GardenLabel position={[0,2.0,.7]} center distanceFactor={12} style={{pointerEvents:'none'}}>
      <div className={'gardenWorkerTag'+(isActive?' gardenWorkerActive':'')}
        style={{'--glow':s.color}}>
        <strong>{s.symbol} {s.label}</strong><small>{s.sub}</small>
      </div>
    </GardenLabel>
  </group>;
}

function Pico({target,onArrive}) {
  const group=useRef(),arrived=useRef(''),motion=useRef({moving:false});
  useEffect(()=>{arrived.current=''},[target]);
  useFrame((_,dt)=>{
    const s=STATIONS[target];
    const dest={x:s?s.p[0]+.9:0,z:s?s.p[1]+1.0:.5};
    if(!group.current)return;
    const g=group.current;
    const next=walkStep({x:g.position.x,z:g.position.z,yaw:g.rotation.y},dest,dt);
    g.position.x=next.x;g.position.z=next.z;g.rotation.y=next.yaw;
    motion.current={moving:next.moving};
    if(!next.moving && arrived.current!==target){
      arrived.current=target;
      onArrive?.(target);
    }
  });
  return <group ref={group} position={[0,.15,.5]} scale={.8}>
    <Character id="pico" color="#69e8f4" active locomotionRef={motion}/>
    <pointLight position={[0,.5,.2]} color="#5fffe6" intensity={1.1} distance={2.8}/>
    <GardenLabel center position={[0,2.1,0]} distanceFactor={12} style={{pointerEvents:'none'}}>
      <div className="gardenPicoTag">✦ PICO <small>MISSION GUIDE</small></div>
    </GardenLabel>
  </group>;
}

function Handoff({from,to,index}) {
  const a=STATIONS[from],b=STATIONS[to];
  const packet=useRef();
  const curve=useMemo(()=>a&&b?new THREE.QuadraticBezierCurve3(
    new THREE.Vector3(a.p[0],.6,a.p[1]),
    new THREE.Vector3((a.p[0]+b.p[0])/2,2.0,(a.p[1]+b.p[1])/2),
    new THREE.Vector3(b.p[0],.6,b.p[1])):null,[from,to]);
  useFrame(({clock})=>{
    if(packet.current&&curve){
      packet.current.position.copy(curve.getPoint((clock.elapsedTime*.13+index*.16)%1));
    }
  });
  if(!curve)return null;
  return <group>
    <mesh><tubeGeometry args={[curve,28,.025,5,false]}/>
      <meshBasicMaterial color="#65eadb" transparent opacity={.72}/>
    </mesh>
    <mesh ref={packet}>
      <octahedronGeometry args={[.13,0]}/>
      <meshBasicMaterial color="#9cfff0"/>
    </mesh>
  </group>;
}

function CameraDirector({follow,target,controlsRef}) {
  const {camera}=useThree();
  const old=useRef(follow);
  useEffect(()=>{
    if(old.current&&!follow){
      camera.position.set(0,10.0,16.6);
      camera.lookAt(0,1,0);
      controlsRef.current?.target.set(0,1,0);
      controlsRef.current?.update();
    }
    old.current=follow;
  },[follow,camera,controlsRef]);
  useFrame((_,dt)=>{
    if(!follow)return;
    const s=STATIONS[target];
    if(!s)return;
    const desired=new THREE.Vector3(s.p[0]+2.1,6.8,s.p[1]+7.8);
    camera.position.lerp(desired,Math.min(1,dt*1.15));
    camera.lookAt(s.p[0],.9,s.p[1]);
  });
  return null;
}

export default function NimboGardenWorld({
  target='captain',onSelect,busy=false,liveExecute=false,onArrive,
  theme='night',events=[],followWorker=false,started=false,
  modelMission=null, selectedRecordSeq=null
}) {
  const day=theme==='light'||theme==='day';
  const controlsRef=useRef();
  // Keep the labels' DOM outside React Three Fiber's canvas wrapper during teardown.
  const [labelRoot,setLabelRoot]=useState(null);
  const labelPortal=useMemo(()=>({current:labelRoot}),[labelRoot]);
  const sourceEvents=modelMission?.events||events;
  const historical=modelMission?.mode==='historical';
  const visibleEvents=useMemo(()=>replayEventsUntil(sourceEvents,selectedRecordSeq,historical),
    [sourceEvents,selectedRecordSeq,historical]);
  const evidence=useMemo(()=>gardenEvidence(visibleEvents),[visibleEvents]);
  // One visible handoff: the latest recorded stage transition, not simulated agent reasoning.
  const links=useMemo(()=>gardenHandoffs(visibleEvents).slice(-1),[visibleEvents]);
  const atEnd=isAtRecordedEnd(sourceEvents,selectedRecordSeq,historical);
  const live=Boolean(busy&&liveExecute);
  const stages=[
    ['PyTorch baseline',evidence.baseline],
    ['ONNX benchmark',evidence.onnx],
    ['Held-out gates',evidence.heldout],
    ['Offline package',evidence.package]
  ];
  return <div className={'nimboGardenWorld '+(day?'gardenDay':'gardenNight')}>
    <div className="gardenLabels" ref={setLabelRoot}/>
    {labelRoot&&<Canvas className="nimboGardenCanvas" dpr={[1,1.6]}
      camera={{position:[0,10,16.6],fov:44,near:.1,far:100}}
      shadows gl={{antialias:true,powerPreference:'high-performance'}}>
      <GardenLabelPortal.Provider value={labelPortal}>
      <color attach="background" args={[day?'#b7d8d3':'#091d2e']}/>
      <fog attach="fog" args={[day?'#a9d4cf':'#0a2635',22,53]}/>
      <ambientLight intensity={day?1.45:.92}/>
      <hemisphereLight args={['#e6fff5','#224f63',day?1.55:1]}/>
      <directionalLight position={[-5,11,8]} intensity={day?2.4:2.15} castShadow
        shadow-mapSize={[1024,1024]} shadow-camera-left={-15}
        shadow-camera-right={15} shadow-camera-top={12} shadow-camera-bottom={-12}/>
      <pointLight position={[0,7,-4]} intensity={1.5} color="#69e9eb" distance={19}/>
      <Atrium day={day}/>
      <ZonePod x={-6.05} z={-4.15} w={5.1} d={3.7} accent="#b4a1ef"
        title="EXECUTION STUDIO" subtitle="Context · Routing · Runtime"/>
      <ZonePod x={5.6} z={-4.15} w={4.8} d={3.7} accent="#79e6d8"
        title="VISION PAVILION" subtitle="Visual workloads · Other tools"/>
      <ZonePod x={-5.35} z={3.0} w={3.4} d={3.2} accent="#91d7fb"
        title="MISSION CONTROL" subtitle="Intake · Orchestration"/>
      <ZonePod x={5.4} z={3.0} w={3.4} d={3.2} accent="#79eeb6"
        title="VALIDATION GROVE" subtitle="Quality · Safety · Gates"/>
      {Object.keys(STATIONS).map(id=><Station key={id} id={id}
        isActive={started&&target===id} warning={evidence.agentFailed&&id==='vera'}
        peer={links[0]?.[0]===id?links[0]?.[1]:
          links[0]?.[1]===id?links[0]?.[0]:null}
        onSelect={onSelect}/>) }
      {links.map(([from,to],i)=><Handoff key={`${from}-${to}-${i}`}
        from={from} to={to} index={i}/>) }
      <Pico target={target} onArrive={onArrive}/>
      <OrbitControls ref={controlsRef} enabled={!followWorker}
        enableZoom enablePan enableDamping dampingFactor={.07}
        minDistance={7} maxDistance={35}
        minPolarAngle={.4} maxPolarAngle={Math.PI/2.07}
        target={[0,1,0]}/>
      <CameraDirector controlsRef={controlsRef} follow={followWorker} target={target}/>
      </GardenLabelPortal.Provider>
    </Canvas>}
    <div className={'gardenMissionScreen '+(!evidence.count?'gardenScreenIdle':'')}
      aria-live="polite">
      <div className="gardenScreenTop"><span>NIMBO · REAL OPTIMIZATION EVIDENCE</span>
        <b>{live?'● LIVE':evidence.count?'RECORDED · NOT LIVE':'● READY'}</b>
      </div>
      {evidence.count? <>
        <h3>{evidence.package?'M7 · Verified review package':live?'Optimization running':historical?'Replay · evidence at this event':'Recorded mission evidence'}</h3>
        <div className="gardenCheckpointGrid">
          {stages.map(([name,passed])=><div key={name} className={passed?'passed':'pending'}>
            <strong>{passed?'✓':'○'} {name}</strong>
            <small>{passed?'Recorded as passed':'Not yet recorded'}</small>
          </div>)}
        </div>
        {atEnd&&evidence.agentFailed&&<div className="gardenAgentWarning">
          <b>M11 review failed.</b> M7 measurements and package result are preserved.
        </div>}
        {atEnd&&!evidence.agentFailed&&evidence.agentReviewed&&<div className="gardenAgentReview">
          VERA review-only decision recorded. Not production approval.
        </div>}
        <div className="gardenScreenFooter"><span>{historical?`${evidence.count} / ${sourceEvents.length} recorded events`:`${evidence.count} backend events`}</span>
          <span>{STATIONS[target]?.label||'PICO'}</span></div>
      </> : <div className="gardenReadyLine">
        <span>✦ World online</span><span>Start a real mission or replay evidence</span>
      </div>}
    </div>
  </div>;
}
