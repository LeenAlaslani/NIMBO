import React, {useMemo, useRef} from 'react';
import {Canvas, useFrame, useThree} from '@react-three/fiber';
import {ContactShadows, Html, RoundedBox, Text} from '@react-three/drei';
import * as THREE from 'three';

const PLACES = {
  analyze: [-2.9, 0, 1.35],
  route: [-1.55, 0, -1.25],
  execute: [0.45, 0, -1.65],
  verify: [2.75, 0, -0.85],
  deliver: [2.25, 0, 1.5],
};
const LABELS = [['INPUT','analyze'], ['ROUTING','route'], ['MODEL BAY','execute'], ['VERIFY','verify'], ['DELIVERY','deliver']];
const STAGE_COLORS = {
  analyze: '#72c6ff',
  route: '#ae8eff',
  execute: '#63e6cc',
  verify: '#ffcf7a',
  deliver: '#74f0ad',
  success: '#7ef0b1',
  error: '#ff7c99',
  idle: '#8ea6c6'
};
const THEMES = {
  night: {
    background: '#08152d',
    floor: '#12233d',
    gridA: '#315577',
    gridB: '#203750',
    wall: '#172c48',
    banner: '#193d66',
    rack: '#17243a',
    column: '#21334b',
    path: '#287c9b',
    screenOff: '#12324d',
    console: '#58718e',
    seat: '#8da4c2',
    topGlow: '#afcbff',
    sceneLabel: '#b3d9ff',
    ambient: 1.55,
    directional: 2.2,
    accent: '#6be4ff',
  },
  day: {
    background: '#edf5ff',
    floor: '#dfeafa',
    gridA: '#b3c8ea',
    gridB: '#d0dbed',
    wall: '#dce9f9',
    banner: '#b7d3f5',
    rack: '#c8daef',
    column: '#d0e0f2',
    path: '#73bfd1',
    screenOff: '#5a84ac',
    console: '#c5d5e7',
    seat: '#b1c2d7',
    topGlow: '#e7f2ff',
    sceneLabel: '#204268',
    ambient: 1.9,
    directional: 2.8,
    accent: '#3fa4e8',
  }
};

function Mat({color='#243e60', emissive='#000000', intensity=0, metalness=0.38, roughness=0.42}){
  return <meshStandardMaterial color={color} emissive={emissive} emissiveIntensity={intensity} metalness={metalness} roughness={roughness}/>;
}
function Panel({x=0,y=0,z=0,w=1,h=1,color='#18395a'}){
  return <mesh position={[x,y,z]}><boxGeometry args={[w,h,0.055]}/><meshStandardMaterial color={color} emissive={color} emissiveIntensity={0.35}/></mesh>;
}

function Station({name, id, active, theme}){
  const [x,,z] = PLACES[id];
  const palette = THEMES[theme] || THEMES.night;
  const on = active === id;
  const light = STAGE_COLORS[id];
  return <group position={[x,0,z]}>
    <mesh position={[0,0.06,-0.2]} castShadow receiveShadow><boxGeometry args={[1.85,0.12,1.6]}/><Mat color={theme==='day' ? '#d5e3f2' : '#263a58'}/></mesh>
    <mesh position={[0,0.65,-0.5]} castShadow><boxGeometry args={[1.65,0.09,0.7]}/><Mat color={palette.console}/></mesh>
    {[-0.72,0.72].map((a) => <mesh key={a} position={[a,0.34,-0.5]} castShadow><boxGeometry args={[0.09,0.55,0.09]}/><Mat color={theme==='day' ? '#a3b8d0' : '#506881'}/></mesh>)}
    <mesh position={[0,1.2,-0.8]} castShadow><boxGeometry args={[1.5,1.25,0.16]}/><Mat color={theme==='day' ? '#d6e7fb' : '#0a1b32'}/></mesh>
    <Panel x={0} y={1.2} z={-0.70} w={1.31} h={1.02} color={on ? light : palette.screenOff}/>
    <Text position={[0,1.58,-0.656]} fontSize={0.155} color={on ? '#ffffff' : palette.sceneLabel}>{name}</Text>
    {[0,1,2].map((i) => <group key={i}><Panel x={-0.44+i*0.44} y={1.19} z={-0.648} w={0.29} h={0.11} color={i===1 && on ? light : (theme==='day' ? '#7d9ec0' : '#32618e')}/><Panel x={-0.44+i*0.44} y={0.95} z={-0.648} w={0.32} h={0.05} color={theme==='day' ? '#96b0ca' : '#6183ac'}/></group>)}
    <mesh position={[0,0.72,-0.18]}><boxGeometry args={[0.62,0.035,0.25]}/><Mat color={palette.seat}/></mesh>
    <mesh position={[0,0.68,0.08]}><boxGeometry args={[0.2,0.11,0.28]}/><Mat color={theme==='day' ? '#b8c6d7' : '#36445a'}/></mesh>
    <mesh rotation={[-Math.PI/2,0,0]} position={[0,0.015,0.8]}><ringGeometry args={[0.32,0.37,44]}/><meshBasicMaterial color={light} transparent opacity={on ? 0.96 : 0.25}/></mesh>
    {on && <pointLight position={[0,1.4,0.5]} color={light} intensity={theme==='day' ? 1.6 : 2.7} distance={2.7}/>}
  </group>;
}

function Rack({x,z,theme}){
  const palette = THEMES[theme] || THEMES.night;
  return <group position={[x,0,z]}>
    <mesh position={[0,1.55,0]} castShadow><boxGeometry args={[1.25,3.1,0.78]}/><Mat color={palette.rack} metalness={0.64}/></mesh>
    {Array.from({length:7}, (_,i) => <group key={i}><mesh position={[0,0.35+i*0.4,0.4]}><boxGeometry args={[1.02,0.24,0.025]}/><Mat color={theme==='day' ? '#b7cae3' : '#243d59'}/></mesh>{[-0.35,0,0.35].map((j) => <mesh key={j} position={[j,0.35+i*0.4,0.42]}><boxGeometry args={[0.15,0.035,0.04]}/><meshBasicMaterial color={i%2 ? '#63a3f7' : '#6cebd5'}/></mesh>)}</group>)}
  </group>;
}

function Lab({theme}){
  const palette = THEMES[theme] || THEMES.night;
  return <>
    <mesh rotation={[-Math.PI/2,0,0]} receiveShadow><planeGeometry args={[22,20]}/><Mat color={palette.floor} metalness={0.2}/></mesh>
    <gridHelper args={[18,18,palette.gridA,palette.gridB]} position={[0,0.013,0]}/>
    <mesh position={[0,2.55,-5.5]} receiveShadow><boxGeometry args={[19,5.1,0.25]}/><Mat color={palette.wall}/></mesh>
    <mesh position={[0,4.03,-5.3]}><boxGeometry args={[14,0.85,0.08]}/><Mat color={palette.banner} emissive={palette.banner} intensity={0.25}/></mesh>
    <Text position={[0,4.07,-5.23]} fontSize={0.37} color={palette.sceneLabel}>NIMBO / AI OPERATIONS LAB</Text>
    {[-6.5,6.5].map((x) => <group key={x}><mesh position={[x,2.5,-2.2]} castShadow><boxGeometry args={[0.34,5,7]}/><Mat color={palette.column}/></mesh><mesh position={[x,4.85,-2.2]}><boxGeometry args={[0.14,0.14,7]}/><meshBasicMaterial color={palette.topGlow}/></mesh></group>)}
    {[[-5.15,-4.35],[5.15,-4.35],[-5.15,-2.2],[5.15,-2.2]].map(([x,z]) => <Rack key={`${x}:${z}`} x={x} z={z} theme={theme}/>)}
    {[-3,0,3].map((x) => <group key={x}><mesh position={[x,4.55,-0.35]}><boxGeometry args={[2,0.12,0.45]}/><meshBasicMaterial color={palette.topGlow}/></mesh><pointLight position={[x,4.0,-0.4]} color={theme==='day' ? '#d6eeff' : '#afcbff'} intensity={theme==='day' ? 2.3 : 4} distance={5}/></group>)}
    <mesh position={[-5,1.2,1.6]}><boxGeometry args={[0.85,2.3,1.2]}/><Mat color={theme==='day' ? '#d2e1f1' : '#243850'}/></mesh>
    <mesh position={[5.1,1.2,1.3]}><boxGeometry args={[0.9,2.3,1.2]}/><Mat color={theme==='day' ? '#d2e1f1' : '#203551'}/></mesh>
    <mesh rotation={[-Math.PI/2,0,0]} position={[0,0.022,0.25]}><planeGeometry args={[0.10,8.6]}/><meshBasicMaterial color={palette.path} transparent opacity={theme==='day' ? 0.25 : 0.35}/></mesh>
  </>;
}


// The characters represent UI roles, NOT separate inference processes.
// Backend stage is the single live source for lights and worker activity.
const WORKERS={
  classification:{x:-2.9,z:-.15,label:'CLASSIFIER',accent:'#6cbcff'},
  generation:{x:.35,z:-.25,label:'GENERATOR',accent:'#b59aff'},
  image:{x:3.0,z:.05,label:'IMAGE',accent:'#4cd9ca'}
};
const DOCK=[-4.25,0,3.1];
const MANAGER=[-.25,0,2.0];
const INSPECTOR=[3.8,0,2.55];
const activeWorkload=(w)=>WORKERS[w] ? w : 'classification';

function CloudBot({x,z,role='worker',accent='#7fc8fd',active=false,focused=false,disabled=false,theme='night',size=.65}){
 const head=useRef(),arms=useRef([]),eyeL=useRef(),eyeR=useRef();
 const front=role==='manager' ? .45 : role==='inspector' ? -.15 : -.05;
 useFrame(({clock},delta)=>{
   const t=clock.elapsedTime, d=Math.min(delta,.05);
   if(head.current){head.current.rotation.z=THREE.MathUtils.damp(head.current.rotation.z,active?.07:Math.sin(t*.65)*.025,4,d);}
   const beat=active && focused ? Math.sin(t*11)*.18 : Math.sin(t*1.6)*.025;
   for(const arm of arms.current.filter(Boolean))arm.rotation.x=THREE.MathUtils.damp(arm.rotation.x,active?-.56+beat:.10,9,d);
   const blink=Math.abs(Math.sin(t*.55 + x))>.985?.13:1;
   if(eyeL.current)eyeL.current.scale.y=THREE.MathUtils.damp(eyeL.current.scale.y,blink,12,d);
   if(eyeR.current)eyeR.current.scale.y=THREE.MathUtils.damp(eyeR.current.scale.y,blink,12,d);
 });
 const color=disabled?'#93a2b1':active?accent:'#74bce7';
 const white=<meshStandardMaterial color={theme==='day'?'#ffffff':'#e9f2ff'} roughness={.35} metalness={.08}/>;
 return <group position={[x,0,z]} rotation={[0,front,0]} scale={size}>
   <group ref={head}>
    <mesh position={[0,1.26,0]} castShadow><sphereGeometry args={[.62,20,18]}/>{white}</mesh>
    {[[-.4,1.59,-.04,.33],[0,1.78,-.08,.4],[.4,1.55,-.04,.32]].map(([cx,cy,cz,r],i)=><mesh key={i} position={[cx,cy,cz]} castShadow><sphereGeometry args={[r,16,14]}/>{white}</mesh>)}
    <RoundedBox args={[1.08,.58,.12]} position={[0,1.28,.49]} radius={.22} smoothness={4}><meshStandardMaterial color={theme==='day'?'#355074':'#09152d'}/></RoundedBox>
    <mesh ref={eyeL} position={[-.22,1.34,.58]}><sphereGeometry args={[.115,16,12]}/><meshBasicMaterial color={color}/></mesh>
    <mesh ref={eyeR} position={[.22,1.34,.58]}><sphereGeometry args={[.115,16,12]}/><meshBasicMaterial color={color}/></mesh>
   </group>
   {role==='manager'&&<mesh position={[0,1.93,0]}><cylinderGeometry args={[.13,.20,.16,16]}/><Mat color="#f3bf69"/></mesh>}
   {role==='inspector'&&<mesh position={[.43,1.30,.27]}><torusGeometry args={[.15,.045,8,18]}/><meshBasicMaterial color="#f3c46d"/></mesh>}
   {role==='worker'&&<mesh position={[0,1.86,0]}><boxGeometry args={[.48,.09,.36]}/><Mat color={accent}/></mesh>}
   <mesh position={[0,.62,0]} castShadow><sphereGeometry args={[.4,18,16]}/>{white}</mesh>
   {[-1,1].map((side,i)=><group key={side} ref={n=>arms.current[i]=n} position={[side*.48,.78,0]}><mesh position={[side*.035,-.20,.06]}><capsuleGeometry args={[.13,.25,4,10]}/>{white}</mesh></group>)}
   {[-.22,.22].map(v=><mesh key={v} position={[v,.2,.12]}><capsuleGeometry args={[.15,.17,4,10]}/><Mat color={theme==='day'?'#b7cfe9':'#bdd7f4'}/></mesh>)}
 </group>
}

function WorkDesk({type,theme,active,offline=false}){
 const w=WORKERS[type], y=.95, glow=active?w.accent:theme==='day'?'#668cb5':'#34668c';
 return <group position={[w.x,0,w.z-.9]}>
  <mesh position={[0,.56,0]} castShadow><boxGeometry args={[1.55,.09,.76]}/><Mat color={theme==='day'?'#c4d7eb':'#3f5776'}/></mesh>
  {[-.62,.62].map(x=><mesh key={x} position={[x,.29,0]}><boxGeometry args={[.08,.54,.08]}/><Mat color={theme==='day'?'#91aecb':'#4a6382'}/></mesh>)}
  <mesh position={[0,1.2,-.16]}><boxGeometry args={[1.36,.92,.12]}/><Mat color={theme==='day'?'#234268':'#11243d'}/></mesh>
  <mesh position={[0,1.2,-.07]}><boxGeometry args={[1.15,.72,.035]}/><Mat color={offline?'#66758b':active?glow:theme==='day'?'#628caf':'#235377'} emissive={active?glow:'#000000'} intensity={active?.2:0}/></mesh>
  <Text position={[0,1.44,-.035]} fontSize={.15} color="#ffffff">{w.label}</Text>
  <Text position={[0,1.15,-.029]} fontSize={.115} color="#ffffff">{offline?'GPU OFFLINE':active?'RUNNING':'READY / IDLE'}</Text>
  <mesh position={[0,.62,.12]}><boxGeometry args={[.66,.025,.23]}/><Mat color={theme==='day'?'#7699b8':'#92abc7'}/></mesh>
  <mesh rotation={[-Math.PI/2,0,0]} position={[0,.022,.62]}><ringGeometry args={[.34,.39,32]}/><meshBasicMaterial color={w.accent} transparent opacity={active?.9:.23}/></mesh>
  {active&&<pointLight position={[0,1.2,0.2]} color={w.accent} intensity={theme==='day'?1.3:2.5} distance={2.3}/>}
 </group>
}

function Parcel({visible,workload}){
 if(!visible)return null;
 const c=WORKERS[workload]?.accent||'#f7bd73';
 return <group position={[0,1.95,0]}>
  <mesh castShadow><boxGeometry args={[.67,.42,.53]}/><meshStandardMaterial color="#b67d43" roughness={.36} metalness={.15}/></mesh>
  <mesh position={[0,.215,0]}><boxGeometry args={[.69,.035,.53]}/><meshStandardMaterial color="#f2ce8c"/></mesh>
  <mesh position={[0,.01,.27]}><planeGeometry args={[.5,.17]}/><meshBasicMaterial color={c}/></mesh>
 </group>
}

function Courier({stage,workload,resetKey,theme,reducedMotion,delivered}){
 const g=useRef(),l=useRef(),r=useRef(),a=useRef();
 const type=activeWorkload(workload),w=WORKERS[type];
 // Only move on actual stage events. Backend never waits for the courier.
 const destination=stage==='analyze'?DOCK:stage==='route'?MANAGER:stage==='execute'?[w.x,0,w.z+.95]:stage==='verify'?INSPECTOR:stage==='deliver'?[3.3,0,3.25]:DOCK;
 useFrame(({clock},dt)=>{
  if(!g.current)return;
  const d=Math.min(.05,dt), dx=destination[0]-g.current.position.x,dz=destination[2]-g.current.position.z;
  const distance=Math.hypot(dx,dz),moving=distance>.06;
  if(moving){const step=Math.min(distance,d*(reducedMotion?12:2.1));g.current.position.x+=dx/distance*step;g.current.position.z+=dz/distance*step;
   if(!reducedMotion)g.current.rotation.y=THREE.MathUtils.damp(g.current.rotation.y,Math.atan2(dx,dz),8,d);
  }else g.current.rotation.y=THREE.MathUtils.damp(g.current.rotation.y,Math.PI,5,d);
  if(l.current)l.current.rotation.x=THREE.MathUtils.damp(l.current.rotation.x,moving?Math.sin(clock.elapsedTime*12)*.4:0,10,d);
  if(r.current)r.current.rotation.x=THREE.MathUtils.damp(r.current.rotation.x,moving?-Math.sin(clock.elapsedTime*12)*.4:0,10,d);
  if(a.current)a.current.rotation.x=THREE.MathUtils.damp(a.current.rotation.x,moving?-.22:stage==='execute'?-.62:-.12,8,d);
  g.current.position.y=reducedMotion?0:moving?Math.abs(Math.sin(clock.elapsedTime*12))*.035:Math.sin(clock.elapsedTime*1.7)*.018;
 });
 return <group ref={g} key={resetKey} position={DOCK} scale={.65}>
   <CloudBot x={0} z={0} role="courier" accent="#f3c57c" active={stage==='analyze'} theme={theme} size={1}/>
   <Parcel visible={!delivered} workload={type}/>
   <group ref={l} position={[-.2,0,0]}/><group ref={r} position={[.2,0,0]}/><group ref={a}/>
 </group>
}

function SceneCamera({theme,reducedMotion}){
 const {camera}=useThree(),target=useMemo(()=>new THREE.Vector3(0,1.3,0),[]);
 useFrame((_,dt)=>{
  const desired=new THREE.Vector3(8.9,6.8,10.2);
  camera.position.lerp(desired,Math.min(1,dt*(reducedMotion?7:1.7)));
  camera.lookAt(target);
 });
 return null;
}

export default function NimboRobot({location='analyze',mode='idle',reducedMotion=false,theme='night',workload='classification',imageConnected=false,resetKey=0,replaying=false}){
 const palette=THEMES[theme]||THEMES.night;
 const type=activeWorkload(workload);
 const executing=location==='execute'&&mode!=='error';
 const isLiveActive=executing && (replaying || mode==='execute');
 return <Canvas key={resetKey} camera={{position:[8.9,6.8,10.2],fov:50}} shadows dpr={[1,1.5]} gl={{antialias:true}}>
  <color attach="background" args={[palette.background]}/>
  <ambientLight intensity={theme==='day'?.95:1.55}/>
  <hemisphereLight args={[theme==='day'?'#ffffff':'#7d99e3',theme==='day'?'#8a9fb6':'#0f223a',theme==='day'?1.35:.55]}/>
  <directionalLight position={[2,10,5]} intensity={theme==='day'?2.4:2.2} castShadow shadow-mapSize={[512,512]}/>
  <pointLight position={[-3,3,-2]} intensity={theme==='day'?2.5:12} color={palette.accent} distance={10}/>
  <Lab theme={theme}/>
  <group>
   {Object.keys(WORKERS).map(k=><React.Fragment key={k}>
    <WorkDesk type={k} theme={theme} active={isLiveActive&&type===k} offline={k==='image'&&!imageConnected}/>
    <CloudBot x={WORKERS[k].x} z={WORKERS[k].z+.37} role="worker" accent={WORKERS[k].accent} active={isLiveActive&&type===k} focused={isLiveActive&&type===k} disabled={k==='image'&&!imageConnected} theme={theme} size={.62}/>
   </React.Fragment>)}
   <CloudBot x={MANAGER[0]} z={MANAGER[2]} role="manager" accent="#f2c37f" active={location==='route'} theme={theme} size={.78}/>
   <CloudBot x={INSPECTOR[0]} z={INSPECTOR[2]} role="inspector" accent={mode==='error'?'#ff859a':mode==='success'?'#72edaa':'#f7c779'} active={location==='verify'} theme={theme} size={.6}/>
   <Courier stage={location} workload={type} resetKey={resetKey} theme={theme} reducedMotion={reducedMotion} delivered={location==='deliver'}/>
  </group>
  <SceneCamera theme={theme} reducedMotion={reducedMotion}/>
  <ContactShadows position={[0,.018,0]} scale={13} opacity={theme==='day'?.14:.26} blur={2.5} far={4}/>
 </Canvas>;
}
