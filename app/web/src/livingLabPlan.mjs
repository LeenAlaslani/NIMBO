/** M10: stable room/worker map. This never invents backend execution. */
export const ROOMS = Object.freeze({
  execution:{id:'execution',x:-20,title:'EXECUTION STUDIO',subtitle:'Routing · Context · Inference',color:'#54cff8'},
  forge:{id:'forge',x:0,title:'MODEL FORGE',subtitle:'Baseline · Optimization · Verification',color:'#c6a2ff'},
  vision:{id:'vision',x:20,title:'VISION STUDIO',subtitle:'Image generation · Quality',color:'#68ebc5'},
});
export const ROOM_ORDER=Object.freeze(['execution','forge','vision']);
const STATIONS={captain:[-5.2,2.1],toki:[-3.2,-1.2],routa:[-.5,-1.7],model:[3.4,-1.4],vera:[5,2],metri:[1,3.2],forge:[1.2,-3],bundle:[-2.6,3.1]};
export function roomForMission(operation='',mode='classification'){
  if(operation==='auto_onnx'||['optimize','optimize_onnx','validate_onnx'].includes(operation))return 'forge';
  return mode==='image'?'vision':'execution';
}
export function workerLocations(room='execution'){
  const current=ROOMS[room]?room:'execution';
  return Object.fromEntries(Object.entries(STATIONS).map(([id,[x,z]])=>{
    let area=current;
    if(current==='execution'&&['forge','bundle'].includes(id))area='forge';
    if(current==='forge'&&['toki','routa'].includes(id))area='execution';
    if(current==='vision'&&['forge','bundle'].includes(id))area='forge';
    if(current==='vision'&&id==='toki')area='execution';
    return [id,{x:x+ROOMS[area].x,z,room:area}];
  }));
}
export const ROLES={
  captain:'Checks the mission request and the available workload.',
  toki:'Explains real context and token processing when connected.',
  routa:'Explains the actual model route chosen by the backend.',
  model:'Examines model execution and baseline measurements.',
  forge:'Builds the ONNX optimization candidate when the real optimizer reaches that phase.',
  vera:'Checks recorded quality, latency and CPU acceptance gates.',
  metri:'Interprets measured evidence and its limitations.',
  bundle:'Validates package integrity and independent reload for review delivery.'
};
const STAGE_WORKER={queued:'captain',analyze:'captain',optimize:'forge',verify:'vera',deliver:'bundle'};
export function eventWorker(e,timeline=[]){
  if(!e)return null;
  if(Object.hasOwn(STAGE_WORKER,e.stage))return STAGE_WORKER[e.stage];
  const past=timeline.filter(x=>Number.isSafeInteger(x?.seq)&&x.seq<e.seq&&Object.hasOwn(STAGE_WORKER,x.stage));
  return past.length?STAGE_WORKER[past.at(-1).stage]:null;
}
const TERMINAL=new Set(['COMPLETED_REVIEW_ONLY','COMPLETED_REJECTED','FAILED','INTERRUPTED']);
export function finalJob(job){return TERMINAL.has(job?.state)}
export function outcome(job){
  switch(job?.state){
    case 'COMPLETED_REVIEW_ONLY':return {kind:'verified',title:'Review package verified',description:'Independent offline package reload passed. Review only, not production deployment.'};
    case 'COMPLETED_REJECTED':return {kind:'rejected',title:'Optimization rejected',description:'Quality or performance gates rejected the candidate. Original model preserved.'};
    case 'FAILED':return {kind:'failed',title:'Mission failed',description:'The backend stopped without an approved optimized package.'};
    case 'INTERRUPTED':return {kind:'failed',title:'Mission interrupted',description:'Execution was interrupted. No approval is implied.'};
    case 'RUNNING':return {kind:'running',title:'Mission in progress',description:'PICO follows actual backend events, never a fake timer.'};
    case 'QUEUED':return {kind:'running',title:'Mission queued',description:'Waiting for recorded backend execution.'};
    default:return {kind:'ready',title:'Mission ready',description:'Start or load a real job and follow PICO through the laboratory.'};
  }
}
export function workerDialogue(id,event,job){
  const name=id==='model'?'MAKO':String(id||'captain').toUpperCase();
  if(event&&eventWorker(event)===id)return {name,title:`${String(event.stage).toUpperCase()} · ${event.state||'RECORDED'}`,message:String(event.message||'No evidence supplied').slice(0,500),recorded:true};
  if(event&&finalJob(job)&&id==='vera')return {name,title:outcome(job).title,message:outcome(job).description,recorded:true};
  return {name,title:'Station briefing',message:ROLES[id]||'No task assigned to this station.',recorded:false};
}
