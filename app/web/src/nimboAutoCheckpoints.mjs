/* M12.4: checkpoint labels reflect actual backend events, not animation timers. */
const stateOf=(messages,start,success,failure)=>{
  if(messages.some(m=>failure.test(m)))return 'REJECTED';
  if(messages.some(m=>success.test(m)))return 'RECORDED';
  return messages.some(m=>start.test(m))?'ACTIVE':'PENDING';
};
export function recordedCheckpoints(events){
 const messages=(Array.isArray(events)?events:[]).map(e=>String(e?.message||'').replace(/^M7 engine:\s*/i,''));
 return [
  {key:'baseline',title:'PyTorch baseline',state:stateOf(messages,/Starting real pinned PyTorch CPU baseline/i,/PyTorch baseline measured/i,/baseline (failed|rejected)/i)},
  {key:'onnx',title:'ONNX FP32 benchmark',state:stateOf(messages,/Starting ONNX FP32 graph export/i,/ONNX paired benchmark passed local gates|Pilot quality and mean-speed checks passed|ONNX pilot failed required quality/i,/ONNX pilot failed required quality/i)},
  {key:'heldout',title:'Held-out quality gates',state:stateOf(messages,/Starting new heldout paired evaluation/i,/Heldout quality and performance gates passed/i,/ONNX rejected by heldout quality/i)},
  {key:'package',title:'Offline package reload',state:stateOf(messages,/Creating and independently offline-verifying/i,/Offline model package verified; review only/i,/(packager did not certify|package verification failed)/i)},
 ];
}
export function jobHeadline(job,mode){
 if(mode==='historical')return 'RECORDED MISSION · NOT LIVE';
 if(!job)return 'AWAITING BACKEND JOB';
 if(job.state==='FAILED')return 'AGENT / ENGINE FAILURE';
 if(job.state==='COMPLETED_REJECTED')return 'CANDIDATE REJECTED';
 if(job.state==='COMPLETED_REVIEW_ONLY')return 'REVIEW-ONLY RESULT';
 if(job.state==='INTERRUPTED')return 'JOB INTERRUPTED';
 if(job.state==='QUEUED')return 'QUEUED IN BACKEND';
 if(job.state==='RUNNING')return 'BACKEND WORK IN PROGRESS';
 return String(job.state||'UNKNOWN').slice(0,35);
}
