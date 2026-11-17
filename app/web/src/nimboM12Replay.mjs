
/* M12: read-only replay and backend-grounded station routing. */

export function nextRecordedSeq(events,currentSeq=null){
  if(!Array.isArray(events))return null;

  const seqs=[...new Set(
    events
      .map(e=>e?.seq)
      .filter(n=>Number.isSafeInteger(n)&&n>0)
  )].sort((a,b)=>a-b);

  if(currentSeq===null)return seqs[0]??null;
  return seqs.find(n=>n>currentSeq)??null;
}

export function m12PickWorker(event,timeline,fallback){
  if(!event)return 'captain';

  const role=String(event.evidence?.agent_role||'')
    .toLowerCase();

  if(['captain','forge','vera','metri'].includes(role))
    return role;

  const message=String(event.message||'');

  const decision=/^(CAPTAIN|FORGE|VERA|METRI) decision:/i
    .exec(message);

  if(decision)return decision[1].toLowerCase();

  if(/^M7 engine:.*PyTorch.*baseline/i.test(message))
    return 'model';

  if(/^M7 engine:.*ONNX FP32/i.test(message))
    return 'forge';

  if(/^M7 engine:.*(heldout|quality|pilot)/i.test(message))
    return 'vera';

  if(/^M7 engine:.*(package|offline model)/i.test(message))
    return 'bundle';

  return typeof fallback==='function'
    ? fallback(event,timeline)
    : 'captain';
}
