/** UI-only routing of verified backend stages to task-specific robots. */
export const MISSION_WORKERS=Object.freeze({captain:'CAPTAIN',forge:'FORGE',vera:'VERA',bundle:'BUNDLE'});
export const MISSION_FLOW=Object.freeze(['captain','forge','vera','bundle']);
const BY_STAGE=Object.freeze({queued:'captain',analyze:'captain',optimize:'forge',verify:'vera',deliver:'bundle'});
export function missionWorkerForEvent(event,timeline=[]) {
  if(!event)return null;
  if(Object.hasOwn(BY_STAGE,event.stage))return BY_STAGE[event.stage];
  const earlier=timeline.filter(e=>e&&Number.isSafeInteger(e.seq)&&e.seq<event.seq&&Object.hasOwn(BY_STAGE,e.stage));
  return earlier.length?BY_STAGE[earlier.at(-1).stage]:null;
}
