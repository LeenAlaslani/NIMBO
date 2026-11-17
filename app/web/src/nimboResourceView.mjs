/* M12.4: precise units and null-safe visual formatting; no invented telemetry. */
export const measured=n=>typeof n==='number'&&Number.isFinite(n)&&n>=0;
export function cpuLabel(n){return measured(n)?`${n.toFixed(1)}%`:'—';}
export function mibLabel(n){return measured(n)?`${n.toFixed(1)} MiB`:'—';}
export function percentLabel(n){return measured(n)?`${n.toFixed(1)}%`:'—';}
export function fill(n,denominator=100){return measured(n)&&measured(denominator)&&denominator>0?Math.max(0,Math.min(100,100*n/denominator)):0;}
export function points(history,key){
  if(!Array.isArray(history))return '';
  const measuredPoints=history.map((p,i)=>({x:i,y:p?.[key]})).filter(p=>measured(p.y));
  if(measuredPoints.length<2)return '';
  const ymax=Math.max(1,...measuredPoints.map(p=>p.y));
  const span=Math.max(1,history.length-1);
  return measuredPoints.map(p=>`${(240*p.x/span).toFixed(2)},${(48-44*p.y/ymax).toFixed(2)}`).join(' ');
}
