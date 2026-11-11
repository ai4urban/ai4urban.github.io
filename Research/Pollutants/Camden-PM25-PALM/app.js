'use strict';
const $=id=>document.getElementById(id), comparison=true;
let meta,grid,current='baseline',level=3,point=null,request=0;
const fmt=(v,n=3)=>v==null?'—':Number(v).toFixed(n);
const text=(id,v)=>{if($(id))$(id).textContent=v;};
function query(){return `id=${current}&level=${level}&view=${comparison?'comparison':'scenario'}`;}
function status(s,error=false){text('status-text',s);$('status').style.color=error?'#a32929':'';}
function controls(){for(let k of ['emissions','background','wind','rotation'])text(k+'-value',k==='background'?`${fmt($(k).value,2)} µg/m³`:k==='rotation'?`${$(k).value}°`:`${Math.round($(k).value*100)}%`);}
function color(v,lo,hi,diff=false){let t=Math.max(0,Math.min(1,(v-lo)/(hi-lo)));if(diff){let a=t<.5?[38,88,166]:[181,36,45],w=1-Math.abs(t*2-1);return a.map(c=>Math.round(c*(1-w)+248*w));}let stops=[[247,250,218],[127,201,157],[36,131,142],[32,61,106]],p=t*3,i=Math.min(2,Math.floor(p)),f=p-i;return stops[i].map((c,k)=>Math.round(c*(1-f)+stops[i+1][k]*f));}
function drawMap(id,values,diff){let c=$(id),ctx=c.getContext('2d'),im=ctx.createImageData(800,800),lim=grid.difference_limit/+$('contrast').value;for(let y=0;y<800;y++)for(let x=0;x<800;x++){let v=values[y*800+x],o=((799-y)*800+x)*4,rgb=v==null?[231,234,234]:color(v,diff?-lim:grid.scale[0],diff?lim:grid.scale[1],diff);im.data.set([...rgb,255],o);}ctx.putImageData(im,0,0);if(point){ctx.strokeStyle='#111';ctx.lineWidth=2;ctx.beginPath();ctx.arc(point.x,799-point.y,6,0,Math.PI*2);ctx.stroke();}}
function drawAll(){if(!grid)return;drawMap('first-map',grid.first,false);drawMap('second-map',grid.second,false);drawMap('difference-map',grid.difference,true);let lim=grid.difference_limit/+$('contrast').value;document.querySelectorAll('.concentration-low').forEach(e=>e.textContent=fmt(grid.scale[0]));document.querySelectorAll('.concentration-high').forEach(e=>e.textContent=fmt(grid.scale[1]));text('difference-low',fmt(-lim));text('difference-high',fmt(lim));text('colour-note',`Concentration display: ${fmt(grid.scale[0])} to ${fmt(grid.scale[1])} µg/m³ (2nd–98th percentiles). Difference display: ±${fmt(lim)} µg/m³. Numeric values and CSV are unclipped.`);}
function chartContext(id) {
  const canvas = $(id);
  const width = canvas.getBoundingClientRect().width || canvas.width;
  const scale = canvas.width / width;
  const height = canvas.height / scale;
  const context = canvas.getContext('2d');
  context.setTransform(scale, 0, 0, scale, 0, 0);
  context.fillStyle = '#fff';
  context.fillRect(0, 0, width, height);
  context.font = '16px Arial';
  context.fillStyle = '#334';
  return {context, width, height};
}

function chart(id, series, label, zero = false) {
  const {context, width, height} = chartContext(id);
  const values = series.flatMap(line => line.values.filter(value => value !== null));
  if (!values.length) return;
  const lower = Math.min(...values, zero ? 0 : Infinity);
  const upper = Math.max(...values, zero ? 0 : -Infinity);
  const padding = (upper - lower) * .08 || .1;
  const low = lower - padding, high = upper + padding;
  const left = 44, right = width - 28, top = 28, bottom = height - 50;
  const x = value => left + (value - low) / (high - low) * (right - left);
  const y = z => bottom - (z - 15) / 180 * (bottom - top);

  context.textAlign = 'left';
  context.fillText('Height (m)', 6, 16);
  context.strokeStyle = '#c8d1d0';
  context.lineWidth = 1;
  context.textAlign = 'right';
  const heights = bottom - top < 90 ? [15, 95, 195] : [15, 55, 95, 135, 195];
  for (const z of heights) {
    context.beginPath();
    context.moveTo(left, y(z));
    context.lineTo(right, y(z));
    context.stroke();
    context.fillText(z, left - 7, y(z) + 5);
  }

  context.textAlign = 'center';
  const ticks = width < 400 ? 3 : 5;
  for (let i = 0; i < ticks; i++) {
    const value = low + (high - low) * i / (ticks - 1);
    context.fillText(fmt(value, 2), x(value), bottom + 19, 52);
  }
  context.fillText(label, (left + right) / 2, height - 7, width - 16);
  if (zero) {
    context.setLineDash([6, 4]);
    context.beginPath();
    context.moveTo(x(0), top);
    context.lineTo(x(0), bottom);
    context.stroke();
    context.setLineDash([]);
  }

  for (const line of series) {
    context.strokeStyle = line.color;
    context.lineWidth = 2;
    context.beginPath();
    let connected = false;
    line.values.forEach((value, i) => {
      if (value === null) {
        connected = false;
        return;
      }
      const position = [x(value), y(meta.heights[i])];
      if (connected) context.lineTo(...position);
      else context.moveTo(...position);
      connected = true;
    });
    context.stroke();
  }
}
let pointProfile = null;
function drawProfiles() {
  if (!pointProfile) return;
  chart('profile', [
    {values: pointProfile.first, color: '#315c80'},
    {values: pointProfile.second, color: '#bd643d'}
  ], 'Concentration (µg/m³)');
  chart('difference-profile', [{values: pointProfile.difference, color: '#914d66'}],
    comparison ? 'Error (µg/m³)' : 'Change (µg/m³)', true);
}

async function profile() {
  if (!point) return;
  pointProfile = await api(`/api/profile?${query()}&x=${point.x}&y=${point.y}`);
  text('point-label', `E ${fmt(pointProfile.easting, 1)} m · N ${fmt(pointProfile.northing, 1)} m · model heights`);
  drawProfiles();
  $('profile-download').disabled = false;
}
function scatter() {
  if (!grid) return;
  const {context, width, height} = chartContext('scatter');
  const points = grid.scatter;
  if (!points.length) return;
  const values = points.flat();
  const low = Math.min(...values), high = Math.max(...values);
  const span = high - low || 1;
  const left = 36, right = width - 26, top = 32, bottom = height - 45;
  const x = value => left + (value - low) / span * (right - left);
  const y = value => bottom - (value - low) / span * (bottom - top);
  context.strokeStyle = '#899';
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(x(low), y(low));
  context.lineTo(x(high), y(high));
  context.stroke();
  context.fillStyle = '#367f8b55';
  for (const [reference, prediction] of points) context.fillRect(x(reference), y(prediction), 2, 2);
  context.fillStyle = '#334';
  context.textAlign = 'left';
  context.fillText(comparison ? 'Surrogate (µg/m³)' : 'Scenario (µg/m³)', 8, 17, width - 16);
  context.textAlign = 'center';
  context.fillText(fmt(low, 2), left, bottom + 18, 52);
  context.fillText(fmt(high, 2), right, bottom + 18, 52);
  context.fillText(comparison ? 'PALM (µg/m³)' : 'Baseline (µg/m³)', width / 2, height - 7);
}

const chartResize = new ResizeObserver(() => {
  scatter();
  drawProfiles();
});
for (const id of ['profile', 'difference-profile', 'scatter']) chartResize.observe($(id));
function renderTable(){let columns=[['height','Height (m)'],['cells','Valid cells'],['reference_mean',comparison?'PALM mean':'Baseline mean'],['prediction_mean',comparison?'Surrogate mean':'Scenario mean'],['bias',comparison?'Bias':'Mean change'],['mae',comparison?'MAE':'Mean |change|'],['rmse',comparison?'RMSE':'RMS change'],['r','Pearson r']];$('table-head').replaceChildren();let tr=document.createElement('tr');for(let [,label]of columns){let th=document.createElement('th');th.textContent=label;tr.append(th);}$('table-head').append(tr);$('table-body').replaceChildren();for(let row of grid.table){let tr=document.createElement('tr');tr.tabIndex=0;for(let [key]of columns){let td=document.createElement('td');td.textContent=key==='cells'?row[key].toLocaleString():fmt(row[key],key==='height'?0:3);tr.append(td);}tr.onclick=()=>{level=row.level;$('height').value=level;loadGrid();};tr.onkeydown=e=>{if(e.key==='Enter')tr.click();};$('table-body').append(tr);}}
async function loadGrid(){let serial=++request;status('Loading selected height…');try{let g=await api('/api/grid?'+query());if(serial!==request)return;grid=g;text('second-tag',comparison?'Unchanged inputs':(current==='baseline'?'Same as baseline':'Applied scenario'));text('map-definition',comparison?'PALM and surrogate share one concentration scale; error has a separate zero-centred scale.':'Baseline and scenario share one concentration scale; change has a separate zero-centred scale.');drawAll();scatter();renderTable();let s=g.summary;$('correlation-card').hidden=!comparison;text('correlation-value',fmt(s.r));text('metric-1',fmt(s.reference_mean));text('metric-2',fmt(s.prediction_mean));text('metric-3',fmt(s.bias));text('metric-4',fmt(comparison?s.rmse:s.mae));text('cell-count',`${s.cells.toLocaleString()} valid cells`);text('result-title',`${g.height} m · ${comparison?'PALM comparison':'Scenario result'}`);text('applied',comparison?`Unchanged inputs · error = surrogate − PALM · ${meta.reference_case}`:`Applied: emissions ${Math.round(g.parameters.emissions*100)}%, background ${g.parameters.background} µg/m³, wind ${Math.round(g.parameters.wind*100)}%, clockwise rotation ${g.parameters.rotation}°`);text('saved-note','Download the reference concentrations and metrics below.');$('download-csv').onclick=downloadCells;$('download-csv').classList.remove('disabled');$('download-metrics').href='./data/metrics.csv';$('figure-download').disabled=false;status(s.cells?'Ready · '+meta.statistics_scope:'No valid atmospheric Camden cells at this height. Select a higher model level.');text('range-note','Both input hours receive the same physical scenario adjustment. These are qualitative sensitivity experiments.');text('timings','Hourly average PM₂.₅ concentration');if(point)await profile();}catch(e){status(e.message,true);}}
function download(canvas,name){let a=document.createElement('a');a.href=canvas.toDataURL('image/png');a.download=name;a.click();}
function exportMaps(){let c=document.createElement('canvas');c.width=2400;c.height=1000;let x=c.getContext('2d');x.fillStyle='#fff';x.fillRect(0,0,2400,1000);x.fillStyle='#213b43';x.font='28px Arial';x.fillText(`Camden PM2.5 | ${meta.reference_case} | ${grid.height} m | µg/m³`,30,45);x.font='20px Arial';x.fillText($('applied').textContent,30,80);['first','second','difference'].forEach((k,i)=>{x.fillText($(k+'-title').textContent,20+i*800,120);x.drawImage($(k+'-map'),i*800,140,800,800);});x.font='18px Arial';x.fillText($('colour-note').textContent,25,975);download(c,`pm25-${comparison?'comparison':'scenario'}-${grid.height}m.png`);}
async function run(){if(comparison)return;try{$('run').disabled=true;let p=Object.fromEntries(['emissions','background','wind','rotation'].map(k=>[k,+$(k).value]));await api('/api/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(p)});status('Predicting and saving the scenario…');await waitForReady(true);}catch(e){status(e.message,true);$('run').disabled=false;}}
async function waitForReady(changed=false){for(;;){let s=await api('/api/status');status(s.phase);if(s.ready&&!s.busy){if(s.error)throw Error(s.error);if(changed)current=s.current;break;}if(s.error)throw Error(s.error);await new Promise(r=>setTimeout(r,800));}if(!meta){meta=await api('/api/meta');text('case-name',meta.case_simulation);text('case-time',meta.reference_case+' · hourly PALM mean');text('compute-device','Precomputed PALM comparison');text('domain-scope',meta.statistics_scope);text('provenance',JSON.stringify(meta,null,2));text('provenance-summary',`Checkpoint SHA-256: ${meta.checkpoint_sha256}`);text('validation-note','This case is excluded from training by the checkpoint’s configured held-out hour window. App/evaluator parity and raw-versus-compact PALM matching passed. One case does not reproduce the full test-set benchmark.');text('input-copy','The heights are fixed model coordinates. Select a location to see both concentration profiles and the signed error by height.');$('height').replaceChildren(...meta.heights.map((h,i)=>{let o=document.createElement('option');o.value=i;o.textContent=h+' m';return o;}));$('height').value=level;$('height').disabled=false;for(let i=1;i<=3;i++){text('x-left-'+i,fmt(meta.x[0]/1000,2));text('x-right-'+i,fmt(meta.x.at(-1)/1000,2));}text('northing-range',`Northing ${fmt(meta.y[0]/1000,2)}–${fmt(meta.y.at(-1)/1000,2)} km`);}await loadGrid();$('run').disabled=false;text('pending','Ready. Edited controls take effect after Run prediction.');}
$('scenario-controls').hidden=comparison;$('comparison-guide').hidden=!comparison;
if(comparison){text('page-title','Compare PM2.5 with PALM.');text('page-lede','A verified held-out hourly case: physical simulation, surrogate prediction and error on the same grid.');for(let [id,value]of Object.entries({'first-title':'PALM','second-title':'Surrogate','difference-title':'Error','first-tag':'Physical reference','second-tag':'Unchanged inputs','difference-tag':'Surrogate − PALM','first-key':'PALM','second-key':'Surrogate','metric-label-1':'PALM mean','metric-label-2':'Surrogate mean','metric-label-3':'Mean bias','metric-label-4':'RMSE','difference-profile-title':'Error · µg/m³','metric-unit-3':'surrogate − PALM · µg/m³','metric-unit-4':'µg/m³','performance-note':'The scatter plot compares matched cells. RMSE and bias use all valid Camden atmospheric cells at this height. Full-domain verification is recorded separately.'}))text(id,value);}
for(let k of ['emissions','background','wind','rotation'])$(k).oninput=controls;
$('height').onchange=()=>{level=+$('height').value;loadGrid();};$('contrast').onchange=drawAll;$('run').onclick=run;$('reset').onclick=()=>{for(let [k,v]of Object.entries({emissions:1,background:0,wind:1,rotation:0}))$(k).value=v;controls();};document.querySelectorAll('[data-preset]').forEach(b=>b.onclick=()=>{$('reset').click();if(b.dataset.preset==='emissions')$('emissions').value=.8;else $('rotation').value=30;controls();});$('enlarge').onclick=()=>{$('maps').classList.toggle('focus-difference');$('enlarge').setAttribute('aria-pressed',$('maps').classList.contains('focus-difference'));};$('figure-download').onclick=exportMaps;$('profile-download').onclick=()=>{let c=document.createElement('canvas');c.width=1440;c.height=620;let x=c.getContext('2d');x.fillStyle='white';x.fillRect(0,0,1440,620);x.fillStyle='#234';x.font='18px Arial';x.fillText(meta.reference_case+' | '+$('point-label').textContent,20,25);x.fillText($('applied').textContent,20,55);x.drawImage($('profile'),0,80);x.drawImage($('difference-profile'),720,80);download(c,'pm25-profiles.png');};
for(let key of ['first','second','difference']){let c=$(key+'-map');function cell(e){let r=c.getBoundingClientRect();return {x:Math.min(799,Math.max(0,Math.floor((e.clientX-r.left)/r.width*800))),y:799-Math.min(799,Math.max(0,Math.floor((e.clientY-r.top)/r.height*800))) };}c.onmousemove=e=>{if(!grid)return;let p=cell(e),i=p.y*800+p.x;text('inspect-value',`${comparison?'PALM':'Baseline'} ${fmt(grid.first[i])} · ${comparison?'Surrogate':'Scenario'} ${fmt(grid.second[i])} · ${comparison?'Error':'Change'} ${fmt(grid.difference[i])} µg/m³`);text('inspect-location',`E ${fmt(meta.x[p.x],1)} m · N ${fmt(meta.y[p.y],1)} m`);};c.onclick=e=>{point=cell(e);drawAll();profile().catch(e=>status(e.message,true));};}
controls();waitForReady().catch(e=>status(e.message,true));
