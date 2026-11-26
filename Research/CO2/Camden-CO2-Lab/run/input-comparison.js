/* Display the archived physical inputs, independently of prediction loading. */
(() => {
  const host = document.getElementById('input-plots');
  const ids = ['emissions', 'wind-speed', 'rotation', 'background'];
  let reference, emissions, ceiling;
  const colours = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  const text = (x,y,value,size=14) => `<text x="${x}" y="${y}" fill="#192d2d" font-size="${size}">${value}</text>`;
  const line = (x1,y1,x2,y2) => `<line x1="${x1}" y1="${y1}" x2="${x2}" y2="${y2}" stroke="#a8b8ad"/>`;
  const number = x => Math.abs(x)>=100 ? x.toFixed(0) : Math.abs(x)>=1 ? x.toFixed(1) : x.toFixed(3);
  function profile(title, unit, original, selected) {
    let low=Math.min(...original,...selected),high=Math.max(...original,...selected);
    const pad=Math.max((high-low)*.1,Math.abs(low)*.001,.001);low-=pad;high+=pad;
    const x=v=>66+(v-low)/(high-low)*248;
    const ymin=reference.heights[0],ymax=reference.heights.at(-1);
    const y=z=>252-(z-ymin)/(ymax-ymin)*205;
    const points=values=>values.map((v,i)=>`${x(v)},${y(reference.heights[i])}`).join(' ');
    let content=text(12,22,'Height (m)',14)+line(66,47,66,252)+line(66,252,314,252);
    for(const z of [ymin,1000,2000,ymax])content+=text(10,y(z)+5,z)+line(62,y(z),314,y(z));
    for(const v of [low,(low+high)/2,high])content+=text(x(v)-16,275,number(v),12);
    content+=text(115,298,unit,13)+`<polyline points="${points(original)}" fill="none" stroke="#586965" stroke-width="5"/><polyline points="${points(selected)}" fill="none" stroke="#c35b2f" stroke-width="2" stroke-dasharray="5 3"/>`;
    return `<figure><figcaption>${title}</figcaption><svg viewBox="0 0 340 310" role="img" aria-label="${title}: reference and selected vertical profiles across 60 heights">${content}</svg></figure>`;
  }
  function drawMap(id,factor) {
    const canvas=document.getElementById(id),ctx=canvas.getContext('2d');
    const image=ctx.createImageData(800,800);
    for(let row=0;row<800;row++)for(let col=0;col<800;col++){
      const value=emissions[(799-row)*800+col]*factor;
      const t=Math.min(1,Math.max(0,Math.log1p(value)/Math.log1p(ceiling)))*4;
      const k=Math.min(3,Math.floor(t)),a=t-k,j=(row*800+col)*4;
      for(let c=0;c<3;c++)image.data[j+c]=Math.round(colours[k][c]*(1-a)+colours[k+1][c]*a);
      image.data[j+3]=255;
    }
    ctx.putImageData(image,0,0);
    canvas.addEventListener('pointermove',event=>{
      const b=canvas.getBoundingClientRect(),x=Math.min(799,Math.max(0,Math.floor((event.clientX-b.left)*800/b.width))),y=799-Math.min(799,Math.max(0,Math.floor((event.clientY-b.top)*800/b.height)));
      document.getElementById('emission-readout').textContent=`Cell ${x}, ${y} (10 m grid): reference ${emissions[y*800+x].toPrecision(5)}, selected ${(emissions[y*800+x]*factor).toPrecision(5)} kg m⁻² h⁻¹.`;
    });
  }
  function render() {
    if(!reference||!emissions)return;
    const values=ids.map(id=>Number(document.getElementById(id+'-value').value));
    if(ids.some(id=>document.getElementById(id+'-value').value==='')||!values.every(Number.isFinite)){
      host.textContent='Choose valid input values to compare with the reference.';return;
    }
    const [scale,wind,rotation,shift]=values;
    if(scale<50||scale>150||Math.abs(wind)>200||Math.abs(rotation)>180||Math.abs(shift)>5){host.textContent='Choose values within the available scenario ranges.';return;}
    const p=reference.profiles.map(channel=>channel.values),a=rotation*Math.PI/180;
    const u=p[0].map((v,i)=>wind/100*(v*Math.cos(a)+p[1][i]*Math.sin(a)));
    const v=p[1].map((v,i)=>wind/100*(-p[0][i]*Math.sin(a)+v*Math.cos(a)));
    const selected=[u,v,p[2],p[3],p[4],p[5].map(v=>v+shift)];
    const titles=['Eastward wind u','Northward wind v','Vertical wind w','Potential temperature θ','Specific humidity qᵥ','Boundary CO₂'];
    const units=['m s⁻¹','m s⁻¹','m s⁻¹','K','kg kg⁻¹','ppm'];
    host.innerHTML=`<section class="emission-inputs"><h3>Horizontal emissions field</h3><div class="emission-pair"><figure><figcaption>Reference · 100%</figcaption><canvas id="reference-emissions" width="800" height="800" role="img" aria-label="Reference horizontal emissions on the 800 by 800 grid"></canvas></figure><figure><figcaption>Selected · ${scale}%</figcaption><canvas id="selected-emissions" width="800" height="800" role="img" aria-label="Selected horizontal emissions on the same grid and colour scale"></canvas></figure></div><div class="emission-gradient"></div><p>Shared colour scale: 0 to ${ceiling.toPrecision(4)} kg m⁻² h⁻¹, clipped at the reference 99th percentile × 1.5; logarithmic colour spacing. Both maps cover the full 8 × 8 km domain, north upward, on the 10 m model input grid. The selected factor multiplies every cell.</p><p id="emission-readout">Move over either map to read the original and selected emission values.</p></section><section class="forcing-inputs"><h3>Vertical atmospheric forcing profiles · 60 levels</h3><p>Solid grey: reference. Dashed orange: selected. These are the actual input profiles at 25–2,975 m, not the 16 output heights. ERA5 supplies meteorological forcing; CAMS EGG4 supplies CO₂. Wind scaling and rotation modify u and v at every level; the CO₂ shift applies at every level. Other profiles remain unchanged.</p><div class="forcing-profiles">${p.map((values,i)=>profile(titles[i],units[i],values,selected[i])).join('')}</div></section>`;
    drawMap('reference-emissions',1);drawMap('selected-emissions',scale/100);
    // Show a consistent paired readout when inspecting either map.
    document.getElementById('reference-emissions').addEventListener('pointermove',event=>{
      const b=event.target.getBoundingClientRect(),x=Math.min(799,Math.max(0,Math.floor((event.clientX-b.left)*800/b.width))),y=799-Math.min(799,Math.max(0,Math.floor((event.clientY-b.top)*800/b.height)));
      const value=emissions[y*800+x];document.getElementById('emission-readout').textContent=`Cell ${x}, ${y} (10 m grid): reference ${value.toPrecision(5)}, selected ${(value*scale/100).toPrecision(5)} kg m⁻² h⁻¹.`;
    });
  }
  let queued=false;
  function schedule(){if(queued)return;queued=true;requestAnimationFrame(()=>{queued=false;render();});}
  document.addEventListener('input',schedule);document.addEventListener('change',schedule);
  new MutationObserver(schedule).observe(document.getElementById('pending'),{childList:true,subtree:true,characterData:true});
  document.getElementById('reset').addEventListener('click',schedule);
  host.textContent='Loading reference vertical profiles and horizontal emissions…';
  (async()=>{
    const r=await fetch('reference-inputs.json?v=co2-20261126-3');if(!r.ok)throw Error('Profile unavailable');reference=await r.json();
    const e=await fetch(reference.emissions.file);if(!e.ok)throw Error('Emissions unavailable');
    emissions=new Float32Array(await new Response(e.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
    if(emissions.length!==800*800||reference.heights.length!==60)throw Error('Unexpected input dimensions');
    const sorted=emissions.slice().sort();ceiling=sorted[Math.floor(sorted.length*.99)]*1.5||reference.emissions.max*1.5;
    render();
  })().catch(()=>{host.textContent='Could not load the reference input fields. Reload to try again.';});
})();
