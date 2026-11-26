/* Optional reference-input viewer; it does not change the prediction inputs. */
(() => {
  const panel=document.getElementById('input-figures'),host=document.getElementById('input-plots');
  let data,emission,loading;
  const boroughOutline=[[[498.984,83.409],[512.553,134.77],[511.701,181.87],[539.361,235.073],[552.233,243.043],[561.055,256.712],[583.023,277.366],[589.897,298.606],[611.805,321.969],[623.022,381.779],[646.708,436.345],[647.369,506.392],[666.213,503.749],[693.139,514.778],[697.56,538.431],[715.509,552.98],[706.636,563.192],[711.125,573.595],[720.672,583.157],[743.88,589.958],[749.222,600.2],[757.773,598.903],[770.689,638.949],[753.372,648.175],[711.711,643.777],[728.062,675.88],[712.543,681.97],[706.168,668.75],[673.603,686.548],[657.228,682.889],[655.428,685.758],[649.814,680.06],[619.676,708.256],[605.318,699.754],[594.25,659.875],[582.336,664.729],[582.976,659.772],[575.002,653.52],[571.752,655.272],[563.024,640.112],[559.001,642.933],[555.534,637.771],[550.871,640.901],[508.645,586.548],[503.673,586.165],[504.216,578.365],[499.753,578.361],[499.989,585.323],[492.275,585.745],[488.001,571.503],[478.636,574.323],[436.555,436.563],[390.647,450.751],[383.134,440.27],[352.334,457.889],[320.007,427.443],[310.324,436.757],[302.815,429.984],[292.326,438.674],[287.637,431.945],[288.803,418.435],[256.793,415.439],[241.109,420.714],[218.075,434.044],[193.587,470.977],[187.458,475.952],[167.936,456.831],[103.591,373.717],[56.913,307.327],[10.098,250.425],[33.771,232.484],[42.182,243.692],[61.885,235.367],[63.491,247.698],[107.673,236.916],[106.463,227.093],[114.323,207.067],[116.996,203.974],[119.919,206.84],[126.014,202.948],[124.779,186.34],[137.999,188.39],[155.293,176.025],[166.234,172.261],[170.411,140.784],[175.39,131.636],[189.698,124],[196.779,115.07],[212.775,108.821],[215.276,105.428],[213.251,103.373],[232.636,86.782],[231.389,83.826],[249.888,79.538],[251.009,73.994],[277.279,87.32],[285.78,62.981],[298.227,51.611],[325.409,44.87],[344.575,49.534],[372.654,45.726],[391.85,53.348],[444.583,57.548],[454.901,64.164],[484.944,72.879],[498.984,83.409]]];
  const labels=['Eastward wind u','Northward wind v','Potential temperature θ','Specific humidity qᵥ','Incoming PM₂.₅'];
  const units=['m s⁻¹','m s⁻¹','K','kg kg⁻¹','µg m⁻³'];
  const text=(x,y,s)=>`<text x="${x}" y="${y}" font-size="14" fill="#234">${s}</text>`;
  function profile(values,selected,title,unit){
    let lo=Math.min(...values,...selected),hi=Math.max(...values,...selected),pad=Math.max((hi-lo)*.1,Math.abs(lo)*.001,.00001);lo-=pad;hi+=pad;
    const x=v=>58+(v-lo)/(hi-lo)*240,y=z=>230-(z-15)/180*185;
    const points=v=>v.map((n,i)=>`${x(n)},${y(data.heights[i])}`).join(' ');
    let axes=text(8,22,'Height (m)');
    for(const z of [15,55,115,195])axes+=text(8,y(z)+4,z)+`<path d="M58 ${y(z)}H310" stroke="#ccd8cf"/>`;
    for(const v of [lo,(lo+hi)/2,hi])axes+=text(x(v)-13,252,Math.abs(v)>=1?v.toFixed(1):v.toPrecision(2));
    return `<figure><figcaption>${title}</figcaption><svg viewBox="0 0 340 285" role="img" aria-label="${title}: reference and selected profiles at 12 input heights">${axes}${text(120,276,unit)}<polyline points="${points(values)}" fill="none" stroke="#586965" stroke-width="5"/><polyline points="${points(selected)}" fill="none" stroke="#c35b2f" stroke-width="2" stroke-dasharray="5 3"/></svg></figure>`;
  }
  function settings(){
    if(document.getElementById('comparison-guide')&&!document.getElementById('comparison-guide').hidden)return [1,0,1,0];
    return ['emissions','background','wind','rotation'].map(id=>Number(document.getElementById(id).value));
  }
  function map(id,hour,factor){
    const c=document.getElementById(id),ctx=c.getContext('2d'),im=ctx.createImageData(800,800);
    const colours=[[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
    for(let row=0;row<800;row++)for(let col=0;col<800;col++){
      const value=emission[hour*640000+(799-row)*800+col]*factor;
      const t=Math.min(1,Math.max(0,Math.log1p(value/data.emissionPeak*100)/Math.log1p(200)))*4;
      const k=Math.min(3,Math.floor(t)),a=t-k,offset=(row*800+col)*4;
      for(let channel=0;channel<3;channel++)im.data[offset+channel]=Math.round(colours[k][channel]*(1-a)+colours[k+1][channel]*a);
      im.data[offset+3]=255;
    }
    ctx.putImageData(im,0,0);ctx.beginPath();for(const ring of boroughOutline){ring.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.closePath();}ctx.strokeStyle="#ffffff";ctx.lineWidth=5;ctx.stroke();ctx.strokeStyle="#172f2a";ctx.lineWidth=2.5;ctx.stroke();
  }
  function render(){
    if(!panel.open||!data||!emission)return;
    const [scale,shift,wind,rotation]=settings();if(![scale,shift,wind,rotation].every(Number.isFinite))return;
    const hour=Number(document.getElementById('input-hour').value),p=data.dynamic[hour],a=rotation*Math.PI/180;
    const selected=[p[0].map((u,i)=>wind*(u*Math.cos(a)+p[1][i]*Math.sin(a))),p[1].map((v,i)=>wind*(-p[0][i]*Math.sin(a)+v*Math.cos(a))),p[2],p[3],p[4].map(v=>v+shift)];
    host.innerHTML=`<h3>Horizontal emissions · ${data.hours[hour]}</h3><div class="emission-pair"><figure><figcaption>Reference · 100%</figcaption><canvas id="reference-emissions" width="800" height="800" role="img" aria-label="Reference emissions across the horizontal model domain"></canvas></figure><figure><figcaption>Selected · ${Math.round(scale*100)}%</figcaption><canvas id="selected-emissions" width="800" height="800" role="img" aria-label="Selected emissions on the same colour scale"></canvas></figure></div><div class="input-gradient"></div><p>Shared colour scale: 0–200% of the reference-domain peak across both input hours, with logarithmic colour spacing. The maps show the actual spatially varying input fields across the full 8 × 8 km model domain, north upward. The outlined borough is Camden; surrounding emissions remain visible. These horizontal input fields are independent of the selected viewing height. Scaling applies to every cell in both hours.</p><h3>Vertical forcing profiles</h3><p>Solid grey: reference. Dashed orange: selected. These are the 12 input levels consumed by the surrogate. The same wind and incoming-concentration adjustments apply to both input hours; temperature and humidity remain unchanged.</p><div class="forcing-profiles">${p.map((v,i)=>profile(v,selected[i],labels[i],units[i])).join('')}</div>`;
    map('reference-emissions',hour,1);map('selected-emissions',hour,scale);
  }
  async function load(){
    if(data&&emission)return render();if(loading)return;
    host.textContent='Loading reference input fields…';loading=true;
    try{
      const r=await fetch('./reference-inputs.json');if(!r.ok)throw Error();data=await r.json();
      const e=await fetch('./'+data.emissionsFile);if(!e.ok)throw Error();
      emission=new Float32Array(await new Response(e.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
      if(emission.length!==2*800*800||data.heights.length!==12)throw Error();render();
    }catch{data=null;host.textContent='Could not load the input figures. Close and reopen this panel to retry.';}finally{loading=false;}
  }
  let queued=false;function update(){if(!panel.open||!data||queued)return;queued=true;requestAnimationFrame(()=>{queued=false;render();});}
  panel.addEventListener('toggle',()=>{if(panel.open)load();});
  document.addEventListener('input',update);document.addEventListener('change',update);
  for(const id of ['pending','applied']){const e=document.getElementById(id);if(e)new MutationObserver(update).observe(e,{childList:true,characterData:true,subtree:true});}
  document.getElementById('reset')?.addEventListener('click',update);
})();
