(() => {
  const panel = document.getElementById('input-figures');
  const host = document.getElementById('reference-input-plots');
  const root = new URL('../Camden-CO2-Lab/run/', document.baseURI);
  const titles = ['Eastward wind u', 'Northward wind v', 'Vertical wind w', 'Potential temperature θ', 'Specific humidity qᵥ', 'Boundary CO₂'];
  const units = ['m s⁻¹', 'm s⁻¹', 'm s⁻¹', 'K', 'kg kg⁻¹', 'ppm'];
  const number = n => Number(n).toPrecision(4).replace(/\.?0+$/, m => m.includes('.') ? '' : m);
  function profile(p, heights, i) {
    let lo = Math.min(...p.values), hi = Math.max(...p.values);
    const pad = (hi - lo) * .08 || Math.max(Math.abs(lo) * .01, .01);
    lo -= pad; hi += pad;
    const x = v => 44 + (v - lo) / (hi - lo) * 172;
    const y = z => 174 - z / 3000 * 140;
    const points = p.values.map((v,j) => `${x(v)},${y(heights[j])}`).join(' ');
    return `<svg viewBox="0 0 240 215" role="img" aria-label="${titles[i]} reference vertical profile, ${units[i]}"><title>${titles[i]} (${units[i]})</title><text x="44" y="15" font-size="12">${titles[i]}</text><path d="M44 34V174H216" fill="none" stroke="#8b9b94"/><text x="3" y="39" font-size="10">3000</text><text x="26" y="177" font-size="10">0</text><text x="3" y="105" font-size="10">m</text><text x="44" y="190" font-size="10">${number(lo)}</text><text x="216" y="190" text-anchor="end" font-size="10">${number(hi)}</text><text x="130" y="208" text-anchor="middle" font-size="10">${units[i]}</text><polyline points="${points}" fill="none" stroke="#1b6257" stroke-width="2"/></svg>`;
  }
  let loaded = false;
  panel.addEventListener('toggle', async () => {
    if (!panel.open || loaded) return;
    loaded = true;
    host.textContent = 'Loading reference inputs…';
    try {
      const response = await fetch(new URL('reference-inputs.json', root));
      if (!response.ok) throw new Error('Metadata unavailable');
      const reference = await response.json();
      const responseMap = await fetch(new URL(reference.emissions.file, root));
      if (!responseMap.ok) throw new Error('Emissions unavailable');
      const values = new Float32Array(await new Response(responseMap.body.pipeThrough(new DecompressionStream('gzip'))).arrayBuffer());
      const {width, height} = reference.emissions;
      if (values.length !== width * height || reference.profiles.length !== 6) throw new Error('Unexpected input dimensions');
      const sorted = values.slice().sort();
      const ceiling = sorted[Math.floor(sorted.length * .99)] * 1.5 || reference.emissions.max;
      host.innerHTML = `<div class="input-layout"><div><h3>Reference emissions</h3><canvas id="baseline-emissions" width="${width}" height="${height}" role="img" aria-label="Archived baseline emissions array"></canvas><div class="emission-scale"></div><p>0–${number(ceiling)} kg m⁻² h⁻¹ (values above this scale are clipped in colour only).</p><p id="emission-value">Hover over a cell for its emission value.</p><p>Archived 800 × 800 input array, resized from a 600 × 600 source at 100 m spacing. Geographic mapping to Camden is unverified; resizing adds no spatial detail.</p></div><div><h3>Vertical atmospheric forcing · 60 levels</h3><p>Reference profiles at 25–2,975 m. Meteorological forcing comes from ERA5; boundary CO₂ comes from CAMS EGG4.</p><div class="profile-grid">${reference.profiles.map((p,i)=>profile(p,reference.heights,i)).join('')}</div></div></div>`;
      const canvas = document.getElementById('baseline-emissions');
      const context = canvas.getContext('2d');
      const pixels = context.createImageData(width,height);
      const colours = [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
      for (let y=0;y<height;y++) for (let x=0;x<width;x++) {
        const value = values[(height-1-y)*width+x];
        const position = Math.min(1,Math.max(0,value/ceiling))*4;
        const a = Math.min(3,Math.floor(position)), t=position-a;
        const at=(y*width+x)*4;
        for(let c=0;c<3;c++) pixels.data[at+c]=colours[a][c]*(1-t)+colours[a+1][c]*t;
        pixels.data[at+3]=255;
      }
      context.putImageData(pixels,0,0);
      canvas.addEventListener('pointermove',event=>{
        const box=canvas.getBoundingClientRect();
        const x=Math.min(width-1,Math.max(0,Math.floor((event.clientX-box.left)/box.width*width)));
        const y=Math.min(height-1,Math.max(0,Math.floor((event.clientY-box.top)/box.height*height)));
        document.getElementById('emission-value').textContent=`Cell ${x+1}, ${height-y}: ${number(values[(height-1-y)*width+x])} kg m⁻² h⁻¹ (resized input array).`;
      });
    } catch (error) {
      loaded=false;
      host.textContent='Could not load reference inputs. Close and reopen this panel to retry.';
    }
  });
})();
