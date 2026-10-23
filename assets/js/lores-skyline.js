/* Exact residual_skyline data from the paper, drawn in the browser.
 * See assets/data/lores/skyline-provenance.json and skyline-source.txt.
 * Every mesh vertex is retained; the two panels share geometry and color scales.
 */
(() => {
  "use strict";
  const root = document.querySelector("#lr-skyline");
  if (!root) return;
  const rows = 64, columns = 2048, peak = 18.875, zLimit = peak * 1.02;
  const view = { width: 400, height: 350, x: 172, y: 160, scale: 155, distance: 10 };
  const elevation = 32 * Math.PI / 180, azimuth = -58 * Math.PI / 180;
  const right = [-Math.sin(azimuth), Math.cos(azimuth), 0];
  const up = [-Math.sin(elevation) * Math.cos(azimuth), -Math.sin(elevation) * Math.sin(azimuth), Math.cos(elevation)];
  const eye = [Math.cos(elevation) * Math.cos(azimuth), Math.cos(elevation) * Math.sin(azimuth), Math.sin(elevation)];
  const dot = (a, b) => a.reduce((sum, n, i) => sum + n * b[i], 0);
  const world = (channel, token, value) => [(channel / 2048 - 0.5) * 1.5, token / 128 - 0.5, (value / zLimit - 0.5) * 0.95];
  const project = (channel, token, value) => {
    const p = world(channel, token, value), depth = dot(p, eye);
    const scale = view.scale * view.distance / (view.distance - depth);
    return [view.x + scale * dot(p, right), view.y - scale * dot(p, up), depth];
  };

  // Matplotlib's 256-entry LinearSegmentedColormap with PowerNorm(gamma=0.4).
  const stops = [[4, 82, 117], [8, 144, 153], [124, 203, 162], [252, 222, 156]];
  const palette = Array.from({ length: 256 }, (_, i) => {
    const position = i / 255 * 3, segment = Math.min(2, Math.floor(position));
    return stops[segment].map((v, j) => v + (stops[segment + 1][j] - v) * (position - segment));
  });
  const colorIndex = value => Math.min(255, Math.floor(Math.pow(value / peak, 0.4) * 256));
  const colors = palette.map(rgb => `rgb(${rgb.join(",")})`);

  function axes(plot) {
    const svgNS = "http://www.w3.org/2000/svg";
    const grid = plot.querySelector(".lr-skyline-grid"), overlay = plot.querySelector(".lr-skyline-axes");
    function path(target, points) {
      const element = document.createElementNS(svgNS, "path");
      element.setAttribute("d", points.map((p, i) => `${i ? "L" : "M"}${project(...p).slice(0, 2).join(",")}`).join(" "));
      target.append(element);
    }
    function text(label, point, dx, dy, className = "", rotate = 0) {
      const [x, y] = project(...point), element = document.createElementNS(svgNS, "text");
      element.setAttribute("x", x + dx); element.setAttribute("y", y + dy);
      element.setAttribute("text-anchor", "middle");
      if (className) element.setAttribute("class", className);
      if (rotate) element.setAttribute("transform", `rotate(${rotate} ${x + dx} ${y + dy})`);
      element.textContent = label; overlay.append(element);
    }
    for (const z of [0, 5, 10, 15]) {
      path(grid, [[0, 0, z], [0, 128, z], [2048, 128, z]]);
      text(z, [2048, 128, z], 16, 4);
    }
    for (const x of [0, 1024, 2048]) {
      path(grid, [[x, 0, 0], [x, 128, 0], [x, 128, zLimit]]);
      text(x, [x, 0, 0], 0, 20);
    }
    for (const y of [0, 64, 128]) {
      path(grid, [[2048, y, 0], [0, y, 0], [0, y, zLimit]]);
      // The shared corner already has the channel's 2048 tick.
      if (y) text(y, [2048, y, 0], 19, 19);
    }
    path(overlay, [[0, 0, 0], [2048, 0, 0], [2048, 128, 0], [2048, 128, zLimit]]);
    text("Channel", [1024, 0, 0], -6, 46, "lr-skyline-label", 18);
    text("Token", [2048, 64, 0], 44, 29, "lr-skyline-label", -38);
    text("|x|", [2048, 128, zLimit / 2], 44, 0, "lr-skyline-label");
  }

  function webglRenderer(canvas, values) {
    const gl = canvas.getContext("webgl2", { alpha: true, antialias: true, preserveDrawingBuffer: true });
    if (!gl) return null;
    function shader(type, source) {
      const result = gl.createShader(type);
      gl.shaderSource(result, source); gl.compileShader(result);
      if (!gl.getShaderParameter(result, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(result));
      return result;
    }
    const vertex = shader(gl.VERTEX_SHADER, `#version 300 es
      precision highp float;
      uniform highp sampler2D heights;
      uniform vec3 right, up, eye;
      flat out float magnitude;
      void main() {
        int cell = gl_VertexID / 6, corner = gl_VertexID % 6;
        ivec2 base = ivec2(cell % ${columns - 1}, cell / ${columns - 1});
        // Two triangles per cell, matching the four original mesh vertices.
        ivec2 offsets[6] = ivec2[6](ivec2(0,0),ivec2(1,0),ivec2(1,1),ivec2(0,0),ivec2(1,1),ivec2(0,1));
        ivec2 coord = base + offsets[corner];
        float height = texelFetch(heights, coord, 0).r;
        vec3 p = vec3((float(coord.x)/2048.0-0.5)*1.5, float(coord.y)*2.0/128.0-0.5, (height/${zLimit}-0.5)*0.95);
        float w = ${view.distance}.0 - dot(p, eye);
        gl_Position = vec4(
          ${2 * view.x / view.width - 1}*w + ${2 * view.scale * view.distance / view.width}*dot(p,right),
          ${1 - 2 * view.y / view.height}*w + ${2 * view.scale * view.distance / view.height}*dot(p,up),
          5.0*w-48.0, w);
        magnitude = texelFetch(heights, base, 0).r;
      }`);
    const fragment = shader(gl.FRAGMENT_SHADER, `#version 300 es
      precision highp float;
      flat in float magnitude;
      out vec4 color;
      void main() {
        float index = min(255.0, floor(pow(magnitude/${peak}, 0.4)*256.0));
        float p = index/255.0*3.0;
        vec3 a = vec3(4,82,117), b = vec3(8,144,153), c = vec3(124,203,162), d = vec3(252,222,156);
        vec3 rgb = p<1.0 ? mix(a,b,p) : p<2.0 ? mix(b,c,p-1.0) : mix(c,d,p-2.0);
        color = vec4(rgb/255.0,1.0);
      }`);
    const program = gl.createProgram();
    gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(program));
    gl.deleteShader(vertex); gl.deleteShader(fragment); gl.useProgram(program);
    const texture = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.R32F, columns, rows, 0, gl.RED, gl.FLOAT, values);
    gl.uniform1i(gl.getUniformLocation(program, "heights"), 0);
    for (const [name, vector] of Object.entries({ right, up, eye })) gl.uniform3fv(gl.getUniformLocation(program, name), vector);
    gl.enable(gl.DEPTH_TEST); gl.clearColor(0, 0, 0, 0);
    return () => {
      gl.viewport(0, 0, canvas.width, canvas.height);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.drawArrays(gl.TRIANGLES, 0, (rows - 1) * (columns - 1) * 6);
    };
  }

  // Same full mesh for browsers without WebGL2, sorted from back to front.
  function canvasRenderer(canvas, values) {
    const ctx = canvas.getContext("2d");
    const points = Array.from(values, (v, i) => project(i % columns, Math.floor(i / columns) * 2, v));
    const cells = [];
    for (let row = 0; row < rows - 1; row++) for (let col = 0; col < columns - 1; col++) {
      const i = row * columns + col, corners = [i, i + 1, i + columns + 1, i + columns];
      cells.push({ corners, depth: corners.reduce((sum, j) => sum + points[j][2], 0), color: colors[colorIndex(values[i])] });
    }
    cells.sort((a, b) => a.depth - b.depth);
    return () => {
      ctx.setTransform(canvas.width / view.width, 0, 0, canvas.height / view.height, 0, 0);
      ctx.clearRect(0, 0, view.width, view.height);
      for (const cell of cells) {
        ctx.beginPath();
        cell.corners.forEach((j, k) => k ? ctx.lineTo(points[j][0], points[j][1]) : ctx.moveTo(points[j][0], points[j][1]));
        ctx.closePath(); ctx.fillStyle = cell.color; ctx.fill();
      }
    };
  }

  async function load() {
    const status = root.querySelector("#lr-skyline-status");
    try {
      const response = await fetch(root.dataset.source);
      if (!response.ok) throw new Error(`Skyline data: HTTP ${response.status}`);
      const buffer = await response.arrayBuffer();
      if (buffer.byteLength !== 2 * rows * columns * 4) throw new Error("Unexpected skyline data length");
      const source = new DataView(buffer), values = new Float32Array(2 * rows * columns);
      for (let i = 0; i < values.length; i++) {
        values[i] = source.getFloat32(i * 4, true);
        if (!Number.isFinite(values[i]) || values[i] < 0 || values[i] > peak) throw new Error("Invalid skyline value");
      }
      const plots = [...root.querySelectorAll(".lr-skyline-plot")];
      plots.forEach((plot, i) => {
        axes(plot);
        let canvas = plot.querySelector("canvas"), draw;
        const data = values.subarray(i * rows * columns, (i + 1) * rows * columns);
        try { draw = webglRenderer(canvas, data); } catch (error) { console.warn("Skyline: using Canvas fallback", error); }
        if (!draw) {
          // A canvas that acquired a WebGL context cannot subsequently use 2D.
          const replacement = canvas.cloneNode(); canvas.replaceWith(replacement); canvas = replacement;
          draw = canvasRenderer(canvas, data);
        }
        let pending;
        const resize = () => {
          cancelAnimationFrame(pending);
          pending = requestAnimationFrame(() => {
            const pixels = Math.min(window.devicePixelRatio || 1, 3);
            const width = Math.round(plot.clientWidth * pixels), height = Math.round(width * view.height / view.width);
            if (!width || canvas.width === width && canvas.height === height) return;
            canvas.width = width; canvas.height = height; draw();
          });
        };
        new ResizeObserver(resize).observe(plot);
        window.addEventListener("resize", resize, { passive: true });
        canvas.addEventListener("webglcontextlost", event => event.preventDefault());
        canvas.addEventListener("webglcontextrestored", () => { draw = webglRenderer(canvas, data); draw(); });
        resize();
      });
      status.hidden = true;
    } catch (error) {
      status.textContent = "The activation data could not be loaded. Reload to try again.";
      console.error(error);
    }
  }

  const observer = new IntersectionObserver(entries => {
    if (entries.some(entry => entry.isIntersecting)) { observer.disconnect(); load(); }
  }, { rootMargin: "500px" });
  observer.observe(root);
})();
