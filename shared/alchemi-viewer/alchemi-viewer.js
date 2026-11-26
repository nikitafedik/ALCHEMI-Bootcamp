/* ALCHEMI viewer for the course notebooks (three.js 0.186.1, window.THREE).
 * One plain script, no imports, no network. Defines window.AlchemiViewer:
 *   mountStructure(node, structure, options)  one structure, rotatable (W1 card view)
 *   mountPlayer(root, items, options)         trajectories side by side, one play control (W2)
 * Units: positions in Angstrom, forces in eV/Angstrom.
 *
 * Speed and WebGL contexts (browsers keep at most about 16 per page):
 * - Structure views share one renderer per page. It draws a view, copies the picture into the
 *   view's own 2D canvas and is released when no view is being dragged (idle_release, default on).
 *   A page of structure views therefore holds at most one live context, and none when idle.
 * - A player keeps one renderer for all its cards: one canvas over the card grid, one viewport per
 *   card, no copying.
 * - Phong shading, a sky light and two camera lights: shaders compile in tens of milliseconds, not a second.
 * - Spheres and bonds darken softly toward their edges, like a studio render, so white hydrogen stays
 *   visible on the light panel. It is part of the same shader: no extra draws.
 * - Views are built when they scroll into sight, a few per frame, so the page never freezes.
 */
(function () {
  "use strict";
  const THREE = window.THREE;
  const RADII = {"covalent":{"H":0.31,"He":0.28,"Li":1.28,"Be":0.96,"B":0.84,"C":0.76,"N":0.71,"O":0.66,"F":0.57,"Ne":0.58,"Na":1.66,"Mg":1.41,"Al":1.21,"Si":1.11,"P":1.07,"S":1.05,"Cl":1.02,"Ar":1.06,"K":2.03,"Ca":1.76,"Sc":1.7,"Ti":1.6,"V":1.53,"Cr":1.39,"Mn":1.39,"Fe":1.32,"Co":1.26,"Ni":1.24,"Cu":1.32,"Zn":1.22,"Ga":1.22,"Ge":1.2,"As":1.19,"Se":1.2,"Br":1.2,"Kr":1.16,"Rb":2.2,"Sr":1.95,"Y":1.9,"Zr":1.75,"Nb":1.64,"Mo":1.54,"Tc":1.47,"Ru":1.46,"Rh":1.42,"Pd":1.39,"Ag":1.45,"Cd":1.44,"In":1.42,"Sn":1.39,"Sb":1.39,"Te":1.38,"I":1.39,"Xe":1.4,"Cs":2.44,"Ba":2.15,"La":2.07,"Ce":2.04,"Pr":2.03,"Nd":2.01,"Pm":1.99,"Sm":1.98,"Eu":1.98,"Gd":1.96,"Tb":1.94,"Dy":1.92,"Ho":1.92,"Er":1.89,"Tm":1.9,"Yb":1.87,"Lu":1.87,"Hf":1.75,"Ta":1.7,"W":1.62,"Re":1.51,"Os":1.44,"Ir":1.41,"Pt":1.36,"Au":1.36,"Hg":1.32,"Tl":1.45,"Pb":1.46,"Bi":1.48,"Po":1.4,"At":1.5,"Rn":1.5,"Fr":2.6,"Ra":2.21,"Ac":2.15,"Th":2.06,"Pa":2.0,"U":1.96,"Np":1.9,"Pu":1.87},"vdw":{"H":1.2,"He":1.4,"Li":1.82,"Be":1.53,"B":1.92,"C":1.7,"N":1.55,"O":1.52,"F":1.47,"Ne":1.54,"Na":2.27,"Mg":1.73,"Al":1.84,"Si":2.1,"P":1.8,"S":1.8,"Cl":1.75,"Ar":1.88,"K":2.75,"Ca":2.31,"Sc":2.0,"Ti":2.0,"V":2.0,"Cr":2.0,"Mn":2.0,"Fe":2.0,"Co":2.0,"Ni":1.63,"Cu":1.4,"Zn":1.39,"Ga":1.87,"Ge":2.11,"As":1.85,"Se":1.9,"Br":1.85,"Kr":2.02,"Rb":3.03,"Sr":2.49,"Y":2.0,"Zr":2.0,"Nb":2.0,"Mo":2.0,"Tc":2.0,"Ru":2.0,"Rh":2.0,"Pd":1.63,"Ag":1.72,"Cd":1.58,"In":1.93,"Sn":2.17,"Sb":2.06,"Te":2.06,"I":1.98,"Xe":2.16,"Cs":3.43,"Ba":2.49,"La":2.0,"Ce":2.0,"Pr":2.0,"Nd":2.0,"Pm":2.0,"Sm":2.0,"Eu":2.0,"Gd":2.0,"Tb":2.0,"Dy":2.0,"Ho":2.0,"Er":2.0,"Tm":2.0,"Yb":2.0,"Lu":2.0,"Hf":2.0,"Ta":2.0,"W":2.0,"Re":2.0,"Os":2.0,"Ir":2.0,"Pt":1.75,"Au":1.66,"Hg":1.55,"Tl":1.96,"Pb":2.02,"Bi":2.07,"Po":1.97,"At":2.02,"Rn":2.2,"Fr":3.48,"Ra":2.83,"Ac":2.0,"Th":2.0,"Pa":2.0,"U":1.86,"Np":2.0,"Pu":2.0}};
  // One element palette across the course (design kit). Metals sit darker than the light panel.
  const ELEMENT = {
    H: "#E8ECEF", Li: "#C39BD3", B: "#E8A87C", C: "#5C6770", N: "#00A3E0", O: "#E05252", F: "#8DC63F",
    Na: "#A08CD8", Mg: "#D8B26E", Al: "#9C8FB3", Si: "#C9A27E", P: "#F0A23B", S: "#E1C542", Cl: "#4CB944",
    K: "#8F72D8", Ca: "#8FB06A", Sr: "#3E9E96", Ba: "#5C8F6E", Ti: "#9AA7B8", Fe: "#C8743F", Co: "#5A7FC8", Ni: "#63A37A", Cu: "#C47B48",
    Zn: "#8294AE", Br: "#A3473A", Ag: "#8E979F", I: "#8B4FA3", Pt: "#7D8893", Au: "#D9A63A",
  };
  const NONMETAL = new Set(["H", "B", "C", "N", "O", "F", "Si", "P", "S", "Cl", "Se", "Br", "I"]);
  const FORCE_RED = new THREE.Color("#D9534F");
  const F_LOW = 1.0, F_HIGH = 20.0;  // eV/Angstrom, log scale: element colour at or below F_LOW, full red at or above F_HIGH
  const THEMES = {
    // Panel behind the structure: light sits a step below the kit wash so white hydrogen keeps its
    // edge; dark is the kit wash. Cell lines use line_ref.
    light: { bg: "#E8EBEE", cell: "#B8B8B8", ground: 0x8A939B },
    dark: { bg: "#252C32", cell: "#5E6A74", ground: 0x1A1F23 },
  };
  // Edge darkening: the share of light lost at the silhouette, and how fast it falls off from there.
  const EDGE = { depth: 0.35, power: 1.6 };
  const DPR = Math.min(window.devicePixelRatio || 1, 2);
  const T0 = performance.now();
  const stats = { views: 0, built: 0, draws: 0, draw_ms: 0, first_ms: null, contexts_made: 0, contexts_live: 0 };

  function themeName(opt) {
    if (opt && (opt.theme === "light" || opt.theme === "dark")) return opt.theme;
    try {
      const flag = window.parent && window.parent.document && window.parent.document.body.getAttribute("data-jp-theme-light");
      if (flag === "true") return "light";
      if (flag === "false") return "dark";
    } catch (e) { /* not same-origin */ }
    return "light";   // course default; dark only when JupyterLab says so or the caller asks
  }

  function forceMix(f) {
    if (!(f > F_LOW)) return 0;
    return Math.min(1, Math.log(f / F_LOW) / Math.log(F_HIGH / F_LOW));
  }

  const tmpM = new THREE.Matrix4(), tmpA = new THREE.Vector3(), tmpB = new THREE.Vector3(), tmpMid = new THREE.Vector3();
  const tmpDir = new THREE.Vector3(), tmpQ = new THREE.Quaternion(), tmpS = new THREE.Vector3(), tmpP = new THREE.Vector3();
  const UP = new THREE.Vector3(0, 1, 0), tmpC = new THREE.Color();

  // Rotate so the two broadest axes face the camera (power iteration on the 3x3 covariance).
  function viewFrame(pos) {
    const n = pos.length, c = [0, 0, 0];
    pos.forEach(p => { c[0] += p[0] / n; c[1] += p[1] / n; c[2] += p[2] / n; });
    const cov = [[0, 0, 0], [0, 0, 0], [0, 0, 0]];
    pos.forEach(p => { const d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]]; for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) cov[i][j] += d[i] * d[j]; });
    const axes = [];
    for (let k = 0; k < 2; k++) {
      let v = k === 0 ? [1, 0.3, 0.2] : [0.2, 1, 0.3];
      for (let it = 0; it < 60; it++) {
        let w = [0, 1, 2].map(i => cov[i][0] * v[0] + cov[i][1] * v[1] + cov[i][2] * v[2]);
        axes.forEach(a => { const d = w[0] * a[0] + w[1] * a[1] + w[2] * a[2]; w = w.map((x, i) => x - d * a[i]); });
        const l = Math.hypot(...w) || 1; v = w.map(x => x / l);
      }
      axes.push(v);
    }
    const a3 = [axes[0][1] * axes[1][2] - axes[0][2] * axes[1][1], axes[0][2] * axes[1][0] - axes[0][0] * axes[1][2], axes[0][0] * axes[1][1] - axes[0][1] * axes[1][0]];
    const M = [axes[0], axes[1], a3];
    return p => { const d = [p[0] - c[0], p[1] - c[1], p[2] - c[2]]; return M.map(a => a[0] * d[0] + a[1] * d[1] + a[2] * d[2]); };
  }

  // Geometry and materials are plain objects; any renderer uploads them on first use.
  const GEO = { sphere: new THREE.SphereGeometry(1, 24, 16), stick: new THREE.CylinderGeometry(1, 1, 1, 12, 1, true) };
  const MAT = {
    atom: new THREE.MeshPhongMaterial({ shininess: 100, specular: 0x8a8a8a }),
    stick: new THREE.MeshPhongMaterial({ shininess: 60, specular: 0x4a4a4a }),
    cell: { light: new THREE.LineBasicMaterial({ color: THEMES.light.cell }), dark: new THREE.LineBasicMaterial({ color: THEMES.dark.cell }) },
  };
  [MAT.atom, MAT.stick].forEach(m => {
    m.onBeforeCompile = sh => {
      sh.fragmentShader = sh.fragmentShader.replace("#include <opaque_fragment>", `
        { float edge = 1.0 - abs(dot(normalize(normal), normalize(vViewPosition)));
          outgoingLight *= 1.0 - ${EDGE.depth.toFixed(2)} * pow(edge, ${EDGE.power.toFixed(2)}); }
        #include <opaque_fragment>`);
    };
    m.customProgramCacheKey = () => "av-edge";
  });

  function makeRenderer(canvas) {
    const r = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true, powerPreference: "high-performance" });
    r.setPixelRatio(DPR);
    r.outputColorSpace = THREE.SRGBColorSpace;
    stats.contexts_made++; stats.contexts_live++;
    return r;
  }
  function dropRenderer(r) {
    r.dispose(); r.forceContextLoss(); stats.contexts_live--;
  }

  /* One structure scene: atoms, bonds, cell, camera and lights. It owns no renderer. */
  class View {
    constructor(sys, opt, seeThrough) {
      const name = themeName(opt), th = THEMES[name];
      this.n = sys.symbols.length;
      // Periodic systems keep the lab frame with z straight up on screen (three.js up is +y);
      // molecules turn their two broadest axes to the camera.
      const periodic = !!sys.periodic || (sys.cell_edges || []).length > 0;
      const toView = periodic ? (p => [p[0], p[2], -p[1]]) : viewFrame(sys.frames[0]);
      this.frames = sys.frames.map(fr => fr.map(toView));
      this.cellEdges = (sys.cell_edges || []).map(e => [toView(e.start), toView(e.end)]);
      this.periodic = periodic;
      const scene = new THREE.Scene();
      // The player draws on a transparent overlay; the rounded card panel behind it is the background.
      scene.background = seeThrough ? null : new THREE.Color(th.bg);
      scene.add(new THREE.HemisphereLight(0xffffff, th.ground, 1.7));
      this.camera = new THREE.PerspectiveCamera(30, 1, 0.05, 4000);
      // Key light high above right (a clear terminator) and a soft fill from the left, both fixed to the camera.
      const key = new THREE.DirectionalLight(0xffffff, 2.6); key.position.set(3, 9, 6); this.camera.add(key);
      const fill = new THREE.DirectionalLight(0xffffff, 0.6); fill.position.set(-6, 2, 3); this.camera.add(fill);
      scene.add(this.camera); this.scene = scene;
      const hasMetal = sys.symbols.some(s => !NONMETAL.has(s));
      this.base = sys.symbols.map(s => new THREE.Color(ELEMENT[s] || "#9AA3AB"));
      this.size = sys.symbols.map(s => 0.25 * (RADII.vdw[s] || 2.0) * (periodic && !NONMETAL.has(s) ? 1.25 : 1));
      const cov = sys.symbols.map(s => RADII.covalent[s] || 0.8);
      const bondable = sys.symbols.map(s => !(periodic && hasMetal && !NONMETAL.has(s)));
      this.atoms = new THREE.InstancedMesh(GEO.sphere, MAT.atom, this.n);
      this.atoms.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      this.atoms.frustumCulled = false;
      scene.add(this.atoms);
      const p0 = this.frames[0];
      this.bonds = [];
      for (let i = 0; i < this.n; i++) for (let j = i + 1; j < this.n; j++) {
        if (!bondable[i] || !bondable[j]) continue;
        const d = Math.hypot(p0[i][0] - p0[j][0], p0[i][1] - p0[j][1], p0[i][2] - p0[j][2]);
        if (d > 0.4 && d < 1.15 * (cov[i] + cov[j])) this.bonds.push([i, j]);
      }
      if (this.bonds.length) {
        this.sticks = new THREE.InstancedMesh(GEO.stick, MAT.stick, 2 * this.bonds.length);
        this.sticks.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
        this.sticks.frustumCulled = false;
        scene.add(this.sticks);
      }
      if (this.cellEdges.length) {
        const g = new THREE.BufferGeometry();
        g.setAttribute("position", new THREE.Float32BufferAttribute(this.cellEdges.flat(2), 3));
        scene.add(new THREE.LineSegments(g, MAT.cell[name]));
      }
      this.forces = sys.atom_forces || null;
      this.frame = -1;
      this.setFrame(0);
      this.fit();
      stats.built++;
    }
    attach(element) {
      this.controls = new THREE.OrbitControls(this.camera, element);
      this.controls.enableDamping = true; this.controls.dampingFactor = 0.12;
      this.controls.target.copy(this.center); this.controls.update();
      this.controls.addEventListener("change", () => { this.dirty = true; });
    }
    fit() {
      // Points to frame, each with its atom radius. Molecules: the whole structure. Periodic: the
      // atoms and the cell's footprint, but not the empty height of a slab's vacuum.
      const p0 = this.frames[0];
      let pts = p0.map((p, i) => [p[0], p[1], p[2], this.size[i]]);
      if (this.cellEdges.length) {
        let lo = Infinity, hi = -Infinity;
        p0.forEach(p => { lo = Math.min(lo, p[1]); hi = Math.max(hi, p[1]); });
        const clamp = p => [p[0], Math.min(hi + 2.5, Math.max(lo - 2.5, p[1])), p[2], 0];
        pts = pts.concat(this.cellEdges.flat().map(clamp));
      }
      this.fitPts = pts;
      this.dir = new THREE.Vector3(0.38, 0.28, 1).normalize();
      // Centre on the box the structure covers on screen from the first view direction.
      const ax = this.axes(this.dir), lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (const p of pts) for (let k = 0; k < 3; k++) {
        const d = ax[k].x * p[0] + ax[k].y * p[1] + ax[k].z * p[2];
        lo[k] = Math.min(lo[k], d - p[3]); hi[k] = Math.max(hi[k], d + p[3]);
      }
      this.center = new THREE.Vector3();
      for (let k = 0; k < 3; k++) this.center.addScaledVector(ax[k], (lo[k] + hi[k]) / 2);
      this.camera.lookAt(this.center);
    }
    // Screen axes for a camera looking back along dir with +y up: right, up, toward the camera.
    axes(dir) {
      const back = dir.clone().normalize(), right = new THREE.Vector3().crossVectors(UP, back).normalize();
      return [right, new THREE.Vector3().crossVectors(back, right), back];
    }
    // Camera distance from the centre that keeps every point and its radius inside the view.
    fitDistance(dir, tanH, tanV) {
      const ax = this.axes(dir), c = this.center;
      let dist = 0;
      for (const p of this.fitPts) {
        const x = p[0] - c.x, y = p[1] - c.y, z = p[2] - c.z;
        const sx = Math.abs(ax[0].x * x + ax[0].y * y + ax[0].z * z) + p[3];
        const sy = Math.abs(ax[1].x * x + ax[1].y * y + ax[1].z * z) + p[3];
        const sz = ax[2].x * x + ax[2].y * y + ax[2].z * z;
        dist = Math.max(dist, Math.max(sx / tanH, sy / tanV) + sz + p[3]);
      }
      return dist;
    }
    setAspect(a) {
      if (a === this.aspect) return;
      this.aspect = a; this.camera.aspect = a; this.camera.updateProjectionMatrix();
      // Fit the structure's outline in the view with a small margin, keeping the view direction.
      const tanV = 0.9 * Math.tan(this.camera.fov * Math.PI / 360), tanH = tanV * a;
      const dir = this.placed ? tmpP.copy(this.camera.position).sub(this.center).normalize().clone() : this.dir;
      const dist = this.fitDistance(dir, tanH, tanV);
      this.camera.position.copy(this.center).addScaledVector(dir, dist);
      this.placed = true;
      if (this.controls) this.controls.update();
      this.dirty = true;
    }
    setFrame(f) {
      f = Math.max(0, Math.min(f, this.frames.length - 1));
      if (f === this.frame) return;
      this.frame = f;
      const pos = this.frames[f], F = this.forces ? this.forces[Math.min(f, this.forces.length - 1)] : null;
      for (let i = 0; i < this.n; i++) {
        const s = this.size[i];
        tmpM.makeScale(s, s, s).setPosition(pos[i][0], pos[i][1], pos[i][2]);
        this.atoms.setMatrixAt(i, tmpM);
        tmpC.copy(this.base[i]); if (F) tmpC.lerp(FORCE_RED, forceMix(F[i]));
        this.atoms.setColorAt(i, tmpC);
      }
      this.atoms.instanceMatrix.needsUpdate = true;
      this.atoms.instanceColor.needsUpdate = true;
      if (this.sticks) {
        for (let b = 0; b < this.bonds.length; b++) {
          const [i, j] = this.bonds[b];
          tmpA.set(pos[i][0], pos[i][1], pos[i][2]); tmpB.set(pos[j][0], pos[j][1], pos[j][2]);
          tmpMid.addVectors(tmpA, tmpB).multiplyScalar(0.5);
          for (const [k, from, a] of [[2 * b, tmpA, i], [2 * b + 1, tmpB, j]]) {
            tmpDir.subVectors(tmpMid, from); const len = tmpDir.length();
            tmpQ.setFromUnitVectors(UP, tmpDir.normalize()); tmpS.set(0.11, len, 0.11);
            tmpP.copy(from).addScaledVector(tmpDir, len / 2);
            tmpM.compose(tmpP, tmpQ, tmpS);
            this.sticks.setMatrixAt(k, tmpM);
            tmpC.copy(this.base[a]); if (F) tmpC.lerp(FORCE_RED, forceMix(F[a]));
            this.sticks.setColorAt(k, tmpC);
          }
        }
        this.sticks.instanceMatrix.needsUpdate = true;
        this.sticks.instanceColor.needsUpdate = true;
      }
      this.dirty = true;
    }
    // Damping keeps the camera moving for a moment after a drag; true while it still moves.
    step() { return this.controls ? this.controls.update() : false; }
  }

  /* ---------- scheduler: one animation loop for everything on the page ---------- */
  const loops = new Set();
  function startLoop(fn) {
    loops.add(fn);
    if (loops.size === 1) requestAnimationFrame(function run(t) { loops.forEach(f => f(t)); requestAnimationFrame(run); });
  }
  const seen = new IntersectionObserver(entries => entries.forEach(e => {
    e.target.__avVisible = e.isIntersecting;
    if (e.isIntersecting && e.target.__avOnShow) e.target.__avOnShow();
  }), { rootMargin: "200px" });
  function el(tag, cls, text) { const n = document.createElement(tag); if (cls) n.className = cls; if (text !== undefined) n.textContent = text; return n; }

  /* ---------- structure views: one shared renderer, pictures kept in 2D canvases ---------- */
  const still = { r: null, canvas: null, lastActive: 0, views: [], active: 0 };
  function stillRenderer() {
    if (!still.r) { still.canvas = document.createElement("canvas"); still.r = makeRenderer(still.canvas); }
    return still.r;
  }
  function stillLoop(t) {
    let budget = 12;  // ms of drawing per frame
    const t0 = performance.now();
    for (const s of still.views) {
      if (!s.host.__avVisible) continue;
      if (!s.view) {
        if (performance.now() - t0 > budget) break;
        s.view = new View(s.sys, s.opt); s.view.attach(s.canvas);
        s.resize();
      }
      const moving = s.view.step();
      if ((s.view.dirty || moving) && s.w) {
        if (performance.now() - t0 > budget && s.drawn) continue;
        s.draw();
      }
    }
    // Release the shared context as soon as every visible view has its picture and nobody is
    // pointing at a view: create, draw, copy and release in one frame, so card iframes on one page
    // never hold contexts at the same time. After an interaction, keep it 400 ms for the next drag.
    if (still.active > 0) still.lastActive = t;
    if (still.r && still.release && still.active === 0 &&
        still.views.every(s => !s.host.__avVisible || (s.view && s.drawn && !s.view.dirty))) {
      if (!still.lastActive || t - still.lastActive > 400) { dropRenderer(still.r); still.r = null; }
    }
  }

  function mountStructure(node, structure, options) {
    const opt = Object.assign({ height: 172, idle_release: true }, options || {});
    node.classList.add("av-structure", `av-${themeName(opt)}`);
    const host = el("div", "av-canvas"); host.style.height = `${opt.height}px`;
    // The canvas takes its size from the card (CSS); its pixel size follows the card times DPR.
    const canvas = el("canvas"); canvas.style.width = "100%"; canvas.style.height = "100%"; canvas.tabIndex = 0; canvas.setAttribute("aria-label", `${structure.name || "Structure"}, drag to rotate`);
    host.appendChild(canvas); node.appendChild(host);
    const ctx = canvas.getContext("2d");
    still.release = still.release === undefined ? !!opt.idle_release : still.release && !!opt.idle_release;
    const s = {
      host, canvas, opt, view: null, w: 0, h: 0, drawn: false,
      sys: Object.assign({}, structure, { frames: [structure.positions] }),
      resize() {
        const w = host.clientWidth, h = host.clientHeight; if (!w || !h) return;
        const cw = Math.round(w * DPR), ch = Math.round(h * DPR);
        s.w = w; s.h = h;
        // A new pixel size clears the 2D canvas, so the view is drawn again.
        if (cw !== canvas.width || ch !== canvas.height) { canvas.width = cw; canvas.height = ch; if (s.view) s.view.dirty = true; }
        if (s.view) s.view.setAspect(w / h);
      },
      draw() {
        const t = performance.now(), r = stillRenderer();
        r.setSize(s.w, s.h, false);
        r.render(s.view.scene, s.view.camera);
        ctx.drawImage(still.canvas, 0, 0, canvas.width, canvas.height);
        s.view.dirty = false; s.drawn = true;
        stats.draws++; stats.draw_ms += performance.now() - t;
        if (stats.first_ms === null) stats.first_ms = performance.now() - T0;
      },
    };
    const on = () => { if (!s.on) { s.on = true; still.active++; } };
    const off = () => { if (s.on) { s.on = false; still.active--; } };
    canvas.addEventListener("pointerenter", on); canvas.addEventListener("pointerleave", off);
    canvas.addEventListener("focus", on); canvas.addEventListener("blur", off);
    new ResizeObserver(() => s.resize()).observe(host);
    // The picture stays in the 2D canvas, so scrolling back into sight needs no redraw.
    seen.observe(host);
    still.views.push(s); stats.views++;
    startLoop(stillLoop);
    return s;
  }

  /* ---------- player: one renderer, one canvas over the grid, one viewport per card ---------- */
  const CHECK = '<svg class="av-done" viewBox="0 0 24 24" width="16" height="16" aria-hidden="true"><circle cx="12" cy="12" r="11" fill="#76B900"/><path d="M7 12.5l3.2 3.2L17 9" fill="none" stroke="#FFFFFF" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round"/></svg>';

  const RADIUS = 8;   // structure panel corners, as .av-canvas in the CSS
  function roundRect(x, y, w, h, r) {
    const f = v => +v.toFixed(1);
    return `M${f(x + r)},${f(y)}H${f(x + w - r)}A${r},${r} 0 0 1 ${f(x + w)},${f(y + r)}V${f(y + h - r)}A${r},${r} 0 0 1 ${f(x + w - r)},${f(y + h)}` +
      `H${f(x + r)}A${r},${r} 0 0 1 ${f(x)},${f(y + h - r)}V${f(y + r)}A${r},${r} 0 0 1 ${f(x + r)},${f(y)}Z`;
  }
  function mountPlayer(root, items, options) {
    const opt = Object.assign({ height: 200, frame_interval_ms: 120, converged_fmax: 0.05, step_labels: null }, options || {});
    root.classList.add("av-player", `av-${themeName(opt)}`);
    const bar = el("div", "av-bar");
    const play = el("button", "av-btn av-btn-primary", "Play"); play.type = "button";
    const slider = el("input", "av-slider"); slider.type = "range"; slider.min = "0"; slider.step = "any";   // fractional while playing, so the bar glides slider.setAttribute("aria-label", "Step");
    const step = el("span", "av-step");
    bar.append(play, slider, step); root.appendChild(bar);
    const wrap = el("div", "av-stage");
    const grid = el("div", "av-grid"); grid.style.setProperty("--av-cols", String(Math.min(items.length, 5)));
    const overlay = el("canvas", "av-overlay");
    wrap.append(grid, overlay); root.appendChild(wrap);
    const nFrames = Math.max(...items.map(it => it.frames.length));
    slider.max = String(nFrames - 1);
    const labels = Array.isArray(opt.step_labels) && opt.step_labels.length ? opt.step_labels
      : (Array.isArray(items[0].step_labels) && items[0].step_labels.length ? items[0].step_labels : null);   // per-item labels from the sample shape
    const label = f => labels ? labels[Math.min(f, labels.length - 1)] : f + 1;
    const total = labels ? labels[labels.length - 1] : nFrames;
    let renderer = null, lastClip = "";
    let frame = 0, playing = false, last = 0;
    const cards = items.map(it => {
      const card = el("div", "av-card");
      const head = el("div", "av-head");
      const title = el("strong", "", it.name);
      const right = el("span", "av-muted", `${it.atom_count} atoms`);
      head.append(title, right);
      const host = el("div", "av-canvas"); host.style.height = `${opt.height}px`; host.tabIndex = 0;
      host.setAttribute("aria-label", `${it.name}, drag to rotate`);
      const read = el("div", "av-readout");
      card.append(head, host, read); grid.appendChild(card);
      // The frame where the system stops: given by the caller, else the first frame at or below the threshold.
      let stop = Number.isInteger(it.stop_frame) ? it.stop_frame : it.fmax.findIndex(v => v <= opt.converged_fmax);
      if (!Number.isInteger(it.stop_frame) && Number.isInteger(it.converged_step) && Array.isArray(it.step_labels)) {
        const k = it.step_labels.indexOf(it.converged_step); if (k >= 0) stop = k;   // the real step where it converged
      }
      if (stop < 0) stop = null;
      // Readout built once; the number glides between frames instead of jumping (see tickReadouts).
      read.innerHTML = '<span class="av-muted">max |F|</span> <b class="av-num"></b> <span class="av-muted">eV/Å</span>';
      const c = { it, host, card, head, read, num: read.querySelector(".av-num"), stop, view: null, target: null, shown: null, text: "" };
      host.__avOnShow = () => { if (c.view) c.view.dirty = true; };
      seen.observe(host);
      stats.views++;
      return c;
    });
    const show = f => {
      frame = f; if (!playing) slider.value = String(f); step.textContent = `Step ${label(f)} / ${total}`;
      cards.forEach(c => {
        if (c.view) c.view.setFrame(f);
        const fi = Math.min(f, c.it.fmax.length - 1), fm = c.it.fmax[fi];
        c.target = fm; if (c.shown === null || !playing) c.shown = fm;
        const done = c.stop !== null && f >= c.stop;
        if (done !== c.done) {
          c.done = done; c.card.classList.toggle("is-done", done);
          const mark = c.head.querySelector(".av-done");
          if (done && !mark) c.head.querySelector("strong").insertAdjacentHTML("afterbegin", CHECK);
          if (!done && mark) mark.remove();
        }
      });
    };
    slider.addEventListener("input", () => { playing = false; play.textContent = "Play"; show(Math.round(Number(slider.value))); });
    play.addEventListener("click", () => { playing = !playing; play.textContent = playing ? "Pause" : "Play"; if (!playing) slider.value = String(frame); if (playing && frame >= nFrames - 1) show(0); });
    // Ease toward the target in log space (forces span decades), about one frame interval to settle.
    const fmt = v => v.toFixed(2);
    let lastT = 0;
    const tickReadouts = t => {
      const dt = lastT ? t - lastT : 0; lastT = t;
      const k = 1 - Math.exp(-dt / (opt.frame_interval_ms * 0.45));
      for (const c of cards) {
        if (c.target === null) continue;
        if (c.shown !== c.target) {
          const a = Math.log(Math.max(c.shown, 1e-6)), b = Math.log(Math.max(c.target, 1e-6));
          c.shown = Math.abs(b - a) < 1e-3 ? c.target : Math.exp(a + (b - a) * k);
        }
        const txt = fmt(c.shown);
        if (txt !== c.text) { c.text = txt; c.num.textContent = txt; }
      }
    };
    show(0); tickReadouts(0);
    const resize = () => {
      const w = wrap.clientWidth, h = wrap.clientHeight; if (!w || !h || !renderer) return;
      renderer.setSize(w, h, false); overlay.style.width = `${w}px`; overlay.style.height = `${h}px`;
      cards.forEach(c => { if (c.view) c.view.dirty = true; });
    };
    new ResizeObserver(resize).observe(wrap);
    startLoop(t => {
      if (!wrap.__avVisible && !cards.some(c => c.host.__avVisible)) return;
      if (!renderer) { renderer = makeRenderer(overlay); renderer.setScissorTest(true); resize(); }
      const t0 = performance.now();
      for (const c of cards) {
        if (!c.view && c.host.__avVisible && performance.now() - t0 < 12) {
          c.view = new View(Object.assign({}, c.it, { periodic: !!c.it.cell }), opt, true);
          c.view.attach(c.host); c.view.setFrame(frame);
        }
      }
      if (playing && t - last > opt.frame_interval_ms) {
        last = t;
        if (frame >= nFrames - 1) { playing = false; play.textContent = "Play"; slider.value = String(frame); } else show(frame + 1);
      }
      // Glide the bar between frames while playing: fraction of the current frame interval.
      if (playing) slider.value = String(Math.min(nFrames - 1, frame + Math.min(1, (t - last) / opt.frame_interval_ms)));
      tickReadouts(t);
      let any = false;
      for (const c of cards) if (c.view && (c.view.step() || c.view.dirty)) any = true;
      if (!any) return;
      const box = wrap.getBoundingClientRect(), H = wrap.clientHeight;
      renderer.setScissorTest(false); renderer.setClearColor(0x000000, 0); renderer.clear();
      renderer.setScissorTest(true);
      const td = performance.now();
      let clip = "";
      for (const c of cards) {
        if (!c.view) continue;
        const r = c.host.getBoundingClientRect();
        const x = r.left - box.left, y = H - (r.bottom - box.top), w = r.width, h = r.height;
        if (!w || !h) continue;
        clip += roundRect(r.left - box.left, r.top - box.top, w, h, RADIUS);
        c.view.setAspect(w / h);
        renderer.setViewport(x, y, w, h); renderer.setScissor(x, y, w, h);
        renderer.render(c.view.scene, c.view.camera);
        c.view.dirty = false;
      }
      // Round the overlay's corners to match the panels; the overlay takes no pointer events.
      if (clip !== lastClip) { lastClip = clip; overlay.style.clipPath = `path("${clip}")`; }
      stats.draws++; stats.draw_ms += performance.now() - td;
      if (stats.first_ms === null) stats.first_ms = performance.now() - T0;
    });
    wrap.__avOnShow = () => cards.forEach(c => { if (c.view) c.view.dirty = true; });
    seen.observe(wrap);
    return { cards, show };
  }

  window.AlchemiViewer = { mountStructure, mountPlayer, theme: themeName, stats, version: "0.3.2" };
})();
