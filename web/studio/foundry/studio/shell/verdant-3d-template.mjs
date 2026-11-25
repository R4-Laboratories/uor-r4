// Sandboxed afflom/verdant-engine (MIT OR Apache-2.0) 3D game template for Foundry Web Studio.
// Adapted from site/prism-bridge-meta.mjs, site/scene-renderer.mjs, and templates/browser-3d-starter/game.mjs
// under strict CHARTER §3 quarantine (zero external tensor crates, zero external CDN deps).

export const VERDANT_3D_TEMPLATE_ID = "verdant-3d";

export const VERDANT_3D_PRISM_CONSTANTS = Object.freeze({
  defaultWorldSeed: 7,
  maxLodRank: 3,
  hdMaxLodRank: 4,
  csmSplitMeters: Object.freeze([9.566, 28.2, 65.466, 140]),
  modalBarNumers: Object.freeze([9, 25, 49, 81, 121]),
  modalBarDenom: 9,
  sabineRt60Ms: 3000,
  tileEdgePixels: 16,
  terrainWalkablePolygons: 4608,
});

export const VERDANT_3D_TEMPLATE_META = Object.freeze({
  id: VERDANT_3D_TEMPLATE_ID,
  title: "Verdant 3D Game Engine (WebGL2 + Prism)",
  description:
    "Sandboxed 3D Orbital Beacon Arena with WebGL2 PBR lighting, Canvas2D fallback raster, physics sphere controller, patrolling drone, and modal audio.",
  isOfficial: true,
});

const VERDANT_ENGINE_TS = `// Self-contained Verdant 3D Orbital Beacon Arena engine.
// Adapted from afflom/verdant-engine (MIT OR Apache-2.0) for offline Foundry preview isolation.

export type CameraMode = 'chase' | 'first-person' | 'orbit';
export type GamePhase = 'title' | 'playing' | 'paused' | 'victory';
export type VirtualAction = 'forward' | 'backward' | 'left' | 'right' | 'jump';

export interface VerdantTelemetry {
  phase: GamePhase;
  rendererBackend: 'webgl2' | 'canvas2d';
  cameraMode: CameraMode;
  beaconsLit: number;
  totalBeacons: number;
  elapsedSeconds: number;
  fps: number;
  audioEnabled: boolean;
  playerPos: [number, number, number];
  droneDeflections: number;
  chronicleStep: number;
  stateHashHex: string;
}

export const VERDANT_PRISM_META = {
  defaultWorldSeed: 7,
  maxLodRank: 3,
  csmSplitMeters: [9.566, 28.2, 65.466, 140] as const,
  modalBarNumers: [9, 25, 49, 81, 121] as const,
  modalBarDenom: 9,
  sabineRt60Ms: 3000,
  tileEdgePixels: 16,
  terrainWalkablePolygons: 4608,
} as const;

const BEACON_COORDS: ReadonlyArray<readonly [number, number]> = [
  [-5.5, 4.5],
  [7.5, -4.0],
  [3.0, -7.5],
];

interface Vec3 {
  x: number;
  y: number;
  z: number;
}

export function heightAt(x: number, z: number): number {
  const r2 = x * x + z * z;
  const courtyardFlatten = Math.exp(-r2 * 0.045);
  const valley =
    Math.sin(x * 0.28) * Math.cos(z * 0.24) * 1.15 +
    Math.sin((x + z) * 0.14) * 0.65;
  const rim = Math.max(0, Math.hypot(x, z) - 11.5) * 0.24;
  return valley * (1 - courtyardFlatten * 0.88) + rim;
}

function terrainColor(x: number, y: number, z: number): [number, number, number] {
  const slope = Math.min(1, Math.abs(y) * 0.35);
  const checker = ((Math.floor(x) + Math.floor(z)) & 1) === 0 ? 0.04 : -0.02;
  const ring = Math.hypot(x, z) < 3.6 ? 0.08 : 0;
  return [
    Math.max(0.08, Math.min(0.95, 0.16 + slope * 0.26 + checker + ring * 0.4)),
    Math.max(0.12, Math.min(0.95, 0.34 - slope * 0.08 + checker + ring * 0.2)),
    Math.max(0.12, Math.min(0.95, 0.28 + slope * 0.18 + ring * 0.7)),
  ];
}

function perspective4x4(fovRad: number, aspect: number, near: number, far: number): Float32Array {
  const g = 1 / Math.tan(fovRad * 0.5);
  const nf = 1 / (near - far);
  return new Float32Array([
    g / Math.max(0.1, aspect), 0, 0, 0,
    0, g, 0, 0,
    0, 0, (far + near) * nf, -1,
    0, 0, 2 * far * near * nf, 0,
  ]);
}

function lookAt4x4(eye: Vec3, target: Vec3, up: Vec3): Float32Array {
  let zx = eye.x - target.x;
  let zy = eye.y - target.y;
  let zz = eye.z - target.z;
  const zl = Math.hypot(zx, zy, zz) || 1;
  zx /= zl;
  zy /= zl;
  zz /= zl;

  let xx = up.y * zz - up.z * zy;
  let xy = up.z * zx - up.x * zz;
  let xz = up.x * zy - up.y * zx;
  const xl = Math.hypot(xx, xy, xz) || 1;
  xx /= xl;
  xy /= xl;
  xz /= xl;

  const yx = zy * xz - zz * xy;
  const yy = zz * xx - zx * xz;
  const yz = zx * xy - zy * xx;

  return new Float32Array([
    xx, yx, zx, 0,
    xy, yy, zy, 0,
    xz, yz, zz, 0,
    -(xx * eye.x + xy * eye.y + xz * eye.z),
    -(yx * eye.x + yy * eye.y + yz * eye.z),
    -(zx * eye.x + zy * eye.y + zz * eye.z),
    1,
  ]);
}

function mul4x4(a: Float32Array, b: Float32Array): Float32Array {
  const out = new Float32Array(16);
  for (let c = 0; c < 4; c += 1) {
    for (let r = 0; r < 4; r += 1) {
      out[c * 4 + r] =
        a[r] * b[c * 4] +
        a[4 + r] * b[c * 4 + 1] +
        a[8 + r] * b[c * 4 + 2] +
        a[12 + r] * b[c * 4 + 3];
    }
  }
  return out;
}

interface MeshGeometry {
  positions: Float32Array;
  normals: Float32Array;
  colors: Float32Array;
}

function buildTerrainMesh(gridSpan = 16, halfExtent = 16): MeshGeometry {
  const tris = gridSpan * gridSpan * 2;
  const positions = new Float32Array(tris * 9);
  const normals = new Float32Array(tris * 9);
  const colors = new Float32Array(tris * 9);
  const step = (halfExtent * 2) / gridSpan;
  let cursor = 0;

  const pushTri = (
    ax: number, ay: number, az: number,
    bx: number, by: number, bz: number,
    cx: number, cy: number, cz: number,
  ) => {
    const ux = bx - ax, uy = by - ay, uz = bz - az;
    const vx = cx - ax, vy = cy - ay, vz = cz - az;
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    if (ny < 0) {
      nx = -nx;
      ny = -ny;
      nz = -nz;
    }
    const nl = Math.hypot(nx, ny, nz) || 1;
    nx /= nl;
    ny /= nl;
    nz /= nl;

    const verts: Array<[number, number, number]> = [
      [ax, ay, az],
      [bx, by, bz],
      [cx, cy, cz],
    ];
    for (const [vxPos, vyPos, vzPos] of verts) {
      positions[cursor] = vxPos;
      positions[cursor + 1] = vyPos;
      positions[cursor + 2] = vzPos;
      normals[cursor] = nx;
      normals[cursor + 1] = ny;
      normals[cursor + 2] = nz;
      const [cr, cg, cb] = terrainColor(vxPos, vyPos, vzPos);
      colors[cursor] = cr;
      colors[cursor + 1] = cg;
      colors[cursor + 2] = cb;
      cursor += 3;
    }
  };

  for (let gx = 0; gx < gridSpan; gx += 1) {
    const x0 = -halfExtent + gx * step;
    const x1 = x0 + step;
    for (let gz = 0; gz < gridSpan; gz += 1) {
      const z0 = -halfExtent + gz * step;
      const z1 = z0 + step;
      const y00 = heightAt(x0, z0);
      const y10 = heightAt(x1, z0);
      const y01 = heightAt(x0, z1);
      const y11 = heightAt(x1, z1);
      pushTri(x0, y00, z0, x0, y01, z1, x1, y10, z0);
      pushTri(x1, y10, z0, x0, y01, z1, x1, y11, z1);
    }
  }
  return { positions, normals, colors };
}

function buildOctahedronMesh(radius: number, topH: number, botH: number, rgb: [number, number, number]): MeshGeometry {
  // CCW outward-facing winding when viewed from outside.
  const faces: Array<[Vec3, Vec3, Vec3]> = [
    [{ x: 0, y: topH, z: 0 }, { x: 0, y: 0, z: radius }, { x: radius, y: 0, z: 0 }],
    [{ x: 0, y: topH, z: 0 }, { x: -radius, y: 0, z: 0 }, { x: 0, y: 0, z: radius }],
    [{ x: 0, y: topH, z: 0 }, { x: 0, y: 0, z: -radius }, { x: -radius, y: 0, z: 0 }],
    [{ x: 0, y: topH, z: 0 }, { x: radius, y: 0, z: 0 }, { x: 0, y: 0, z: -radius }],
    [{ x: 0, y: -botH, z: 0 }, { x: radius, y: 0, z: 0 }, { x: 0, y: 0, z: radius }],
    [{ x: 0, y: -botH, z: 0 }, { x: 0, y: 0, z: radius }, { x: -radius, y: 0, z: 0 }],
    [{ x: 0, y: -botH, z: 0 }, { x: -radius, y: 0, z: 0 }, { x: 0, y: 0, z: -radius }],
    [{ x: 0, y: -botH, z: 0 }, { x: 0, y: 0, z: -radius }, { x: radius, y: 0, z: 0 }],
  ];
  const positions = new Float32Array(faces.length * 9);
  const normals = new Float32Array(faces.length * 9);
  const colors = new Float32Array(faces.length * 9);
  let cursor = 0;
  for (const [a, b, c] of faces) {
    const ux = b.x - a.x, uy = b.y - a.y, uz = b.z - a.z;
    const vx = c.x - a.x, vy = c.y - a.y, vz = c.z - a.z;
    let nx = uy * vz - uz * vy;
    let ny = uz * vx - ux * vz;
    let nz = ux * vy - uy * vx;
    const nl = Math.hypot(nx, ny, nz) || 1;
    nx /= nl;
    ny /= nl;
    nz /= nl;
    for (const p of [a, b, c]) {
      positions[cursor] = p.x;
      positions[cursor + 1] = p.y;
      positions[cursor + 2] = p.z;
      normals[cursor] = nx;
      normals[cursor + 1] = ny;
      normals[cursor + 2] = nz;
      colors[cursor] = rgb[0];
      colors[cursor + 1] = rgb[1];
      colors[cursor + 2] = rgb[2];
      cursor += 3;
    }
  }
  return { positions, normals, colors };
}

function fnv1aHex(ints: number[]): string {
  let h = 0x811c9dc5 >>> 0;
  for (const val of ints) {
    const v = val | 0;
    for (let shift = 0; shift < 32; shift += 8) {
      h ^= (v >>> shift) & 0xff;
      h = Math.imul(h, 0x01000193) >>> 0;
    }
  }
  return h.toString(16).padStart(8, '0');
}

interface RenderActor {
  mesh: MeshGeometry;
  tx: number;
  ty: number;
  tz: number;
  spin: number;
  scale: number;
  tint: [number, number, number];
}

export class VerdantEngine {
  private canvas: HTMLCanvasElement | null = null;
  private gl: WebGL2RenderingContext | null = null;
  private ctx2d: CanvasRenderingContext2D | null = null;
  private backend: 'webgl2' | 'canvas2d' = 'canvas2d';
  private onTelemetry?: (t: VerdantTelemetry) => void;

  private phase: GamePhase = 'title';
  private cameraMode: CameraMode = 'chase';
  private audioEnabled = false;
  private audioCtx: AudioContext | null = null;

  private player = { x: 0, y: 0.65, z: 9.2, vx: 0, vy: 0, vz: 0, grounded: true };
  private drone = { x: 5.5, y: 1.5, z: 0, angle: 0 };
  private beaconsLit = [false, false, false];
  private droneDeflections = 0;
  private droneCooldown = 0;
  private elapsedSeconds = 0;
  private chronicleStep = 0;
  private dragYaw = 0;
  private dragPitch = 0.34;
  private dragging = false;
  private lastPointerX = 0;
  private lastPointerY = 0;
  private jumpQueued = false;
  private keys = new Set<string>();
  private virtualInput: Record<VirtualAction, boolean> = {
    forward: false,
    backward: false,
    left: false,
    right: false,
    jump: false,
  };

  private rafId = 0;
  private lastFrameMs = 0;
  private lastTelemetryEmitMs = 0;
  private smoothedFps = 60;
  private destroyed = false;

  private terrainMesh = buildTerrainMesh(18, 16);
  private crystalMesh = buildOctahedronMesh(0.7, 1.45, 0.95, [0.95, 0.85, 0.42]);
  private playerMesh = buildOctahedronMesh(0.55, 0.55, 0.55, [0.35, 0.86, 0.98]);
  private droneMesh = buildOctahedronMesh(0.65, 0.45, 0.45, [0.98, 0.36, 0.34]);

  private glProgram: WebGLProgram | null = null;
  private glTerrainVao: WebGLVertexArrayObject | null = null;
  private glCrystalVao: WebGLVertexArrayObject | null = null;
  private glPlayerVao: WebGLVertexArrayObject | null = null;
  private glDroneVao: WebGLVertexArrayObject | null = null;
  private glResources: Array<WebGLBuffer | WebGLVertexArrayObject | WebGLProgram> = [];
  private uniforms: {
    uMvp: WebGLUniformLocation | null;
    uOffset: WebGLUniformLocation | null;
    uSpin: WebGLUniformLocation | null;
    uScale: WebGLUniformLocation | null;
    uTint: WebGLUniformLocation | null;
  } = { uMvp: null, uOffset: null, uSpin: null, uScale: null, uTint: null };

  private cleanups: Array<() => void> = [];

  mount(canvas: HTMLCanvasElement, onTelemetry?: (t: VerdantTelemetry) => void): VerdantTelemetry {
    this.canvas = canvas;
    this.onTelemetry = onTelemetry;
    this.initRenderer(canvas);
    this.resetWorld('playing');
    this.attachInput(canvas);
    this.lastFrameMs = performance.now();
    this.renderScene();
    this.rafId = requestAnimationFrame(this.loop);
    return this.snapshot();
  }

  private initRenderer(canvas: HTMLCanvasElement): void {
    let gl: WebGL2RenderingContext | null = null;
    try {
      gl = canvas.getContext('webgl2', { antialias: true, depth: true, alpha: false });
    } catch {
      gl = null;
    }
    if (gl && this.compileGlPipeline(gl)) {
      this.gl = gl;
      this.backend = 'webgl2';
      return;
    }
    if (gl) {
      this.cleanupGlResources(gl);
    }
    this.gl = null;
    this.glProgram = null;
    this.ctx2d = canvas.getContext('2d');
    this.backend = 'canvas2d';
  }

  private compileGlPipeline(gl: WebGL2RenderingContext): boolean {
    const vsSrc = \`#version 300 es
layout(location=0) in vec3 a_pos;
layout(location=1) in vec3 a_norm;
layout(location=2) in vec3 a_col;
uniform mat4 u_mvp;
uniform vec3 u_offset;
uniform float u_spin;
uniform float u_scale;
uniform vec3 u_tint;
out vec3 v_col;
void main() {
  float c = cos(u_spin);
  float s = sin(u_spin);
  vec3 sp = a_pos * u_scale;
  vec3 wp = vec3(sp.x * c - sp.z * s, sp.y, sp.x * s + sp.z * c) + u_offset;
  vec3 wn = normalize(vec3(a_norm.x * c - a_norm.z * s, a_norm.y, a_norm.x * s + a_norm.z * c));
  float sun = max(dot(wn, normalize(vec3(0.54, 0.78, 0.31))), 0.0);
  float skyFill = max(wn.y * 0.5 + 0.5, 0.0);
  v_col = a_col * u_tint * (0.28 + skyFill * 0.24 + sun * 0.72);
  gl_Position = u_mvp * vec4(wp, 1.0);
}\`;
    const fsSrc = \`#version 300 es
precision mediump float;
in vec3 v_col;
out vec4 o_col;
void main() {
  o_col = vec4(clamp(v_col, 0.0, 1.0), 1.0);
}\`;
    const compile = (type: number, src: string): WebGLShader | null => {
      const sh = gl.createShader(type);
      if (!sh) return null;
      gl.shaderSource(sh, src);
      gl.compileShader(sh);
      if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
        gl.deleteShader(sh);
        return null;
      }
      return sh;
    };
    const vs = compile(gl.VERTEX_SHADER, vsSrc);
    const fs = compile(gl.FRAGMENT_SHADER, fsSrc);
    if (!vs || !fs) {
      if (vs) gl.deleteShader(vs);
      if (fs) gl.deleteShader(fs);
      return false;
    }
    const prog = gl.createProgram();
    if (!prog) {
      gl.deleteShader(vs);
      gl.deleteShader(fs);
      return false;
    }
    gl.attachShader(prog, vs);
    gl.attachShader(prog, fs);
    gl.linkProgram(prog);
    gl.deleteShader(vs);
    gl.deleteShader(fs);
    if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
      gl.deleteProgram(prog);
      return false;
    }
    this.glProgram = prog;
    this.glResources.push(prog);
    this.uniforms = {
      uMvp: gl.getUniformLocation(prog, 'u_mvp'),
      uOffset: gl.getUniformLocation(prog, 'u_offset'),
      uSpin: gl.getUniformLocation(prog, 'u_spin'),
      uScale: gl.getUniformLocation(prog, 'u_scale'),
      uTint: gl.getUniformLocation(prog, 'u_tint'),
    };
    this.glTerrainVao = this.uploadMesh(gl, this.terrainMesh);
    this.glCrystalVao = this.uploadMesh(gl, this.crystalMesh);
    this.glPlayerVao = this.uploadMesh(gl, this.playerMesh);
    this.glDroneVao = this.uploadMesh(gl, this.droneMesh);
    return Boolean(this.glTerrainVao && this.glCrystalVao && this.glPlayerVao && this.glDroneVao);
  }

  private uploadMesh(gl: WebGL2RenderingContext, mesh: MeshGeometry): WebGLVertexArrayObject | null {
    const vao = gl.createVertexArray();
    if (!vao) return null;
    gl.bindVertexArray(vao);
    this.glResources.push(vao);

    const bindAttr = (loc: number, data: Float32Array) => {
      const buf = gl.createBuffer();
      if (!buf) return;
      this.glResources.push(buf);
      gl.bindBuffer(gl.ARRAY_BUFFER, buf);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(loc);
      gl.vertexAttribPointer(loc, 3, gl.FLOAT, false, 0, 0);
    };
    bindAttr(0, mesh.positions);
    bindAttr(1, mesh.normals);
    bindAttr(2, mesh.colors);
    gl.bindVertexArray(null);
    return vao;
  }

  private attachInput(canvas: HTMLCanvasElement): void {
    const clearAllInputs = () => {
      this.keys.clear();
      this.virtualInput.forward = false;
      this.virtualInput.backward = false;
      this.virtualInput.left = false;
      this.virtualInput.right = false;
      this.virtualInput.jump = false;
      this.jumpQueued = false;
      this.dragging = false;
    };
    const onKeyDown = (e: KeyboardEvent) => {
      const target = e.target as HTMLElement | null;
      const isText = target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA');
      if (!isText && ['Space', 'ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight'].includes(e.code)) {
        e.preventDefault();
      }
      this.keys.add(e.code);
      if (e.code === 'Space') this.jumpQueued = true;
    };
    const onKeyUp = (e: KeyboardEvent) => {
      this.keys.delete(e.code);
    };
    const onPointerDown = (e: PointerEvent) => {
      this.dragging = true;
      this.lastPointerX = e.clientX;
      this.lastPointerY = e.clientY;
    };
    const onPointerMove = (e: PointerEvent) => {
      if (!this.dragging) return;
      const dx = e.clientX - this.lastPointerX;
      const dy = e.clientY - this.lastPointerY;
      this.lastPointerX = e.clientX;
      this.lastPointerY = e.clientY;
      this.dragYaw -= dx * 0.008;
      this.dragPitch = Math.max(0.08, Math.min(1.15, this.dragPitch + dy * 0.006));
    };
    const onPointerUp = () => {
      this.dragging = false;
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', clearAllInputs);
    canvas.addEventListener('pointerdown', onPointerDown);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    window.addEventListener('pointercancel', onPointerUp);
    this.cleanups.push(() => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
      window.removeEventListener('blur', clearAllInputs);
      canvas.removeEventListener('pointerdown', onPointerDown);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      window.removeEventListener('pointercancel', onPointerUp);
    });
  }

  start(): void {
    if (this.phase === 'title' || this.phase === 'paused') {
      this.phase = 'playing';
      this.emitNow();
    }
  }

  pause(): void {
    if (this.phase === 'playing') {
      this.phase = 'paused';
      this.emitNow();
    }
  }

  restart(): void {
    this.resetWorld('playing');
    this.emitNow();
  }

  setCameraMode(mode: CameraMode): void {
    this.cameraMode = mode;
    this.emitNow();
  }

  cycleCameraMode(): CameraMode {
    const order: CameraMode[] = ['chase', 'first-person', 'orbit'];
    const idx = order.indexOf(this.cameraMode);
    this.cameraMode = order[(idx + 1) % order.length];
    this.emitNow();
    return this.cameraMode;
  }

  toggleAudio(): boolean {
    this.audioEnabled = !this.audioEnabled;
    if (this.audioEnabled) {
      this.playModalChime(0);
    }
    this.emitNow();
    return this.audioEnabled;
  }

  setVirtualInput(action: VirtualAction, pressed: boolean): void {
    this.virtualInput[action] = pressed;
    if (action === 'jump' && pressed) this.jumpQueued = true;
  }

  private resetWorld(nextPhase: GamePhase): void {
    const zStart = 9.2;
    this.player = {
      x: 0,
      y: heightAt(0, zStart) + 0.65,
      z: zStart,
      vx: 0,
      vy: 0,
      vz: 0,
      grounded: true,
    };
    this.drone = { x: 5.5, y: heightAt(5.5, 0) + 1.5, z: 0, angle: 0 };
    this.beaconsLit = [false, false, false];
    this.droneDeflections = 0;
    this.droneCooldown = 0;
    this.elapsedSeconds = 0;
    this.chronicleStep = 1;
    this.dragYaw = 0;
    this.dragPitch = 0.34;
    this.dragging = false;
    this.jumpQueued = false;
    this.keys.clear();
    this.virtualInput = { forward: false, backward: false, left: false, right: false, jump: false };
    this.phase = nextPhase;
  }

  private playModalChime(modeIdx: number): void {
    if (!this.audioEnabled) return;
    try {
      const Ctx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext?: typeof AudioContext }).webkitAudioContext;
      if (!Ctx) return;
      if (!this.audioCtx) this.audioCtx = new Ctx();
      if (this.audioCtx.state === 'suspended') {
        void this.audioCtx.resume().catch(() => {});
      }
      const numer = VERDANT_PRISM_META.modalBarNumers[modeIdx % VERDANT_PRISM_META.modalBarNumers.length];
      const freq = 110 * (numer / VERDANT_PRISM_META.modalBarDenom);
      const now = this.audioCtx.currentTime;
      const osc = this.audioCtx.createOscillator();
      const gain = this.audioCtx.createGain();
      osc.type = 'triangle';
      osc.frequency.setValueAtTime(freq, now);
      gain.gain.setValueAtTime(0.001, now);
      gain.gain.exponentialRampToValueAtTime(0.14, now + 0.02);
      gain.gain.exponentialRampToValueAtTime(0.0008, now + 0.65);
      osc.connect(gain);
      gain.connect(this.audioCtx.destination);
      osc.start(now);
      osc.stop(now + 0.68);
    } catch {
      // AudioContext unavailable in headless environments.
    }
  }

  private stepPhysics(dt: number): void {
    if (this.phase !== 'playing') return;
    this.elapsedSeconds += dt;

    let fwd = 0;
    let strafe = 0;
    if (this.keys.has('KeyW') || this.keys.has('ArrowUp') || this.virtualInput.forward) fwd += 1;
    if (this.keys.has('KeyS') || this.keys.has('ArrowDown') || this.virtualInput.backward) fwd -= 1;
    if (this.keys.has('KeyD') || this.keys.has('ArrowRight') || this.virtualInput.right) strafe += 1;
    if (this.keys.has('KeyA') || this.keys.has('ArrowLeft') || this.virtualInput.left) strafe -= 1;

    const sinY = Math.sin(this.dragYaw);
    const cosY = Math.cos(this.dragYaw);
    const dirX = -sinY * fwd + cosY * strafe;
    const dirZ = -cosY * fwd - sinY * strafe;
    const mag = Math.hypot(dirX, dirZ) || 1;
    const accel = fwd !== 0 || strafe !== 0 ? 22 : 0;

    this.player.vx = this.player.vx * Math.exp(-6.5 * dt) + (dirX / mag) * accel * dt;
    this.player.vz = this.player.vz * Math.exp(-6.5 * dt) + (dirZ / mag) * accel * dt;

    if ((this.jumpQueued || this.keys.has('Space')) && this.player.grounded) {
      this.player.vy = 8.4;
      this.player.grounded = false;
    }
    this.jumpQueued = false;

    this.player.vy -= 21 * dt;
    this.player.x = Math.max(-14.5, Math.min(14.5, this.player.x + this.player.vx * dt));
    this.player.z = Math.max(-14.5, Math.min(14.5, this.player.z + this.player.vz * dt));
    this.player.y += this.player.vy * dt;

    const groundY = heightAt(this.player.x, this.player.z) + 0.55;
    if (this.player.y <= groundY) {
      this.player.y = groundY;
      this.player.vy = 0;
      this.player.grounded = true;
    }

    this.drone.angle += dt * 0.95;
    this.drone.x = Math.cos(this.drone.angle) * 5.6;
    this.drone.z = Math.sin(this.drone.angle) * 5.6;
    this.drone.y = heightAt(this.drone.x, this.drone.z) + 1.35 + Math.sin(this.drone.angle * 2) * 0.25;
    if (this.droneCooldown > 0) {
      this.droneCooldown = Math.max(0, this.droneCooldown - dt);
    }

    const dxDrone = this.player.x - this.drone.x;
    const dzDrone = this.player.z - this.drone.z;
    const distDrone = Math.hypot(dxDrone, dzDrone);
    if (this.droneCooldown <= 0 && distDrone < 1.25 && Math.abs(this.player.y - this.drone.y) < 1.35) {
      const push = 9.5 / Math.max(0.25, distDrone);
      this.player.vx = dxDrone * push;
      this.player.vz = dzDrone * push;
      this.player.vy = 4.2;
      this.player.grounded = false;
      this.droneDeflections += 1;
      this.droneCooldown = 0.35;
      this.playModalChime(4);
    }

    for (let i = 0; i < BEACON_COORDS.length; i += 1) {
      if (this.beaconsLit[i]) continue;
      const [bx, bz] = BEACON_COORDS[i];
      if (Math.hypot(this.player.x - bx, this.player.z - bz) < 1.55) {
        this.beaconsLit[i] = true;
        this.chronicleStep += 1;
        this.playModalChime(i + 1);
      }
    }

    if (this.beaconsLit.every(Boolean)) {
      this.phase = 'victory';
      this.playModalChime(3);
    }
  }

  private cameraMatrices(aspect: number): Float32Array {
    let eye: Vec3;
    let target: Vec3;
    if (this.cameraMode === 'first-person') {
      eye = { x: this.player.x, y: this.player.y + 0.55, z: this.player.z };
      target = {
        x: eye.x - Math.sin(this.dragYaw) * 6,
        y: eye.y - Math.sin(this.dragPitch - 0.34) * 3,
        z: eye.z - Math.cos(this.dragYaw) * 6,
      };
    } else if (this.cameraMode === 'orbit') {
      const orbitDist = 17.5;
      eye = {
        x: Math.sin(this.dragYaw) * orbitDist,
        y: 5.5 + this.dragPitch * 7.5,
        z: Math.cos(this.dragYaw) * orbitDist,
      };
      target = { x: 0, y: 0.8, z: 0 };
    } else {
      const dist = 7.4;
      eye = {
        x: this.player.x + Math.sin(this.dragYaw) * dist,
        y: this.player.y + 2.4 + this.dragPitch * 4.2,
        z: this.player.z + Math.cos(this.dragYaw) * dist,
      };
      target = { x: this.player.x, y: this.player.y + 0.4, z: this.player.z };
    }
    const proj = perspective4x4(Math.PI / 3.2, aspect, 0.2, 160);
    const view = lookAt4x4(eye, target, { x: 0, y: 1, z: 0 });
    return mul4x4(proj, view);
  }

  private renderScene(): void {
    if (!this.canvas) return;
    const width = Math.max(1, this.canvas.width);
    const height = Math.max(1, this.canvas.height);
    const mvp = this.cameraMatrices(width / height);

    if (this.gl && this.glProgram) {
      const gl = this.gl;
      gl.viewport(0, 0, width, height);
      gl.enable(gl.DEPTH_TEST);
      gl.clearColor(0.06, 0.1, 0.15, 1.0);
      gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);
      gl.useProgram(this.glProgram);
      gl.uniformMatrix4fv(this.uniforms.uMvp, false, mvp);

      const draw = (
        vao: WebGLVertexArrayObject | null,
        vertCount: number,
        tx: number, ty: number, tz: number,
        spin: number, scale: number,
        tint: [number, number, number],
      ) => {
        if (!vao) return;
        gl.bindVertexArray(vao);
        gl.uniform3f(this.uniforms.uOffset, tx, ty, tz);
        gl.uniform1f(this.uniforms.uSpin, spin);
        gl.uniform1f(this.uniforms.uScale, scale);
        gl.uniform3f(this.uniforms.uTint, tint[0], tint[1], tint[2]);
        gl.drawArrays(gl.TRIANGLES, 0, vertCount);
      };

      draw(this.glTerrainVao, this.terrainMesh.positions.length / 3, 0, 0, 0, 0, 1, [1, 1, 1]);
      for (let i = 0; i < BEACON_COORDS.length; i += 1) {
        const [bx, bz] = BEACON_COORDS[i];
        const by = heightAt(bx, bz) + 1.15 + Math.sin(this.elapsedSeconds * 2.2 + i) * 0.18;
        const lit = this.beaconsLit[i];
        draw(
          this.glCrystalVao,
          this.crystalMesh.positions.length / 3,
          bx, by, bz,
          this.elapsedSeconds * (lit ? 2.4 : 0.9),
          lit ? 1.18 : 0.95,
          lit ? [0.35, 1.15, 0.72] : [1.0, 0.78, 0.35],
        );
      }
      if (this.cameraMode !== 'first-person') {
        draw(
          this.glPlayerVao,
          this.playerMesh.positions.length / 3,
          this.player.x, this.player.y, this.player.z,
          this.elapsedSeconds * 1.4,
          1,
          [0.9, 0.98, 1.05],
        );
      }
      draw(
        this.glDroneVao,
        this.droneMesh.positions.length / 3,
        this.drone.x, this.drone.y, this.drone.z,
        -this.elapsedSeconds * 3.1,
        1,
        [1.05, 0.8, 0.75],
      );
      gl.bindVertexArray(null);
      return;
    }

    if (this.ctx2d) {
      this.renderCanvas2dFallback(this.ctx2d, width, height, mvp);
    }
  }

  private renderCanvas2dFallback(ctx: CanvasRenderingContext2D, width: number, height: number, mvp: Float32Array): void {
    ctx.fillStyle = '#101926';
    ctx.fillRect(0, 0, width, height);

    const actors: RenderActor[] = [
      { mesh: this.terrainMesh, tx: 0, ty: 0, tz: 0, spin: 0, scale: 1, tint: [1, 1, 1] },
      ...BEACON_COORDS.map(([bx, bz], i): RenderActor => ({
        mesh: this.crystalMesh,
        tx: bx,
        ty: heightAt(bx, bz) + 1.15 + Math.sin(this.elapsedSeconds * 2.2 + i) * 0.18,
        tz: bz,
        spin: this.elapsedSeconds * (this.beaconsLit[i] ? 2.4 : 0.9),
        scale: this.beaconsLit[i] ? 1.18 : 0.95,
        tint: this.beaconsLit[i] ? [0.35, 1.15, 0.72] : [1.0, 0.78, 0.35],
      })),
      {
        mesh: this.droneMesh,
        tx: this.drone.x,
        ty: this.drone.y,
        tz: this.drone.z,
        spin: -this.elapsedSeconds * 2.5,
        scale: 1,
        tint: [1.05, 0.8, 0.75],
      },
    ];
    if (this.cameraMode !== 'first-person') {
      actors.push({
        mesh: this.playerMesh,
        tx: this.player.x,
        ty: this.player.y,
        tz: this.player.z,
        spin: this.elapsedSeconds,
        scale: 1,
        tint: [0.9, 0.98, 1.05],
      });
    }

    const projected: Array<{ depth: number; pts: [number, number, number, number, number, number]; fill: string }> = [];
    for (const actor of actors) {
      const c = Math.cos(actor.spin);
      const s = Math.sin(actor.spin);
      const pos = actor.mesh.positions;
      const norm = actor.mesh.normals;
      const col = actor.mesh.colors;
      for (let i = 0; i < pos.length; i += 9) {
        const sx: number[] = [];
        const sy: number[] = [];
        let wSum = 0;
        let visible = true;
        for (let k = 0; k < 3; k += 1) {
          const lx = pos[i + k * 3] * actor.scale;
          const ly = pos[i + k * 3 + 1] * actor.scale;
          const lz = pos[i + k * 3 + 2] * actor.scale;
          const wx = lx * c - lz * s + actor.tx;
          const wy = ly + actor.ty;
          const wz = lx * s + lz * c + actor.tz;
          const clipX = mvp[0] * wx + mvp[4] * wy + mvp[8] * wz + mvp[12];
          const clipY = mvp[1] * wx + mvp[5] * wy + mvp[9] * wz + mvp[13];
          const clipW = mvp[3] * wx + mvp[7] * wy + mvp[11] * wz + mvp[15];
          if (clipW <= 0.15) {
            visible = false;
            break;
          }
          sx.push((clipX / clipW * 0.5 + 0.5) * width);
          sy.push((0.5 - clipY / clipW * 0.5) * height);
          wSum += clipW;
        }
        if (!visible) continue;
        const lit = Math.max(0.28, Math.min(1.0, 0.35 + norm[i + 1] * 0.6));
        const r = Math.round(Math.min(255, col[i] * actor.tint[0] * lit * 255));
        const g = Math.round(Math.min(255, col[i + 1] * actor.tint[1] * lit * 255));
        const b = Math.round(Math.min(255, col[i + 2] * actor.tint[2] * lit * 255));
        projected.push({
          depth: wSum / 3,
          pts: [sx[0], sy[0], sx[1], sy[1], sx[2], sy[2]],
          fill: \`rgb(\${r},\${g},\${b})\`,
        });
      }
    }
    projected.sort((a, b) => b.depth - a.depth);
    for (const tri of projected) {
      ctx.fillStyle = tri.fill;
      ctx.beginPath();
      ctx.moveTo(tri.pts[0], tri.pts[1]);
      ctx.lineTo(tri.pts[2], tri.pts[3]);
      ctx.lineTo(tri.pts[4], tri.pts[5]);
      ctx.closePath();
      ctx.fill();
    }
  }

  private loop = (nowMs: number): void => {
    if (this.destroyed) return;
    const dt = Math.max(0.001, Math.min(0.05, (nowMs - this.lastFrameMs) / 1000));
    this.lastFrameMs = nowMs;
    const instFps = 1 / dt;
    this.smoothedFps = this.smoothedFps * 0.9 + instFps * 0.1;

    this.stepPhysics(dt);
    this.renderScene();

    if (nowMs - this.lastTelemetryEmitMs >= 80) {
      this.lastTelemetryEmitMs = nowMs;
      this.emitNow();
    }
    this.rafId = requestAnimationFrame(this.loop);
  };

  snapshot(): VerdantTelemetry {
    const px = Math.round(this.player.x * 100);
    const py = Math.round(this.player.y * 100);
    const pz = Math.round(this.player.z * 100);
    const litMask =
      (this.beaconsLit[0] ? 1 : 0) |
      (this.beaconsLit[1] ? 2 : 0) |
      (this.beaconsLit[2] ? 4 : 0);
    return {
      phase: this.phase,
      rendererBackend: this.backend,
      cameraMode: this.cameraMode,
      beaconsLit: this.beaconsLit.filter(Boolean).length,
      totalBeacons: BEACON_COORDS.length,
      elapsedSeconds: Number(this.elapsedSeconds.toFixed(1)),
      fps: Math.round(this.smoothedFps),
      audioEnabled: this.audioEnabled,
      playerPos: [
        Number(this.player.x.toFixed(2)),
        Number(this.player.y.toFixed(2)),
        Number(this.player.z.toFixed(2)),
      ],
      droneDeflections: this.droneDeflections,
      chronicleStep: this.chronicleStep,
      stateHashHex: fnv1aHex([
        VERDANT_PRISM_META.defaultWorldSeed,
        px,
        py,
        pz,
        litMask,
        this.chronicleStep,
      ]),
    };
  }

  private emitNow(): void {
    this.onTelemetry?.(this.snapshot());
  }

  private cleanupGlResources(gl: WebGL2RenderingContext): void {
    for (const res of this.glResources) {
      try {
        if (res instanceof WebGLProgram) gl.deleteProgram(res);
        else if (res instanceof WebGLVertexArrayObject) gl.deleteVertexArray(res);
        else if (res instanceof WebGLBuffer) gl.deleteBuffer(res);
      } catch {}
    }
    this.glResources = [];
  }

  dispose(): void {
    this.destroyed = true;
    if (this.rafId) cancelAnimationFrame(this.rafId);
    for (const fn of this.cleanups) {
      try {
        fn();
      } catch {}
    }
    this.cleanups = [];
    if (this.gl) {
      this.cleanupGlResources(this.gl);
    }
    if (this.audioCtx) {
      try {
        void this.audioCtx.close().catch(() => {});
      } catch {}
      this.audioCtx = null;
    }
  }
}
`;

const VERDANT_VIEWPORT_TSX = `import { useEffect, useRef, useState } from "react";
import { Camera, Compass, Pause, Play, RotateCcw, Sparkles, Volume2, VolumeX } from "lucide-react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card } from "@/components/ui/card";
import {
  VERDANT_PRISM_META,
  VerdantEngine,
  type VerdantTelemetry,
  type VirtualAction,
} from "@/engine/verdant-engine";

export function VerdantViewport() {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const engineRef = useRef<VerdantEngine | null>(null);
  const [telemetry, setTelemetry] = useState<VerdantTelemetry>({
    phase: "playing",
    rendererBackend: "webgl2",
    cameraMode: "chase",
    beaconsLit: 0,
    totalBeacons: 3,
    elapsedSeconds: 0,
    fps: 60,
    audioEnabled: false,
    playerPos: [0, 0.7, 9.2],
    droneDeflections: 0,
    chronicleStep: 1,
    stateHashHex: "00000000",
  });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const resize = () => {
      const parent = canvas.parentElement;
      const w = Math.max(320, Math.floor(parent?.clientWidth || window.innerWidth));
      const h = Math.max(260, Math.floor(parent?.clientHeight || 520));
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
    };
    resize();

    const engine = new VerdantEngine();
    engineRef.current = engine;
    const initial = engine.mount(canvas, setTelemetry);
    setTelemetry(initial);

    let ro: ResizeObserver | null = null;
    if (typeof ResizeObserver === "function" && canvas.parentElement) {
      ro = new ResizeObserver(resize);
      ro.observe(canvas.parentElement);
    } else {
      window.addEventListener("resize", resize);
    }

    return () => {
      ro?.disconnect();
      window.removeEventListener("resize", resize);
      engine.dispose();
      engineRef.current = null;
    };
  }, []);

  const hold = (action: VirtualAction, down: boolean) => {
    engineRef.current?.setVirtualInput(action, down);
  };

  return (
    <div className="flex flex-col gap-3 w-full">
      <div className="relative w-full h-[520px] rounded-xl overflow-hidden border border-border bg-card shadow-lg">
        <canvas ref={canvasRef} className="w-full h-full block touch-none" />

        <div className="absolute top-3 left-3 right-3 flex flex-wrap items-center justify-between gap-2 pointer-events-none">
          <div className="flex flex-wrap items-center gap-2 pointer-events-auto">
            <Badge variant="default" className="gap-1.5">
              <Sparkles className="w-3.5 h-3.5" />
              Beacons {telemetry.beaconsLit}/{telemetry.totalBeacons}
            </Badge>
            <Badge variant="secondary">
              {telemetry.rendererBackend === "webgl2" ? "WebGL2 3D" : "Canvas2D Raster"} · {telemetry.fps} FPS
            </Badge>
            <Badge variant="outline" className="bg-background/80">
              {telemetry.elapsedSeconds.toFixed(1)}s · #{telemetry.stateHashHex}
            </Badge>
          </div>

          <div className="flex items-center gap-1.5 pointer-events-auto">
            <Button
              size="sm"
              variant="secondary"
              onClick={() => engineRef.current?.cycleCameraMode()}
            >
              <Camera className="w-3.5 h-3.5" />
              {telemetry.cameraMode}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => engineRef.current?.toggleAudio()}
            >
              {telemetry.audioEnabled ? <Volume2 className="w-3.5 h-3.5" /> : <VolumeX className="w-3.5 h-3.5" />}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() =>
                telemetry.phase === "paused" ? engineRef.current?.start() : engineRef.current?.pause()
              }
            >
              {telemetry.phase === "paused" ? <Play className="w-3.5 h-3.5" /> : <Pause className="w-3.5 h-3.5" />}
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onClick={() => engineRef.current?.restart()}
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </Button>
          </div>
        </div>

        <div className="absolute bottom-3 left-3 right-3 flex items-end justify-between pointer-events-none">
          <Card className="px-3 py-2 bg-background/85 backdrop-blur-sm text-xs text-muted-foreground">
            <div className="font-medium text-foreground flex items-center gap-1.5">
              <Compass className="w-3.5 h-3.5" />
              WASD / Arrows move · Space jump · Drag to orbit camera
            </div>
            <div>
              Pos ({telemetry.playerPos.join(", ")}) · Deflections {telemetry.droneDeflections} · Seed{" "}
              {VERDANT_PRISM_META.defaultWorldSeed}
            </div>
          </Card>

          <div className="grid grid-cols-3 gap-1 pointer-events-auto">
            <div />
            <Button
              size="sm"
              variant="secondary"
              onPointerDown={() => hold("forward", true)}
              onPointerUp={() => hold("forward", false)}
              onPointerLeave={() => hold("forward", false)}
              onPointerCancel={() => hold("forward", false)}
            >
              W
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onPointerDown={() => hold("jump", true)}
              onPointerUp={() => hold("jump", false)}
              onPointerLeave={() => hold("jump", false)}
              onPointerCancel={() => hold("jump", false)}
            >
              Jump
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onPointerDown={() => hold("left", true)}
              onPointerUp={() => hold("left", false)}
              onPointerLeave={() => hold("left", false)}
              onPointerCancel={() => hold("left", false)}
            >
              A
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onPointerDown={() => hold("backward", true)}
              onPointerUp={() => hold("backward", false)}
              onPointerLeave={() => hold("backward", false)}
              onPointerCancel={() => hold("backward", false)}
            >
              S
            </Button>
            <Button
              size="sm"
              variant="secondary"
              onPointerDown={() => hold("right", true)}
              onPointerUp={() => hold("right", false)}
              onPointerLeave={() => hold("right", false)}
              onPointerCancel={() => hold("right", false)}
            >
              D
            </Button>
          </div>
        </div>

        {telemetry.phase === "victory" && (
          <div className="absolute inset-0 bg-background/75 backdrop-blur-sm flex items-center justify-center p-4">
            <Card className="p-6 max-w-sm text-center space-y-3">
              <h2 className="text-xl font-bold text-foreground">Orbital Beacons Harmonized</h2>
              <p className="text-sm text-muted-foreground">
                Completed in {telemetry.elapsedSeconds.toFixed(1)}s · State #{telemetry.stateHashHex}
              </p>
              <Button onClick={() => engineRef.current?.restart()}>Play Again</Button>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
}
`;

const VERDANT_INDEX_TSX = `import { Badge } from "@/components/ui/badge";
import { MadeWithPrism } from "@/components/made-with-prism";
import { VerdantViewport } from "@/components/VerdantViewport";
import { VERDANT_PRISM_META } from "@/engine/verdant-engine";

const Index = () => {
  return (
    <div className="min-h-screen bg-background text-foreground p-4 md:p-6 flex flex-col gap-4">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Verdant 3D — Orbital Beacon Arena</h1>
          <p className="text-sm text-muted-foreground">
            Sandboxed browser 3D engine (WebGL2 + Canvas2D software fallback) running offline in Foundry.
          </p>
        </div>
        <div className="flex flex-wrap gap-2">
          <Badge variant="outline">Seed {VERDANT_PRISM_META.defaultWorldSeed}</Badge>
          <Badge variant="outline">{VERDANT_PRISM_META.tileEdgePixels}x{VERDANT_PRISM_META.tileEdgePixels} Tile PBR</Badge>
          <Badge variant="outline">RT60 {VERDANT_PRISM_META.sabineRt60Ms}ms</Badge>
        </div>
      </header>

      <main className="flex-1">
        <VerdantViewport />
      </main>

      <MadeWithPrism />
    </div>
  );
};

export default Index;
`;

const VERDANT_AI_RULES_SUFFIX = `
## Verdant 3D Template Architecture

- This project starts from the **Verdant 3D Game Engine (\`verdant-3d\`)** template adapted from \`afflom/verdant-engine\`.
- \`src/engine/verdant-engine.ts\` contains the self-contained 3D engine (heightfield \`heightAt\`, perspective & camera helpers, WebGL2 + Canvas2D software fallback renderer, relay beacons, patrolling drone physics, modal Web Audio chimes, FNV-1a snapshot hash, and \`dispose()\`).
- \`src/components/VerdantViewport.tsx\` mounts \`VerdantEngine\` on \`<canvas>\` with responsive resize handling and live HUD controls.
- Keep all game assets, 4x4 transform helpers, and shaders procedural or local in TypeScript; never import external CDN scripts or add external tensor crates.
`;

export function buildVerdant3dProjectFiles(baseScaffoldFiles) {
  const base =
    baseScaffoldFiles && typeof baseScaffoldFiles === "object" && !Array.isArray(baseScaffoldFiles)
      ? baseScaffoldFiles
      : {};
  const next = { ...base };
  next["src/engine/verdant-engine.ts"] = VERDANT_ENGINE_TS;
  next["src/components/VerdantViewport.tsx"] = VERDANT_VIEWPORT_TSX;
  next["src/pages/Index.tsx"] = VERDANT_INDEX_TSX;
  const baseAiRules = typeof base["AI_RULES.md"] === "string" ? base["AI_RULES.md"].trimEnd() : "";
  next["AI_RULES.md"] = `${baseAiRules}\n${VERDANT_AI_RULES_SUFFIX}`.trimStart();
  return next;
}
