"""
abstract flight -- seeded, audio-reactive Blender scene.

    blender -b -P abstract_flight.py -- --audio song.mp3 --seed 42 --out out.mp4
Options: --fps 30 --res 1920x1080 --samples 32 --save-blend scene.blend
         --frames N (only first N frames)  --still F (render frame F as PNG to --out)
         --no-render (only build the scene / save the .blend)
"""
import sys, os, math, random, colorsys, subprocess, argparse
import bpy
import numpy as np

SCENE_NAME = "abstract flight"


# ----------------------------------------------------------------------------- args
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser()
    p.add_argument("--audio", required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", default="render/abstract_flight.mp4")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--res", default="1920x1080")
    p.add_argument("--samples", type=int, default=32)
    p.add_argument("--save-blend", default=None)
    p.add_argument("--frames", type=int, default=0)
    p.add_argument("--still", type=int, default=None)
    p.add_argument("--no-render", action="store_true")
    return p.parse_args(argv)


# ----------------------------------------------------------------------------- audio
def load_audio(path, sr=22050):
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, check=True, stdout=subprocess.PIPE).stdout
    return np.frombuffer(raw, dtype=np.float32), sr


def smooth_env(x, attack, release):
    y = np.zeros_like(x)
    v = 0.0
    for i, s in enumerate(x):
        v = v + (s - v) * (attack if s > v else release)
        y[i] = v
    return y


def norm(x, lo=5, hi=98):
    a, b = np.percentile(x, lo), np.percentile(x, hi)
    return np.clip((x - a) / max(b - a, 1e-9), 0, 1.2)


def analyse(path, fps):
    sig, sr = load_audio(path)
    hop = sr / fps
    n = int(len(sig) / hop)
    win = 2048
    freqs = np.fft.rfftfreq(win, 1 / sr)
    w = np.hanning(win)
    bands = [(20, 150), (150, 2000), (2000, 10000)]
    masks = [(freqs >= a) & (freqs < b) for a, b in bands]
    E = np.zeros((n, 3))
    rms = np.zeros(n)
    flux = np.zeros(n)
    prev = None
    padded = np.concatenate([sig, np.zeros(win)])
    for i in range(n):
        c = int(i * hop)
        seg = padded[c:c + win]
        spec = np.abs(np.fft.rfft(seg * w))
        for k, m in enumerate(masks):
            E[i, k] = np.sqrt(np.mean(spec[m] ** 2))
        rms[i] = np.sqrt(np.mean(seg ** 2))
        lg = np.log1p(spec * 10)
        if prev is not None:
            flux[i] = np.sum(np.maximum(lg - prev, 0))
        prev = lg
    bass = smooth_env(norm(E[:, 0]), 0.7, 0.18)
    mid = smooth_env(norm(E[:, 1]), 0.6, 0.15)
    high = smooth_env(norm(E[:, 2]), 0.7, 0.2)
    loud = smooth_env(norm(rms), 0.4, 0.08)
    onset = norm(flux)
    # beat detection: peaks above adaptive threshold, min gap 0.18 s
    beat = np.zeros(n)
    gap = int(0.18 * fps)
    last = -gap
    wl = int(fps * 1.0)
    for i in range(n):
        lo, hi = max(0, i - wl), min(n, i + wl)
        thr = onset[lo:hi].mean() + 0.6 * onset[lo:hi].std()
        is_peak = onset[i] > thr and onset[i] >= onset[max(0, i - 1)] and onset[i] >= onset[min(n - 1, i + 1)]
        if is_peak and i - last >= gap:
            beat[i] = min(1.0, 0.5 + onset[i] * 0.5)
            last = i
    # flash envelope: instant attack, exponential decay (~0.2 s)
    env = np.zeros(n)
    v = 0.0
    dec = math.exp(-1.0 / (0.2 * fps))
    for i in range(n):
        v = max(v * dec, beat[i])
        env[i] = v
    return dict(bass=bass, mid=mid, high=high, loud=loud, onset=smooth_env(onset, 0.8, 0.3), beat=env, n=n)


# ----------------------------------------------------------------------------- seeded parameters
def hsv(h, s, v):
    return colorsys.hsv_to_rgb(h % 1.0, s, v)


def make_palette(rng, n=5):
    h = rng.random()
    scheme = rng.choice(["analog", "complement", "triad", "split"])
    offs = {"analog": [0, .06, .12, -.06, .18], "complement": [0, .03, .5, .53, .47],
            "triad": [0, .33, .66, .02, .35], "split": [0, .42, .58, .04, .46]}[scheme]
    cols = []
    for i in range(n):
        cols.append(hsv(h + offs[i % len(offs)] + rng.uniform(-.02, .02),
                        rng.uniform(.45, .95), rng.uniform(.25 + .1 * i, .65 + .07 * i)))
    return sorted(cols, key=lambda c: colorsys.rgb_to_hsv(*c)[2])


OBJ_KINDS = ["ico", "monolith", "spike", "ring", "pillar", "crystal"]
TEX_KINDS = ["noise", "voronoi", "wave", "checker"]


def make_params(seed):
    rng = random.Random(seed)
    P = {}
    P["seed"] = seed
    P["palette"] = make_palette(rng)
    P["horizon"] = hsv(rng.random(), rng.uniform(.2, .6), rng.uniform(.04, .16))
    # terrain shape
    P["amp"] = rng.uniform(45, 85)
    P["t_kind"] = rng.choice(["fbm", "ridged", "billow", "terraced"])
    P["t_scale"] = rng.uniform(0.004, 0.011)
    P["t_detail"] = rng.uniform(3, 7)
    P["t_rough"] = rng.uniform(.4, .6)
    P["t_pow"] = rng.uniform(1.0, 1.8)
    P["t_steps"] = rng.randint(4, 9)
    P["bass_gain"] = rng.uniform(.25, .7)
    P["valley_w"] = rng.uniform(14, 30)
    P["meander_a"] = rng.uniform(15, 55)
    P["meander_f"] = rng.uniform(.004, .012)
    P["meander_ph"] = rng.uniform(0, 6.28)
    P["offset"] = (rng.uniform(0, 1000), rng.uniform(0, 1000), rng.uniform(0, 1000))
    # terrain voxel behaviour: smooth / voxel / hybrid
    mode = rng.choice(["smooth", "voxel", "hybrid"])
    P["t_mode"] = mode
    if mode == "smooth":
        P["t_vbase"], P["t_vdepth"] = rng.uniform(0, .1), rng.uniform(.5, .9)
    elif mode == "voxel":
        P["t_vbase"], P["t_vdepth"] = rng.uniform(.75, .95), rng.uniform(-.3, -.1)
    else:
        P["t_vbase"], P["t_vdepth"] = rng.uniform(.25, .45), rng.uniform(.3, .6)
    P["t_vband"] = rng.choice(["bass", "mid", "loud"])
    P["t_vcell"] = rng.choice([3.0, 4.0, 5.0, 6.0])
    P["t_tex"] = rng.choice(TEX_KINDS)
    P["t_tex_scale"] = rng.uniform(0.02, 0.12)
    P["contour"] = rng.uniform(0.04, 0.15)
    # objects
    kinds = rng.sample(OBJ_KINDS, rng.randint(3, 6))
    P["obj_density"] = rng.uniform(.25, .6)
    P["obj_cell"] = 14.0
    P["objs"] = []
    for k in kinds:
        vm = rng.choice(["smooth", "voxel", "hybrid"])
        if vm == "smooth":
            vb, vd = rng.uniform(0, .08), rng.uniform(.5, .95)
        elif vm == "voxel":
            vb, vd = rng.uniform(.8, .95), rng.uniform(-.3, -.1)
        else:
            vb, vd = rng.uniform(.3, .5), rng.uniform(.3, .55)
        P["objs"].append(dict(
            kind=k, vmode=vm, vbase=vb, vdepth=vd, vband=rng.choice(["bass", "mid", "high", "loud"]),
            vsize=rng.uniform(.1, .3), smin=rng.uniform(1.2, 2.5), smax=rng.uniform(3, 6.5),
            hover=rng.choice([0, 0, rng.uniform(5, 25)]), pulse=rng.uniform(.25, .7),
            tex=rng.choice(TEX_KINDS), tex_scale=rng.uniform(.4, 3.0), palette=make_palette(rng),
            emis=rng.uniform(.05, .4), emis_gain=rng.uniform(1, 4), spin=rng.uniform(0, 1)))
    P["cam_alt"] = P["amp"] * 0.2 + 3
    P["speed"] = rng.uniform(26, 42)
    P["speed_bass"] = rng.uniform(.6, 1.4)
    P["sun_dir"] = (rng.uniform(.3, 1.1), rng.uniform(-.5, .5), rng.uniform(.5, .9))
    P["sun_col"] = hsv(rng.random(), rng.uniform(0, .4), 1.0)
    return P


def cx_of(P, y):
    return P["meander_a"] * math.sin(P["meander_f"] * y + P["meander_ph"])


# ----------------------------------------------------------------------------- node helpers
class NT:
    def __init__(self, tree):
        self.t = tree
        self.nodes = tree.nodes
        self.links = tree.links

    def _set(self, sock, v):
        if v is None:
            return
        if hasattr(v, "node"):
            self.links.new(v, sock)
        elif isinstance(v, (tuple, list)):
            sock.default_value = tuple(v) + ((1.0,) if len(sock.default_value) == 4 and len(v) == 3 else ())
        else:
            sock.default_value = v

    def node(self, idname, **props):
        n = self.nodes.new(idname)
        for k, v in props.items():
            setattr(n, k, v)
        return n

    def sock(self, node, key, outputs=False):
        coll = node.outputs if outputs else node.inputs
        if isinstance(key, int):
            return coll[key]
        for s in coll:
            if s.name == key and s.enabled:
                return s
        return coll[key]

    def make(self, idname, inputs=None, **props):
        n = self.node(idname, **props)
        for k, v in (inputs or {}).items():
            self._set(self.sock(n, k), v)
        return n

    def out(self, n, key=0):
        return self.sock(n, key, True)

    def m(self, op, a, b=None, c=None, clamp=False):
        n = self.node("ShaderNodeMath", operation=op, use_clamp=clamp)
        self._set(n.inputs[0], a)
        if b is not None:
            self._set(n.inputs[1], b)
        if c is not None:
            self._set(n.inputs[2], c)
        return n.outputs[0]

    def v(self, op, a, b=None, scale=None):
        n = self.node("ShaderNodeVectorMath", operation=op)
        self._set(n.inputs[0], a)
        if b is not None:
            self._set(n.inputs[1], b)
        if scale is not None:
            self._set(n.inputs[3], scale)
        return n.outputs[0]

    def rng(self, val, a, b, c, d, smooth=False, clamp=True):
        n = self.node("ShaderNodeMapRange", clamp=clamp,
                      interpolation_type='SMOOTHSTEP' if smooth else 'LINEAR')
        for i, x in enumerate([val, a, b, c, d]):
            self._set(n.inputs[i], x)
        return n.outputs[0]

    def lerp(self, a, b, t):
        return self.m("ADD", a, self.m("MULTIPLY", self.m("SUBTRACT", b, a), t))

    def vlerp(self, a, b, t):
        return self.v("ADD", a, self.v("SCALE", self.v("SUBTRACT", b, a), scale=t))

    def xyz(self, vec):
        n = self.node("ShaderNodeSeparateXYZ")
        self.links.new(vec, n.inputs[0])
        return n.outputs[0], n.outputs[1], n.outputs[2]

    def comb(self, x, y, z):
        n = self.node("ShaderNodeCombineXYZ")
        self._set(n.inputs[0], x)
        self._set(n.inputs[1], y)
        self._set(n.inputs[2], z)
        return n.outputs[0]


def new_gn(name):
    tree = bpy.data.node_groups.new(name, "GeometryNodeTree")
    tree.interface.new_socket("Geometry", in_out="OUTPUT", socket_type="NodeSocketGeometry")
    return tree


def audio_fields(nt, ctl_obj):
    """Object Info of the audio-control empty -> dict of float sockets."""
    info = nt.make("GeometryNodeObjectInfo", {"Object": ctl_obj})
    lx, ly, lz = nt.xyz(nt.out(info, "Location"))
    sx, sy, sz = nt.xyz(nt.out(info, "Scale"))
    return dict(bass=lx, mid=ly, high=lz, onset=sx, loud=sy, beat=sz)


# ----------------------------------------------------------------------------- height function
def height_nodes(nt, P, pos, bass):
    """Terrain height at world-space vector `pos` (float socket)."""
    x, y, _ = nt.xyz(pos)
    ox, oy, oz = P["offset"]
    sc = P["t_scale"]
    q = nt.comb(nt.m("ADD", nt.m("MULTIPLY", x, sc), ox), nt.m("ADD", nt.m("MULTIPLY", y, sc), oy), oz)
    noise = nt.make("ShaderNodeTexNoise", {"Vector": q, "Scale": 1.0, "Detail": P["t_detail"],
                                           "Roughness": P["t_rough"]}, noise_dimensions='3D')
    n = nt.out(noise, "Fac")
    k = P["t_kind"]
    if k == "ridged":
        f = nt.m("POWER", nt.m("SUBTRACT", 1.0, nt.m("ABSOLUTE", nt.m("SUBTRACT", nt.m("MULTIPLY", n, 2.0), 1.0))), 2.0)
        f = nt.rng(f, 0.0, 0.9, 0.0, 1.0)
    else:
        if k == "billow":
            n = nt.m("ADD", nt.m("MULTIPLY", nt.m("ABSOLUTE", nt.m("SUBTRACT", nt.m("MULTIPLY", n, 2.0), 1.0)), 0.9), 0.0)
            f = nt.rng(n, 0.0, 0.7, 0.0, 1.0)
        else:
            f = nt.rng(n, 0.3, 0.7, 0.0, 1.0)
        if k == "terraced":
            st = nt.m("SNAP", f, 1.0 / P["t_steps"])
            f = nt.lerp(f, st, 0.85)
    f = nt.m("POWER", f, P["t_pow"])
    # large-scale mask so some regions are flat
    q2 = nt.comb(nt.m("ADD", nt.m("MULTIPLY", x, sc * 0.3), oy), nt.m("ADD", nt.m("MULTIPLY", y, sc * 0.3), ox), oz)
    mk = nt.out(nt.make("ShaderNodeTexNoise", {"Vector": q2, "Scale": 1.0, "Detail": 2.0}, noise_dimensions='3D'), "Fac")
    f = nt.m("MULTIPLY", f, nt.rng(mk, 0.3, 0.65, 0.45, 1.0))
    # flight corridor
    cx = nt.m("MULTIPLY", nt.m("SINE", nt.m("ADD", nt.m("MULTIPLY", y, P["meander_f"]), P["meander_ph"])), P["meander_a"])
    d = nt.m("ABSOLUTE", nt.m("SUBTRACT", x, cx))
    mask = nt.rng(d, P["valley_w"], P["valley_w"] * 3.2, 0.04, 1.0, smooth=True)
    amp = nt.m("MULTIPLY", P["amp"], nt.m("ADD", 1.0, nt.m("MULTIPLY", bass, P["bass_gain"])))
    return nt.m("MULTIPLY", nt.m("MULTIPLY", f, mask), amp), cx, d


# ----------------------------------------------------------------------------- materials
def ramp_node(nt, colors):
    r = nt.node("ShaderNodeValToRGB")
    els = r.color_ramp.elements
    while len(els) > 2:
        els.remove(els[-1])
    n = len(colors)
    els[0].position = 0.0
    els[1].position = 1.0
    for i in range(n - 2):
        els.new((i + 1) / (n - 1))
    for i, c in enumerate(colors):
        els[i].position = i / (n - 1)
        els[i].color = (*c, 1.0)
    return r


def add_driver(socket, ctl, transform):
    fc = socket.driver_add("default_value")
    d = fc.driver
    d.type = "AVERAGE"
    var = d.variables.new()
    var.type = "TRANSFORMS"
    var.targets[0].id = ctl
    var.targets[0].transform_type = transform
    var.targets[0].transform_space = "WORLD_SPACE"


def mat_audio(nt, ctl):
    ch = dict(bass="LOC_X", mid="LOC_Y", high="LOC_Z", onset="SCALE_X", loud="SCALE_Y", beat="SCALE_Z")
    out = {}
    for k, tr in ch.items():
        v = nt.node("ShaderNodeValue")
        v.label = "A_" + k
        add_driver(v.outputs[0], ctl, tr)
        out[k] = v.outputs[0]
    return out


def tex_factor(nt, kind, coord, scale):
    if kind == "voronoi":
        t = nt.make("ShaderNodeTexVoronoi", {"Vector": coord, "Scale": scale}, feature='DISTANCE_TO_EDGE')
        return nt.rng(nt.out(t, "Distance"), 0.0, 0.25, 1.0, 0.0)
    if kind == "wave":
        t = nt.make("ShaderNodeTexWave", {"Vector": coord, "Scale": scale, "Distortion": 6.0, "Detail": 2.0},
                    wave_type='BANDS')
        return nt.out(t, "Fac")
    if kind == "checker":
        t = nt.make("ShaderNodeTexChecker", {"Vector": coord, "Scale": scale * 2})
        return nt.out(t, "Fac")
    t = nt.make("ShaderNodeTexNoise", {"Vector": coord, "Scale": scale, "Detail": 6.0, "Roughness": .6}, noise_dimensions='3D')
    return nt.out(t, "Fac")


def make_material(name, ctl, palette, tex, tex_scale, horizon, emis, emis_gain, terrain=None):
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    tree = mat.node_tree
    tree.nodes.clear()
    nt = NT(tree)
    A = mat_audio(nt, ctl)
    geo = nt.node("ShaderNodeNewGeometry")
    pos = nt.out(geo, "Position")
    px, py, pz = nt.xyz(pos)
    tf = tex_factor(nt, tex, pos, tex_scale)
    if terrain:
        hnorm = nt.rng(pz, 0.0, terrain["amp"] * 0.8, 0.0, 1.0)
        base = nt.m("ADD", nt.m("MULTIPLY", hnorm, 0.8), nt.m("MULTIPLY", tf, 0.2))
        # contour lines
        cl = nt.m("PINGPONG", nt.m("MULTIPLY", pz, terrain["contour"]), 1.0)
        line = nt.rng(cl, 0.04, 0.1, 1.0, 0.0)
        glow = nt.m("MULTIPLY", line, nt.m("ADD", nt.m("MULTIPLY", A["high"], 1.2), nt.m("MULTIPLY", A["beat"], 1.5)))
        glow = nt.m("ADD", nt.m("MULTIPLY", glow, 0.6), nt.m("MULTIPLY", A["beat"], 0.12))
    else:
        base = tf
        glow = nt.m("ADD", nt.m("MULTIPLY", nt.m("MULTIPLY", tf, tf), nt.m("MULTIPLY", A["high"], emis_gain * 0.5)),
                    nt.m("ADD", emis * 0.5, nt.m("MULTIPLY", A["beat"], emis_gain * 0.4)))
    ramp = ramp_node(nt, palette)
    nt.links.new(base, ramp.inputs[0])
    col = ramp.outputs[0]
    cam = nt.node("ShaderNodeCameraData")
    fade = nt.rng(nt.out(cam, "View Z Depth"), 140.0, 330.0, 0.0, 1.0, smooth=True)
    pr = nt.make("ShaderNodeBsdfPrincipled", {"Base Color": col, "Roughness": 0.55,
                                              "Emission Color": col, "Emission Strength": glow})
    hz = nt.make("ShaderNodeEmission", {"Color": (*horizon, 1.0), "Strength": 1.0})
    mix = nt.make("ShaderNodeMixShader", {"Fac": fade})
    nt.links.new(pr.outputs[0], mix.inputs[1])
    nt.links.new(hz.outputs[0], mix.inputs[2])
    o = nt.node("ShaderNodeOutputMaterial")
    nt.links.new(mix.outputs[0], o.inputs[0])
    return mat


# ----------------------------------------------------------------------------- geometry groups
def build_terrain(P, ctl, cam, mat):
    tree = new_gn("AF_Terrain")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    nt = NT(tree)
    gin = nt.node("NodeGroupInput")
    gout = nt.node("NodeGroupOutput")
    A = audio_fields(nt, ctl)
    caminfo = nt.make("GeometryNodeObjectInfo", {"Object": cam})
    cpx, cpy, _ = nt.xyz(nt.out(caminfo, "Location"))
    bandv = A[P["t_vband"]]
    vox = nt.m("MAXIMUM", 0.0, nt.m("MINIMUM", 1.0, nt.m("ADD", P["t_vbase"], nt.m("MULTIPLY", bandv, P["t_vdepth"]))))
    vcell = P["t_vcell"]
    cell = 2.0
    W, L = 360.0, 400.0

    def lattice_offset(cell_):
        return nt.comb(nt.m("SNAP", cpx, cell_), nt.m("ADD", nt.m("SNAP", cpy, cell_), L * 0.4), 0.0)

    # --- smooth mesh
    grid = nt.make("GeometryNodeMeshGrid", {"Size X": W, "Size Y": L, "Vertices X": int(W / cell) + 1,
                                            "Vertices Y": int(L / cell) + 1})
    tr = nt.make("GeometryNodeTransform", {"Translation": lattice_offset(cell)})
    nt.links.new(nt.out(grid, "Mesh"), tr.inputs["Geometry"])
    pos = nt.out(nt.node("GeometryNodeInputPosition"))
    h, _, _ = height_nodes(nt, P, pos, A["bass"])
    snapped = nt.m("SNAP", h, vcell * 0.8)
    px, py, _ = nt.xyz(pos)
    z = nt.m("SUBTRACT", nt.lerp(h, snapped, vox), 0.06)
    sp = nt.make("GeometryNodeSetPosition", {"Position": nt.comb(px, py, z)})
    nt.links.new(tr.outputs["Geometry"], sp.inputs["Geometry"])
    smat = nt.make("GeometryNodeSetMaterial", {"Material": mat})
    nt.links.new(sp.outputs["Geometry"], smat.inputs["Geometry"])

    # --- voxel columns
    cw, cl_ = 300.0, 360.0
    vg = nt.make("GeometryNodeMeshGrid", {"Size X": cw, "Size Y": cl_, "Vertices X": int(cw / vcell) + 1,
                                          "Vertices Y": int(cl_ / vcell) + 1})
    vt = nt.make("GeometryNodeTransform", {"Translation": lattice_offset(vcell)})
    nt.links.new(nt.out(vg, "Mesh"), vt.inputs["Geometry"])
    pts = nt.make("GeometryNodeMeshToPoints", {"Radius": 0.1})
    nt.links.new(vt.outputs["Geometry"], pts.inputs["Mesh"])
    vpos = nt.out(nt.node("GeometryNodeInputPosition"))
    vh, _, _ = height_nodes(nt, P, vpos, A["bass"])
    top = nt.m("MAXIMUM", nt.m("SNAP", vh, vcell * 0.8), 0.5)
    vx, vy, _ = nt.xyz(vpos)
    ipos = nt.comb(vx, vy, nt.m("MULTIPLY", nt.m("SUBTRACT", top, 8.0), 0.5))
    sp2 = nt.make("GeometryNodeSetPosition", {"Position": ipos})
    nt.links.new(pts.outputs["Points"], sp2.inputs["Geometry"])
    cube = nt.make("GeometryNodeMeshCube", {"Size": (1.0, 1.0, 1.0)})
    sm = nt.rng(vox, 0.12, 0.5, 0.0, 1.0, smooth=True)
    wxy = nt.m("MULTIPLY", sm, vcell * 0.96)
    sc = nt.comb(wxy, wxy, nt.m("ADD", top, 8.0))
    iop = nt.make("GeometryNodeInstanceOnPoints", {"Scale": sc, "Selection": nt.m("GREATER_THAN", vox, 0.12)})
    nt.links.new(sp2.outputs["Geometry"], iop.inputs["Points"])
    nt.links.new(nt.out(cube, "Mesh"), iop.inputs["Instance"])
    vmat = nt.make("GeometryNodeSetMaterial", {"Material": mat})
    nt.links.new(iop.outputs["Instances"], vmat.inputs["Geometry"])

    join = nt.node("GeometryNodeJoinGeometry")
    nt.links.new(smat.outputs["Geometry"], join.inputs[0])
    nt.links.new(vmat.outputs["Geometry"], join.inputs[0])
    nt.links.new(join.outputs[0], gout.inputs[0])
    return tree


def object_mesh(nt, kind, vox, vsize):
    """Return (geometry socket) of a unit-ish mesh with vertex snapping blend `vox`."""
    if kind == "ico":
        g = nt.out(nt.make("GeometryNodeMeshIcoSphere", {"Radius": 1.0, "Subdivisions": 4}), "Mesh")
    elif kind == "monolith":
        c = nt.out(nt.make("GeometryNodeMeshCube", {"Size": (0.7, 0.7, 3.0)}), "Mesh")
        g = nt.out(nt.make("GeometryNodeSubdivideMesh", {"Mesh": c, "Level": 4}), "Mesh")
    elif kind == "spike":
        c = nt.out(nt.make("GeometryNodeMeshCone", {"Radius Bottom": 0.8, "Radius Top": 0.0, "Depth": 3.5,
                                                    "Vertices": 24, "Side Segments": 24}), "Mesh")
        g = c
    elif kind == "pillar":
        c = nt.out(nt.make("GeometryNodeMeshCylinder", {"Radius": 0.6, "Depth": 3.0, "Vertices": 24,
                                                         "Side Segments": 24}), "Mesh")
        g = c
    elif kind == "crystal":
        c = nt.out(nt.make("GeometryNodeMeshIcoSphere", {"Radius": 1.0, "Subdivisions": 2}), "Mesh")
        g = nt.out(nt.make("GeometryNodeTransform", {"Geometry": c, "Scale": (0.6, 0.6, 2.2)}), "Geometry")
    else:  # ring
        circ = nt.out(nt.make("GeometryNodeCurvePrimitiveCircle", {"Radius": 1.4, "Resolution": 64}), "Curve")
        prof = nt.out(nt.make("GeometryNodeCurvePrimitiveCircle", {"Radius": 0.35, "Resolution": 20}), "Curve")
        g = nt.out(nt.make("GeometryNodeCurveToMesh", {"Curve": circ, "Profile Curve": prof}), "Mesh")
        g = nt.out(nt.make("GeometryNodeTransform", {"Geometry": g, "Rotation": (math.pi / 2, 0, 0)}), "Geometry")
    p = nt.out(nt.node("GeometryNodeInputPosition"))
    snapped = nt.v("SNAP", p, (vsize, vsize, vsize))
    sp = nt.make("GeometryNodeSetPosition", {"Geometry": g, "Position": nt.vlerp(p, snapped, vox)})
    flat = nt.make("GeometryNodeSetShadeSmooth", {"Geometry": nt.out(sp, "Geometry"),
                                                  "Shade Smooth": False})
    return nt.out(flat, "Geometry")


def build_objects(P, ctl, cam, mats):
    tree = new_gn("AF_Objects")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    nt = NT(tree)
    nt.node("NodeGroupInput")
    gout = nt.node("NodeGroupOutput")
    A = audio_fields(nt, ctl)
    caminfo = nt.make("GeometryNodeObjectInfo", {"Object": cam})
    cpx, cpy, _ = nt.xyz(nt.out(caminfo, "Location"))
    cell = P["obj_cell"]
    W, L = 336.0, 392.0
    grid = nt.make("GeometryNodeMeshGrid", {"Size X": W, "Size Y": L, "Vertices X": int(W / cell) + 1,
                                            "Vertices Y": int(L / cell) + 1})
    off = nt.comb(nt.m("SNAP", cpx, cell), nt.m("ADD", nt.m("SNAP", cpy, cell), L * 0.4), 0.0)
    tr = nt.make("GeometryNodeTransform", {"Translation": off})
    nt.links.new(nt.out(grid, "Mesh"), tr.inputs["Geometry"])
    pts = nt.make("GeometryNodeMeshToPoints", {"Radius": 0.1})
    nt.links.new(tr.outputs["Geometry"], pts.inputs["Mesh"])
    pos = nt.out(nt.node("GeometryNodeInputPosition"))
    wn = nt.make("ShaderNodeTexWhiteNoise", {"Vector": pos, "W": float(P["seed"] % 997)}, noise_dimensions='4D')
    r1, r2, r3 = nt.xyz(nt.out(wn, "Color"))
    wn2 = nt.make("ShaderNodeTexWhiteNoise", {"Vector": pos, "W": float(P["seed"] % 997) + 7.5}, noise_dimensions='4D')
    r4, r5, r6 = nt.xyz(nt.out(wn2, "Color"))
    px, py, _ = nt.xyz(pos)
    jx = nt.m("ADD", px, nt.m("MULTIPLY", nt.m("SUBTRACT", r1, 0.5), cell * 0.85))
    jy = nt.m("ADD", py, nt.m("MULTIPLY", nt.m("SUBTRACT", r2, 0.5), cell * 0.85))
    jpos = nt.comb(jx, jy, 0.0)
    h, cx, d = height_nodes(nt, P, jpos, A["bass"])
    n_types = len(P["objs"])
    maxhover = max(o["hover"] for o in P["objs"])
    # per-point hover amount is per type; use average hover here with type-specific adjust below
    sp = nt.make("GeometryNodeSetPosition", {"Position": nt.comb(jx, jy, nt.m("ADD", h, nt.m("MULTIPLY", r3, maxhover)))})
    nt.links.new(pts.outputs["Points"], sp.inputs["Geometry"])
    # spawn: grows in as it approaches
    spawn = nt.rng(nt.m("SUBTRACT", jy, cpy), 330.0, 270.0, 0.0, 1.0, smooth=True)
    # keep out of the flight corridor
    cxm = nt.m("MULTIPLY", nt.m("SINE", nt.m("ADD", nt.m("MULTIPLY", jy, P["meander_f"]), P["meander_ph"])), P["meander_a"])
    dd = nt.m("ABSOLUTE", nt.m("SUBTRACT", jx, cxm))
    clear = nt.m("GREATER_THAN", dd, 17.0)
    present = nt.m("MULTIPLY", clear, nt.m("LESS_THAN", r5, P["obj_density"]))
    tyidx = nt.m("FLOOR", nt.m("MULTIPLY", nt.m("MINIMUM", r4, 0.9999), float(n_types)))
    join = nt.node("GeometryNodeJoinGeometry")
    for i, o in enumerate(P["objs"]):
        band = A[o["vband"]]
        vox = nt.m("MAXIMUM", 0.0, nt.m("MINIMUM", 1.0, nt.m("ADD", o["vbase"], nt.m("MULTIPLY", band, o["vdepth"]))))
        geo = object_mesh(nt, o["kind"], vox, o["vsize"])
        sm = nt.make("GeometryNodeSetMaterial", {"Material": mats[i], "Geometry": geo})
        sel = nt.m("MULTIPLY", present, nt.m("COMPARE", tyidx, float(i)))
        size = nt.m("MULTIPLY", nt.m("ADD", o["smin"], nt.m("MULTIPLY", r6, o["smax"] - o["smin"])),
                    nt.m("MULTIPLY", spawn, nt.m("ADD", 1.0, nt.m("MULTIPLY", A["beat"], o["pulse"]))))
        rot = nt.comb(nt.m("MULTIPLY", r1, 6.283 * o["spin"]), nt.m("MULTIPLY", r2, 6.283 * o["spin"]),
                      nt.m("MULTIPLY", r5, 6.283))
        scl = nt.comb(size, size, size)
        hover_adj = nt.comb(0.0, 0.0, nt.m("MULTIPLY", nt.m("SUBTRACT", o["hover"], nt.m("MULTIPLY", r3, maxhover)), 1.0))
        iop = nt.make("GeometryNodeInstanceOnPoints", {"Selection": sel, "Rotation": rot, "Scale": scl})
        # shift the points vertically for this type's own hover
        shift = nt.make("GeometryNodeSetPosition", {"Offset": hover_adj})
        nt.links.new(sp.outputs["Geometry"], shift.inputs["Geometry"])
        nt.links.new(shift.outputs["Geometry"], iop.inputs["Points"])
        nt.links.new(sm.outputs["Geometry"], iop.inputs["Instance"])
        # sink ground-based types so they sit on the terrain
        join_inp = iop.outputs["Instances"]
        nt.links.new(join_inp, join.inputs[0])
    nt.links.new(join.outputs[0], gout.inputs[0])
    return tree


def build_dust(P, ctl, cam, mat):
    tree = new_gn("AF_Dust")
    tree.interface.new_socket("Geometry", in_out="INPUT", socket_type="NodeSocketGeometry")
    nt = NT(tree)
    nt.node("NodeGroupInput")
    gout = nt.node("NodeGroupOutput")
    A = audio_fields(nt, ctl)
    caminfo = nt.make("GeometryNodeObjectInfo", {"Object": cam})
    cpx, cpy, _ = nt.xyz(nt.out(caminfo, "Location"))
    cell, W, L = 8.0, 160.0, 240.0
    grid = nt.make("GeometryNodeMeshGrid", {"Size X": W, "Size Y": L, "Vertices X": int(W / cell) + 1,
                                            "Vertices Y": int(L / cell) + 1})
    off = nt.comb(nt.m("SNAP", cpx, cell), nt.m("ADD", nt.m("SNAP", cpy, cell), L * 0.4), 0.0)
    tr = nt.make("GeometryNodeTransform", {"Translation": off})
    nt.links.new(nt.out(grid, "Mesh"), tr.inputs["Geometry"])
    pts = nt.make("GeometryNodeMeshToPoints", {"Radius": 0.1})
    nt.links.new(tr.outputs["Geometry"], pts.inputs["Mesh"])
    pos = nt.out(nt.node("GeometryNodeInputPosition"))
    wn = nt.make("ShaderNodeTexWhiteNoise", {"Vector": pos, "W": float(P["seed"] % 997) + 3.1}, noise_dimensions='4D')
    r1, r2, r3 = nt.xyz(nt.out(wn, "Color"))
    px, py, _ = nt.xyz(pos)
    npos = nt.comb(nt.m("ADD", px, nt.m("MULTIPLY", nt.m("SUBTRACT", r1, 0.5), cell)),
                   nt.m("ADD", py, nt.m("MULTIPLY", nt.m("SUBTRACT", r2, 0.5), cell)),
                   nt.m("ADD", 2.0, nt.m("MULTIPLY", r3, P["cam_alt"] * 3.5)))
    sp = nt.make("GeometryNodeSetPosition", {"Position": npos})
    nt.links.new(pts.outputs["Points"], sp.inputs["Geometry"])
    ico = nt.make("GeometryNodeMeshIcoSphere", {"Radius": 0.12, "Subdivisions": 1})
    s = nt.m("MULTIPLY", nt.m("ADD", 0.5, r3), nt.m("ADD", 1.0, nt.m("MULTIPLY", A["high"], 2.5)))
    iop = nt.make("GeometryNodeInstanceOnPoints", {"Scale": nt.comb(s, s, s)})
    nt.links.new(sp.outputs["Geometry"], iop.inputs["Points"])
    nt.links.new(nt.out(ico, "Mesh"), iop.inputs["Instance"])
    sm = nt.make("GeometryNodeSetMaterial", {"Material": mat})
    nt.links.new(iop.outputs["Instances"], sm.inputs["Geometry"])
    nt.links.new(sm.outputs["Geometry"], gout.inputs[0])
    return tree


def dust_material(P, ctl):
    mat = bpy.data.materials.new("AF_Dust")
    mat.use_nodes = True
    t = mat.node_tree
    t.nodes.clear()
    nt = NT(t)
    A = mat_audio(nt, ctl)
    c = P["palette"][-1]
    em = nt.make("ShaderNodeEmission", {"Color": (*c, 1.0),
                                        "Strength": nt.m("ADD", 1.5, nt.m("MULTIPLY", A["high"], 6.0))})
    o = nt.node("ShaderNodeOutputMaterial")
    nt.links.new(em.outputs[0], o.inputs[0])
    return mat


def add_gn_object(name, tree):
    me = bpy.data.meshes.new(name)
    ob = bpy.data.objects.new(name, me)
    bpy.context.scene.collection.objects.link(ob)
    mod = ob.modifiers.new("GN", "NODES")
    mod.node_group = tree
    return ob


# ----------------------------------------------------------------------------- animation baking
def key_all(obj, channels, frames):
    """channels: {(path, idx): array}. Inserts per-frame keys."""
    for (path, idx), arr in channels.items():
        for f, v in zip(frames, arr):
            vec = list(getattr(obj, path))
            vec[idx] = float(v)
            setattr(obj, path, vec)
            obj.keyframe_insert(path, index=idx, frame=f)


def key_scalar(owner, path, frames, arr):
    for f, v in zip(frames, arr):
        setattr(owner, path, float(v))
        owner.keyframe_insert(path, frame=f)


# ----------------------------------------------------------------------------- main
def build_scene(args, feats, P):
    # fresh scene
    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.data.scenes[0]
    scene.name = SCENE_NAME
    bpy.context.window.scene = scene if bpy.context.window else None
    fps = args.fps
    n = feats["n"]
    if args.frames:
        n = min(n, args.frames)
    scene.render.fps = fps
    scene.frame_start, scene.frame_end = 1, n
    w, h = [int(v) for v in args.res.lower().split("x")]
    scene.render.resolution_x, scene.render.resolution_y = w, h
    scene.render.resolution_percentage = 100

    # control empty
    ctl = bpy.data.objects.new("AF_AudioCtl", None)
    scene.collection.objects.link(ctl)
    ctl.empty_display_type = "SPHERE"
    ctl.empty_display_size = 0.2
    cam_data = bpy.data.cameras.new("AF_Camera")
    cam_data.clip_end = 1500
    cam = bpy.data.objects.new("AF_Camera", cam_data)
    scene.collection.objects.link(cam)
    scene.camera = cam

    frames = list(range(1, n + 1))
    f = feats
    sl = slice(0, n)
    # audio channels -> empty transforms
    ctl.location = (0, 0, 0)
    key_all(ctl, {("location", 0): f["bass"][sl], ("location", 1): f["mid"][sl], ("location", 2): f["high"][sl],
                  ("scale", 0): f["onset"][sl], ("scale", 1): f["loud"][sl], ("scale", 2): f["beat"][sl]}, frames)

    # camera path (speed follows bass)
    rngc = random.Random(P["seed"] * 7 + 1)
    y = 0.0
    ys = []
    for i in range(n):
        ys.append(y)
        y += (P["speed"] * (0.6 + P["speed_bass"] * f["bass"][i] + 0.3 * f["loud"][i])) / fps
    ys = np.array(ys)
    cxs = np.array([cx_of(P, yy) for yy in ys])
    dcx = np.gradient(cxs, ys + 1e-6 * np.arange(n))
    sway = smooth_env(np.array([math.sin(i / fps * 0.55 + P["seed"]) for i in range(n)]), 0.05, 0.05)
    xs = cxs + sway * 5
    zs = P["cam_alt"] + 2.2 * np.sin(np.arange(n) / fps * 0.4 + P["seed"] * 0.3) + f["bass"][sl] * 1.0
    yaw = -np.arctan(dcx)
    bank = np.clip(-np.gradient(dcx) * 150.0, -0.15, 0.15) + 0.03 * sway
    shake = (f["onset"][sl] * 0.006) * np.sin(np.arange(n) * 2.1)
    pitch = 0.04 * np.sin(np.arange(n) / fps * 0.3) - 0.02
    key_all(cam, {("location", 0): xs, ("location", 1): ys, ("location", 2): zs,
                  ("rotation_euler", 0): math.pi / 2 + pitch + shake,
                  ("rotation_euler", 1): bank,
                  ("rotation_euler", 2): yaw}, frames)
    cam_data.lens = 24
    key_scalar(cam_data, "lens", frames, 22 - 8 * np.clip(f["loud"][sl], 0, 1) * 1.0 + 8 * 0.4)

    # materials
    tmat = make_material("AF_Terrain", ctl, P["palette"], P["t_tex"], P["t_tex_scale"], P["horizon"], 0.0, 0.0,
                         terrain=dict(amp=P["amp"], contour=P["contour"]))
    omats = [make_material("AF_Obj_%d_%s" % (i, o["kind"]), ctl, o["palette"], o["tex"], o["tex_scale"], P["horizon"],
                           o["emis"], o["emis_gain"]) for i, o in enumerate(P["objs"])]
    dmat = dust_material(P, ctl)

    # geometry
    add_gn_object("AF_Terrain", build_terrain(P, ctl, cam, tmat))
    add_gn_object("AF_Objects", build_objects(P, ctl, cam, omats))
    add_gn_object("AF_Dust", build_dust(P, ctl, cam, dmat))

    # lighting / world
    sun_data = bpy.data.lights.new("AF_Sun", "SUN")
    sun_data.energy = 1.2
    sun_data.color = P["sun_col"]
    sun = bpy.data.objects.new("AF_Sun", sun_data)
    scene.collection.objects.link(sun)
    sd = P["sun_dir"]
    sun.rotation_euler = (math.radians(50 + 20 * sd[2]), 0, math.radians(sd[0] * 80 - 40))
    world = bpy.data.worlds.new("AF_World")
    world.use_nodes = True
    bg = world.node_tree.nodes["Background"]
    bg.inputs[0].default_value = (*P["horizon"], 1.0)
    bg.inputs[1].default_value = 1.0
    scene.world = world

    # render settings
    try:
        scene.render.engine = "BLENDER_EEVEE"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    try:
        scene.eevee.taa_render_samples = args.samples
    except Exception:
        pass
    scene.view_settings.view_transform = "Standard"
    try:
        make_compositor(scene)
    except Exception as e:
        print("[abstract flight] compositor glare skipped:", e)

    # audio strip
    seq = scene.sequence_editor_create()
    coll = seq.strips if hasattr(seq, "strips") else seq.sequences
    coll.new_sound("audio", os.path.abspath(args.audio), 1, 1)
    return scene


def make_compositor(scene):
    ng = bpy.data.node_groups.new("AF Compositor", "CompositorNodeTree")
    ng.interface.new_socket("Image", in_out="OUTPUT", socket_type="NodeSocketColor")
    rl = ng.nodes.new("CompositorNodeRLayers")
    gl = ng.nodes.new("CompositorNodeGlare")
    out = ng.nodes.new("NodeGroupOutput")
    ng.links.new(rl.outputs["Image"], gl.inputs["Image"])
    ng.links.new(gl.outputs["Image"], out.inputs[0])
    for k, v in (("glare_type", "BLOOM"), ("quality", "MEDIUM")):
        try:
            setattr(gl, k, v)
        except Exception:
            pass
    for name, val in (("Threshold", 1.5), ("Strength", 0.5), ("Size", 0.5), ("Mix", 0.0)):
        if name in gl.inputs:
            gl.inputs[name].default_value = val
    scene.compositing_node_group = ng


def main():
    args = parse_args()
    P = make_params(args.seed)
    print("[abstract flight] seed", args.seed, "terrain:", P["t_kind"], P["t_mode"],
          "objects:", [(o["kind"], o["vmode"], o["tex"]) for o in P["objs"]])
    feats = analyse(args.audio, args.fps)
    print("[abstract flight] frames:", feats["n"], "beats:", int((feats["beat"] > 0.99).sum()))
    scene = build_scene(args, feats, P)
    if args.save_blend:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_blend)), exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=os.path.abspath(args.save_blend))
    if args.no_render:
        return
    r = scene.render
    out = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    if args.still is not None:
        scene.frame_set(args.still)
        try:
            r.image_settings.media_type = "IMAGE"
        except Exception:
            pass
        r.image_settings.file_format = "PNG"
        r.filepath = out
        bpy.ops.render.render(write_still=True)
    else:
        try:
            r.image_settings.media_type = "VIDEO"
        except Exception:
            pass
        r.image_settings.file_format = "FFMPEG"
        r.ffmpeg.format = "MPEG4"
        r.ffmpeg.codec = "H264"
        r.ffmpeg.constant_rate_factor = "HIGH"
        r.ffmpeg.audio_codec = "AAC"
        r.ffmpeg.audio_bitrate = 192
        r.filepath = out
        bpy.ops.render.render(animation=True)


main()
