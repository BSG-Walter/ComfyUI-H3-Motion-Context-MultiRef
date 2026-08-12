import { app } from "../../scripts/app.js";
import { api } from "../../../scripts/api.js";

const NODE_NAME = "MiniMaxH3Timeline";
const STATE_NAME = "timeline_state";
const MAX_CLIPS = 32;

const SPAN = 240; // ruler length in frames
const PX = 3.5; // pixels per frame (base zoom)
const ZOOM_MIN = 2; // px/frame limits for zoom buttons
const ZOOM_MAX = 24;
const ZOOM_STEP = 1.25;
const SNAP_PX = 12; // magnet radius for clip snapping
const SNAP_PLAY_PX = 24; // wider magnet radius when clips snap to the playhead
const OFFSET_X = 16; // left margin offset
const BTN_W = 16;
const BTN_H = 12;
const RULER_H = 18;
const LANE_H = 34;
const PAD = 4;
const WIDTH = 840;
const HEIGHT = RULER_H + 2 * LANE_H + PAD;
const TOOL_X = WIDTH - 124; // first toolbar column

const COLORS = {
    image: "#7aa2f7",
    video: "#9ece6a",
    audio: "#f7768e",
};
const PLAY_COLOR = "#ff5252";

const defaults = {
    image: () => ({ kind: "image", start: 1, strength: 1 }),
    video: () => ({
        kind: "video",
        start: 1,
        strength: 1,
        len: 22,
        audio_link: true,
        audio_start: null,
        audio_len: null,
        audio_align: "head",
    }),
    audio: () => ({ kind: "audio", start: 1, strength: 1, len: 22, align: "head" }),
};

// --- uploaded media helpers ------------------------------------------------

const IMAGE_EXT = ["png", "jpg", "jpeg", "webp", "bmp", "gif"];
const VIDEO_EXT = ["mp4", "mov", "webm", "mkv", "m4v", "avi"];
const AUDIO_EXT = ["mp3", "wav", "flac", "m4a", "aac", "ogg", "opus", "wma"];

function kindOfFile(m) {
    const ext = (m?.name || "").split(".").pop().toLowerCase();
    if (IMAGE_EXT.includes(ext)) return "image";
    if (VIDEO_EXT.includes(ext)) return "video";
    if (AUDIO_EXT.includes(ext)) return "audio";
    return null;
}

function mediaKey(m) {
    return (m?.subfolder ? m.subfolder + "/" : "") + (m?.name || "");
}

function mediaURL(m) {
    return api.apiURL(
        "/view?" +
            new URLSearchParams({
                filename: m.name,
                type: m.type || "input",
                subfolder: m.subfolder || "",
            }),
    );
}

async function uploadMedia(file) {
    const fd = new FormData();
    fd.append("image", file);
    fd.append("type", "input");
    fd.append("overwrite", "true");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body: fd });
    if (!resp.ok) throw new Error("upload failed: " + resp.status);
    const info = await resp.json();
    return { name: info.name, subfolder: info.subfolder || "", type: info.type || "input" };
}

function pickFile() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*,video/*,audio/*";
    return new Promise((resolve) => {
        input.onchange = () => resolve(input.files?.[0] || null);
        input.click();
    });
}

async function replaceClipMedia(node, c) {
    const file = await pickFile();
    if (!file) return;
    const info = await uploadMedia(file);
    const kind = kindOfFile(info);
    if (kind) c.kind = kind;
    c.file = info;
    c.src_start = 0;
    ensureInputs(node);
    writeState(node);
    fixNodeSize(node);
}

function computePeaks(buf) {
    const data = buf.getChannelData(0);
    const n = 128;
    const peaks = [];
    const step = Math.max(1, Math.floor(data.length / n));
    for (let i = 0; i < n; i++) {
        const s = i * step;
        const e = Math.min(data.length, s + step);
        let mn = 1;
        let mx = -1;
        for (let j = s; j < e; j++) {
            const v = data[j];
            if (v < mn) mn = v;
            if (v > mx) mx = v;
        }
        peaks.push((mn + mx) / 2, Math.max(0.02, mx - mn));
    }
    return peaks; // [mid, amp] pairs
}

function redrawNode(node) {
    const w = node._h3TimelineWidget;
    if (w) {
        w.triggerDraw?.();
        w.redraw?.(node);
    }
}

function ensureMedia(node, c) {
    if (!c?.file) return null;
    const key = mediaKey(c.file);
    let m = node._h3Media.get(key);
    if (!m) {
        m = {
            kind: kindOfFile(c.file) || c.kind,
            url: mediaURL(c.file),
            loaded: false,
            ready: null,
        };
        node._h3Media.set(key, m);
        m.ready = loadMedia(node, c, m);
    }
    return m;
}

function loadMedia(node, _c, m) {
    if (m.kind === "image") {
        const img = new Image();
        img.onload = () => {
            m.img = img;
            m.loaded = true;
            redrawNode(node);
        };
        img.src = m.url;
        return new Promise((ok) => {
            img.decode?.().then(ok).catch(ok);
        });
    }
    if (m.kind === "audio") {
        return fetch(m.url)
            .then((r) => r.arrayBuffer())
            .then((buf) => {
                const actx =
                    node._h3AudioCtx ??
                    (node._h3AudioCtx =
                        new (window.AudioContext || window.webkitAudioContext)());
                return actx.decodeAudioData(buf);
            })
            .then((decoded) => {
                m.buffer = decoded;
                m.peaks = computePeaks(decoded);
                m.loaded = true;
                redrawNode(node);
            })
            .catch(() => {});
    }
    return Promise.resolve(); // video: seeked frames drawn on demand
}

// one shared <video> per node serves thumbnails and the playhead preview
function videoSeek(node, url, t) {
    const st = (node._h3Seek ??= { url: null, t: -1 });
    if (st.url === url && Math.abs(st.t - t) < 1 / 24) return;
    st.url = url;
    st.t = t;
    const v = (node._h3Player ??= Object.assign(document.createElement("video"), {
        muted: true,
        playsInline: true,
        preload: "auto",
    }));
    v.onloadeddata = () => {
        try {
            v.currentTime = t;
        } catch (_) {}
    };
    v.onseeked = () => redrawNode(node);
    if (v.src !== url) v.src = url;
    else {
        try {
            v.currentTime = t;
        } catch (_) {}
    }
}

// per-clip <video> elements render each clip's own thumbnail frame
function thumbEl(node, c, m) {
    let t = node._h3Thumbs.get(c.id);
    if (!t) {
        t = {
            el: Object.assign(document.createElement("video"), {
                muted: true,
                playsInline: true,
                preload: "auto",
            }),
            url: null,
            t: -1,
        };
        node._h3Thumbs.set(c.id, t);
        t.el.onseeked = () => redrawNode(node);
        t.el.onloadeddata = () => {
            try {
                t.el.currentTime = t.t;
            } catch (_) {}
        };
    }
    return t;
}

function thumbSeek(node, c, m, target) {
    const t = thumbEl(node, c, m);
    if (t.url === m.url && Math.abs(t.t - target) < 1 / 24) return t;
    t.url = m.url;
    t.t = target;
    if (t.el.src !== m.url) t.el.src = m.url;
    else {
        try {
            t.el.currentTime = target;
        } catch (_) {}
    }
    return t;
}

function paintCover(ctx, el, r) {
    if (!el || (!el.videoWidth && !el.naturalWidth)) return;
    const w = el.videoWidth || el.naturalWidth;
    const h = el.videoHeight || el.naturalHeight;
    const s = Math.max(r.w / w, r.h / h);
    const dw = w * s;
    const dh = h * s;
    ctx.drawImage(el, r.x + (r.w - dw) / 2, r.y + (r.h - dh) / 2, dw, dh);
}

function paintWaveform(ctx, m, r, ghost) {
    if (!m?.peaks) return;
    const n = m.peaks.length / 2;
    if (n < 2) return;
    ctx.fillStyle = "rgba(0,0,0," + (ghost ? 0.25 : 0.35) + ")";
    ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.fillStyle = ghost ? "rgba(255,255,255,0.5)" : "#fff";
    const cw = (r.w - 4) / (n - 1);
    for (let i = 0; i < n; i++) {
        const amp = Math.max(0.02, m.peaks[i * 2 + 1]);
        const bh = Math.max(1.5, amp * (r.h - 6));
        const x = r.x + 2 + i * cw;
        ctx.fillRect(x, r.y + (r.h - bh) / 2, Math.max(1, cw * 0.7), bh);
    }
}

// --- timeline state --------------------------------------------------------

function clipInputs(c) {
    if (c.file) return [];
    if (c.kind === "video") {
        return [
            [`video_${c.id}`, "IMAGE"],
            [`video_audio_${c.id}`, "AUDIO"],
        ];
    }
    if (c.kind === "image") return [[`image_${c.id}`, "IMAGE"]];
    return [[`audio_${c.id}`, "AUDIO"]];
}

function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
}

function laneOf(kind) {
    return kind === "audio" ? 1 : 0;
}

function blockRect(c, s) {
    const len = c.kind === "image" ? 3 : Number(c.len) || 22;
    return {
        x: OFFSET_X + (Number(c.start) - 1) * s + 2,
        y: RULER_H + laneOf(c.kind) * LANE_H + 4,
        w: Math.max(2, len * s - 4),
        h: LANE_H - 8,
    };
}

function ghostRect(c, s) {
    const start = c.audio_link ? c.start : c.audio_start ?? c.start;
    const len = c.audio_link ? c.len : c.audio_len ?? c.len ?? 22;
    return {
        x: OFFSET_X + (Number(start) - 1) * s + 2,
        y: RULER_H + LANE_H + 4,
        w: Math.max(2, (Number(len) || 22) * s - 4),
        h: LANE_H - 8,
    };
}

// occupied [s, e) frame range of a clip inside a given lane
function laneRange(c, lane) {
    if (lane === 0) {
        const len = c.kind === "image" ? 3 : Number(c.len) || 22;
        return { s: Number(c.start), e: Number(c.start) + len };
    }
    const s = c.kind === "audio" ? Number(c.start) : Number(c.audio_start ?? c.start);
    const len = c.kind === "audio" ? Number(c.len) || 22 : Number(c.audio_len ?? c.len ?? 22);
    return { s, e: s + len };
}

// magnet-snap a value to the playhead boundary (or the span ends). Used by
// the trim handlers so that resizing a clip "kisses" the playhead without
// yanking it toward every other clip edge in the timeline — the collision
// guard already keeps clips from overlapping, so the only free magnet the
// user actually wants here is the playhead.
function probeSnap(node, value, scale) {
    if (node._h3TimelineWidget?._snapEnabled === false) return value;
    const span = node._h3Span ?? SPAN;
    let best = value;
    let dBest = Infinity;
    const probe = (v) => {
        const d = Math.abs(v - value);
        if (d < dBest) {
            dBest = d;
            best = v;
        }
    };
    probe(1);
    probe(span + 1);
    const pl = node._h3TimelineWidget?._play;
    if (pl != null) {
        probe(clamp(Math.round(pl) + 1, 1, span));
    }
    if (dBest <= Math.max(0.5, SNAP_PLAY_PX / scale)) return best;
    return value;
}

// magnet-snap to the playhead boundary (and span ends) within SNAP_PLAY_PX.
// Snapping to other clips is handled by the collision push above: the clip
// is shoved flush against whichever neighbour it would overlap, so there is
// no need to probe arbitrary clip edges here — that only made the magnet
// feel like it was fighting the user at every position.
function resolveMove(node, c, lane, s, len, grab, px) {
    const clips = node._h3Clips;
    const hi = (node._h3Span ?? SPAN) - len + 1;
    // collision: push the clip flush against the neighbour it would overlap.
    // this doubles as the clip-to-clip snap: once resolved the clip sits
    // edge-to-edge with the neighbour, regardless of how fast you dragged.
    for (let guard = 0; guard < clips.length; guard++) {
        let hit = null;
        for (const o of clips) {
            if (o === c || laneOf(o.kind) !== lane) continue;
            const r = laneRange(o, lane);
            if (s < r.e && s + len > r.s) {
                hit = r;
                break;
            }
        }
        if (!hit) break;
        s = grab < (hit.s + hit.e) / 2 ? hit.s - len : hit.e;
        if (s < 1) break;
    }
    s = clamp(s, 1, hi);

    if (node._h3TimelineWidget?._snapEnabled !== false) {
        let best = s;
        let dBest = Infinity;
        const probe = (v) => {
            if (v < 1 || v > hi) return;
            const d = Math.abs(v - s);
            if (d < dBest) {
                dBest = d;
                best = v;
            }
        };
        probe(1);
        probe(hi);
        const pl = node._h3TimelineWidget?._play;
        if (pl != null) {
            const f = clamp(Math.round(pl) + 1, 1, node._h3Span ?? SPAN);
            probe(f);
            probe(f - len);
        }
        if (dBest <= Math.max(0.5, SNAP_PLAY_PX / px)) s = best;
    }
    return clamp(s, 1, hi);
}

function inRect(p, r, pad = 0) {
    return (
        p[0] >= r.x - pad &&
        p[0] <= r.x + r.w + pad &&
        p[1] >= r.y - pad &&
        p[1] <= r.y + r.h + pad
    );
}

function edgeZone(p, r) {
    if (p[0] < r.x + 5) return "trimL";
    if (p[0] > r.x + r.w - 5) return "trimR";
    return null;
}

function btnZone(p) {
    if (p[1] > RULER_H || p[0] < TOOL_X) return null;
    const col = Math.floor((p[0] - TOOL_X) / 20);
    return ["split", "snap", "play", "unit", "in", "out"][col] || null;
}

function playHeadBoundary(node) {
    const w = node._h3TimelineWidget;
    const v = w?._play;
    if (v != null) {
        return clamp(Math.round(v) + 1, 1, node._h3Span ?? SPAN);
    }
    return clamp(w?._frame ?? 1, 1, node._h3Span ?? SPAN);
}

function splitAt(node) {
    const clips = node._h3Clips;
    const f = playHeadBoundary(node);
    const i = clips.findIndex(
        (c) =>
            c.file &&
            (c.kind === "video" || c.kind === "audio") &&
            f > Number(c.start) &&
            f < Number(c.start) + (Number(c.len) || 1),
    );
    if (i < 0) return;
    const c = clips[i];
    const cut = f - Number(c.start);
    const left = { ...c, len: cut };
    const right = {
        ...c,
        start: f,
        len: (Number(c.len) || 1) - cut,
        src_start: (Number(c.src_start) || 0) + cut,
        id: (clips.at(-1)?.id ?? 0) + 1,
    };
    clips.splice(i, 1, left, right);
    writeState(node);
    fixNodeSize(node);
}

function hitTest(node, p, s) {
    if (p[1] < RULER_H) {
        const b = btnZone(p);
        return b ? { zone: b } : { zone: "ruler" };
    }
    for (let i = node._h3Clips.length - 1; i >= 0; i--) {
        const c = node._h3Clips[i];
        if (c.kind === "video") {
            const g = ghostRect(c, s);
            if (inRect(p, g, 4)) {
                const bx = g.x + 8;
                const by = g.y + g.h / 2;
                if (Math.hypot(p[0] - bx, p[1] - by) < 9) {
                    return { i, c, zone: "link" };
                }
                if (!c.audio_link) {
                    return { i, c, zone: edgeZone(p, g) || "audio" };
                }
                return null; // linked: the sound moves with the video
            }
        }
        const r = blockRect(c, s);
        if (inRect(p, r, 2)) {
            if (p[0] < r.x + 16 && p[1] < r.y + 14) {
                return { i, c, zone: "media" };
            }
            if (p[0] > r.x + r.w - 14 && p[1] < r.y + 14) {
                return { i, c, zone: "remove" };
            }
            if (c.kind === "image") return { i, c, zone: "move" };
            return { i, c, zone: edgeZone(p, r) || "move" };
        }
    }
    return null;
}

function stateWidget(node) {
    return node.widgets?.find((w) => w.name === STATE_NAME);
}

function readState(node) {
    const raw = stateWidget(node);
    let clips = [];
    try {
        const parsed = JSON.parse(raw?.value || "{}");
        if (Array.isArray(parsed.clips)) clips = parsed.clips;
    } catch (_) {}
    return clips.filter(
        (c) => c && c.kind && Number.isFinite(Number(c.start)),
    );
}

function writeState(node) {
    const raw = stateWidget(node);
    if (raw) {
        raw.value = JSON.stringify({
            clips: node._h3Clips,
            unit: node._h3TimelineWidget?._unit ?? "f",
        });
    }
    redrawNode(node);
    app.canvas?.setDirtyCanvas?.(true, true);
}

function hideStateWidget(node) {
    const widget = stateWidget(node);
    if (!widget || widget._h3Hidden) return;
    widget._h3Hidden = true;
    widget.hidden = true;
    widget.options ??= {};
    widget.options.hidden = true;
    widget.computeSize = () => [0, -4];
}

function widgetYOffset(node, fallback) {
    if (typeof node.getInputPos !== "function") return fallback;
    const n = node.inputs?.length || 0;
    if (!n) return fallback;
    const p = node.getInputPos(n - 1);
    if (!Array.isArray(p) || !Number.isFinite(p[1])) return fallback;
    const top = Number.isFinite(node.pos?.[1]) ? node.pos[1] : 0;
    return p[1] - top + 16;
}

function fixNodeSize(node) {
    // new_litegraph sizes the node from slots only and ignores widget
    // widths, so the timeline widget would be drawn at ~200px and clipped.
    // Force the node wide enough for the full ruler and lock the size: the
    // timeline has a fixed 840px canvas, a resized node would leave the
    // zoom buttons and ruler misaligned with a dead strip on the right.
    node.resizable = false;
    let h = HEIGHT + 40;
    try {
        if (typeof node.computeSize === "function") h = node.computeSize()[1] + 12;
    } catch (_) {}
    if (typeof node.resize === "function") node.resize(WIDTH, Math.max(h, HEIGHT + 24));
    else node.setSize?.([WIDTH, Math.max(h, HEIGHT + 24)]);
    const w = node._h3TimelineWidget;
    if (w) {
        w.y = widgetYOffset(node, w.y);
        w._rowOf = widgetYOffset(node, w._rowOf);
    }
}

function ensureInputs(node) {
    const keep = new Set();
    for (const c of node._h3Clips) {
        for (const [name, type] of clipInputs(c)) {
            keep.add(name);
            if (!node.inputs?.some((inp) => inp.name === name)) {
                node.addInput(name, type, { label: name });
            }
        }
    }
    for (let i = (node.inputs?.length || 0) - 1; i >= 0; i--) {
        const inp = node.inputs[i];
        if (/^(video_|image_|audio_)/.test(inp.name) && !keep.has(inp.name)) {
            if (inp.link != null) node.disconnectInput(i);
            node.removeInput(i);
        }
    }
}

function removeClip(node, i) {
    const [clip] = node._h3Clips.splice(i, 1);
    node._h3Thumbs?.delete(clip.id);
    for (const [name] of clipInputs(clip)) {
        const slot = node.inputs?.findIndex((inp) => inp.name === name);
        if (slot >= 0) {
            if (node.inputs[slot].link != null) node.disconnectInput(slot);
            node.removeInput(slot);
        }
    }
    ensureInputs(node);
    writeState(node);
    fixNodeSize(node);
}

async function addClipWithMedia(node, kind) {
    if (!node._h3Clips || node._h3Clips.length >= MAX_CLIPS) return;
    const file = await pickFile();
    if (!file) return;
    let info = null;
    try {
        info = await uploadMedia(file);
    } catch (err) {
        console.warn("h3 timeline: upload failed", err);
        return;
    }
    const detected = kindOfFile(info);
    addClip(node, detected || kind, info);
}

function placeAndPushClip(node, newClip) {
    const lane = laneOf(newClip.kind);
    const start = playHeadBoundary(node);
    newClip.start = start;
    const newLen = Number(newClip.len) || (newClip.kind === "image" ? 3 : 22);

    const sameLane = node._h3Clips.filter((c) => c !== newClip && laneOf(c.kind) === lane);
    sameLane.sort((a, b) => Number(a.start) - Number(b.start));

    let pushCursor = start + newLen;
    for (const c of sameLane) {
        const cStart = Number(c.start);
        const cLen = Number(c.len) || 1;
        if (cStart >= start && cStart < pushCursor) {
            c.start = pushCursor;
            pushCursor = c.start + cLen;
        } else if (cStart < start && cStart + cLen > start) {
            c.start = pushCursor;
            pushCursor = c.start + cLen;
        }
    }
}

function addClip(node, kind, info) {
    if (!node._h3Clips || node._h3Clips.length >= MAX_CLIPS) return;
    const c = defaults[kind]();
    c.id = (node._h3Clips.at(-1)?.id ?? 0) + 1;
    if (info) {
        c.file = info;
        c.src_start = 0;
    }
    node._h3Clips.push(c);
    placeAndPushClip(node, c);
    ensureInputs(node);
    writeState(node);
    fixNodeSize(node);
}

function roundRect(ctx, x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.arcTo(x + w, y, x + w, y + h, r);
    ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r);
    ctx.arcTo(x, y, x + w, y, r);
    ctx.closePath();
}

function drawBlock(ctx, color, label, r, ghost, media, node, clip) {
    ctx.globalAlpha = ghost ? 0.35 : 0.55;
    ctx.fillStyle = color;
    roundRect(ctx, r.x, r.y, r.w, r.h, 3);
    ctx.fill();
    ctx.globalAlpha = 1;
    if (media?.kind === "audio") {
        paintWaveform(ctx, media, r, ghost);
    } else if (!ghost && media?.kind === "image" && media.img) {
        ctx.save();
        roundRect(ctx, r.x, r.y, r.w, r.h, 3);
        ctx.clip();
        paintCover(ctx, media.img, r);
        ctx.restore();
    } else if (!ghost && media?.kind === "video" && node?._h3Thumbs?.get(clip?.id)?.el) {
        ctx.save();
        roundRect(ctx, r.x, r.y, r.w, r.h, 3);
        ctx.clip();
        paintCover(ctx, node._h3Thumbs.get(clip.id).el, r);
        ctx.restore();
    }
    ctx.lineWidth = 1;
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = "9px sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(label, r.x + (ghost ? 22 : 4), r.y + r.h / 2);
}

function drawGhost(ctx, c, s, node) {
    const g = ghostRect(c, s);
    drawBlock(ctx, COLORS.audio, `♪ ${c.id}`, g, true, null, node);
    const bx = g.x + 8;
    const by = g.y + g.h / 2;
    ctx.fillStyle = "#222";
    ctx.beginPath();
    ctx.arc(bx, by, 8, 0, Math.PI * 2);
    ctx.fill();
    ctx.lineWidth = 1;
    ctx.strokeStyle = COLORS.audio;
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = "9px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(c.audio_link ? "🔗" : "⛓", bx, by + 0.5);
}

// --- playhead --------------------------------------------------------------

function splitSnap(node, p, s) {
    if (node._h3TimelineWidget?._snapEnabled === false) return p;
    // magnet-snap a playhead boundary to clip edges like resolveMove does;
    // p is 0-based play position, clip boundaries are 1-based frame edges
    const span = node._h3Span ?? SPAN;
    let best = p;
    let dBest = Infinity;
    const probe = (v) => {
        const d = Math.abs(v - p);
        if (d < dBest) {
            dBest = d;
            best = v;
        }
    };
    for (const o of node._h3Clips) {
        const r = laneRange(o, laneOf(o.kind));
        probe(r.s - 1);
        probe(r.e - 1);
        if (o.kind === "video") {
            const v = laneRange(o, 1);
            probe(v.s - 1);
            probe(v.e - 1);
        }
    }
    if (dBest <= Math.max(0.5, SNAP_PX / s)) return best;
    return p;
}

function togglePlay(node) {
    const w = node._h3TimelineWidget;
    if (!w) return;
    w._playing = !w._playing;
    if (w._playing) {
        if (!node._h3AudioCtx) {
            node._h3AudioCtx =
                new (window.AudioContext || window.webkitAudioContext)();
        }
        node._h3AudioCtx.resume?.();
        w._lastTick = null;
        const tick = (now) => {
            if (!w._playing) return;
            const dt = w._lastTick == null ? 0 : (now - w._lastTick) / 1000;
            w._lastTick = now;
            const pos = (w._play ?? 0) + dt * (w._fps || 24);
            w._play = Math.max(0, pos);
            syncPreview(node);
            if (pos > (node._h3Span ?? SPAN)) {
                w._playing = false;
                stopSound(node);
            }
            redrawNode(node);
            requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    } else {
        stopSound(node);
    }
    redrawNode(node);
}

function stopSound(node) {
    const src = node._h3Sound;
    if (src) {
        try {
            src.stop();
        } catch (_) {}
        src.disconnect();
        node._h3Sound = null;
    }
}

function previewClip(node) {
    const w = node._h3TimelineWidget;
    const f = w?._play ?? 0;
    if (w?._play == null) return null;
    return (
        node._h3Clips?.find((c) => {
            const s = Number(c.start) - 1;
            const len = c.kind === "image" ? 3 : Number(c.len) || 1;
            return f >= s && f < s + len;
        }) || null
    );
}

function syncPreview(node) {
    const w = node._h3TimelineWidget;
    const c = previewClip(node);
    const play = w?._play ?? 0;
    if (c?.file) {
        const m = ensureMedia(node, c);
        if (m?.kind === "video" && c.kind === "video") {
            const fps = w?._fps || 24;
            const t = (play - (Number(c.start) - 1)) / fps +
                (Number(c.src_start) || 0) / fps;
            videoSeek(node, m.url, t);
        }
    }
    if (c?.kind === "audio" && w?._playing) startSound(node, c);
    else stopSound(node);
}

function startSound(node, c) {
    const key = mediaKey(c.file);
    if (node._h3Sound && node._h3SoundClip !== key) stopSound(node);
    if (node._h3Sound) return; // already playing this track
    const m = ensureMedia(node, c);
    if (!m?.buffer || !node._h3AudioCtx) return;
    const play = node._h3TimelineWidget?._play ?? 0;
    const off = (play - (Number(c.start) - 1)) / (node._h3TimelineWidget?._fps || 24) +
        (Number(c.src_start) || 0) / (node._h3TimelineWidget?._fps || 24);
    try {
        const src = node._h3AudioCtx.createBufferSource();
        src.buffer = m.buffer;
        src.connect(node._h3AudioCtx.destination);
        src.start(0, Math.max(0, off));
        node._h3Sound = src;
        node._h3SoundClip = key;
    } catch (_) {}
}

function setup(node) {
    hideStateWidget(node);
    node._h3Media ??= new Map();
    node._h3Thumbs ??= new Map();
    node._h3Clips = readState(node);
    if (!node._h3Clips.length) node._h3Clips = [defaults.image()];
    ensureInputs(node);

    if (!node._h3TimelineWidget) {
        const widget = {
            name: "timeline",
            type: "h3_timeline",
            width: WIDTH,
            computedHeight: HEIGHT,
            y: 0,
            _yOff: 1,
            _rowOf: 1,
            _scale: PX,
            _frame: null,
            _unit: "f",
            _fps: 24,
            _drag: null,
            _hover: null,
            _dragPlay: false,
            _play: 0,
            _playing: false,
            _ctxs: new Set(),
            computeSize: () => [WIDTH, HEIGHT],
            draw(ctx, nd, width, y, H) {
                // y is the widget row offset inside the node. Canvas
                // renderer (1.0): widget.y (>= 4). Vue WidgetLegacy (2.0):
                // always 1, canvas is widget-local. NaN/0 -> keep last.
                if (ctx?.canvas?.isConnected) this._ctxs.add(ctx);
                if (Number.isFinite(y)) {
                    if (y >= 4) this._yOff = y;
                    else if (y === 1) this._yOff = 1;
                }
                this._clear(ctx);
                ctx.save();
                ctx.translate(0, this._yOff ?? 1);
                this.paint(ctx, nd, width, H);
                ctx.restore();
            },
            _clear(ctx) {
                // WidgetLegacy canvases are owned entirely by the widget, so
                // clear them or translucent redraws smear. The litegraph
                // graph canvas holds the whole graph and redraws itself each
                // frame, so never touch it.
                if (!ctx.canvas?.className?.includes("cursor-crosshair")) return;
                const t = ctx.getTransform();
                ctx.clearRect(0, 0, ctx.canvas.width / t.a, ctx.canvas.height / t.d);
            },
            redraw(nd) {
                // WidgetLegacy instances (graph body + properties panel)
                // share this widget and fight over triggerDraw, leaving the
                // graph-body canvas stale. Repaint directly on every live
                // canvas context we have ever been asked to draw on.
                for (const ctx of this._ctxs) {
                    if (!ctx.canvas || !ctx.canvas.isConnected) {
                        this._ctxs.delete(ctx);
                        continue;
                    }
                    this._clear(ctx);
                    ctx.save();
                    ctx.translate(0, this._yOff ?? 1);
                    this.paint(ctx, nd, WIDTH, HEIGHT);
                    ctx.restore();
                }
            },
            paint(ctx, nd, width, H) {
                const clips = nd._h3Clips || [];
                const s = this._scale;

                ctx.fillStyle = "rgba(0,0,0,0.25)";
                ctx.fillRect(0, 0, width, H);
                ctx.strokeStyle = "rgba(255,255,255,0.15)";
                ctx.beginPath();
                ctx.moveTo(0, RULER_H + LANE_H + 0.5);
                ctx.lineTo(width, RULER_H + LANE_H + 0.5);
                ctx.stroke();

                const span = nd._h3Span ?? SPAN;
                let tick = 1;
                while (tick * 2 * s < 8) tick *= 2;
                // one label per frame at max zoom-in, one every 24 frames at
                // max zoom-out; further out the spacing target wins so labels
                // never overlap.
                let label = s >= 18 ? 1 : clamp(Math.round(48 / s), 1, 24);
                if (s < 1.5) label = Math.max(label, Math.ceil(40 / s / 10) * 10);
                label = Math.min(Math.ceil(label / tick) * tick, span);
                const sec = this._unit === "s";
                const fps = this._fps;
                ctx.fillStyle = "#888";
                ctx.font = "10px sans-serif";
                ctx.textAlign = "left";
                ctx.textBaseline = "top";
                // ticks and labels in separate loops: a grid step of 2 skips
                // the parity of label multiples, so the old combined loop
                // could draw no numbers at all at max zoom-out
                for (let f = 1; f <= span; f += tick) {
                    const x = OFFSET_X + (f - 1) * s;
                    ctx.fillRect(x, RULER_H - 4, 1, 4);
                }
                for (let f = label; f <= span; f += label) {
                    const x = OFFSET_X + (f - 1) * s;
                    ctx.fillRect(x, RULER_H - 9, 1, 9);
                    let txt = String(f);
                    if (sec) {
                        const v = (f - 1) / fps;
                        txt = Number.isInteger(v) ? String(v) : v.toFixed(1);
                    }
                    ctx.fillText(txt, x + 2, 2);
                }

                ctx.strokeStyle = "#888";
                const snapOn = this._snapEnabled ?? true;
                const btnChars = ["✂", "🧲", this._playing ? "⏹" : "▶", "F", "−", "+"];
                for (const [i, ch] of btnChars.entries()) {
                    const x = TOOL_X + i * 20;
                    roundRect(ctx, x + 0.5, 3, BTN_W, BTN_H, 3);
                    if (i === 1 && snapOn) {
                        ctx.fillStyle = "#3a5a80";
                    } else {
                        ctx.fillStyle = "#333";
                    }
                    ctx.fill();
                    ctx.stroke();
                }
                ctx.fillStyle = "#ddd";
                ctx.font = "11px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                for (const [i, ch] of btnChars.entries()) {
                    ctx.fillText(ch, TOOL_X + i * 20 + BTN_W / 2, 3 + BTN_H / 2 + 0.5);
                }

                ctx.fillText("video", 2, RULER_H + 14);
                ctx.fillText("audio", 2, RULER_H + LANE_H + 14);

                for (const c of clips) {
                    const media = c.file ? ensureMedia(nd, c) : null;
                    if (c.kind === "video") drawGhost(ctx, c, s, nd);
                    if (c.kind === "image") {
                        drawBlock(ctx, COLORS.image, `img ${c.id}`, blockRect(c, s), false, media, nd, c);
                    } else if (c.kind === "video") {
                        drawBlock(ctx, COLORS.video, `video ${c.id}`, blockRect(c, s), false, media, nd, c);
                    } else {
                        drawBlock(ctx, COLORS.audio, `audio ${c.id}`, blockRect(c, s), false, media, nd, c);
                    }
                }

                // video thumbnails: seek each clip's own element to its
                // source start, redraw on loadeddata/seeked
                for (const c of clips) {
                    if (c.kind !== "video" || !c.file) continue;
                    const key = mediaKey(c.file);
                    const m = nd._h3Media?.get(key);
                    if (m?.kind === "video") {
                        thumbSeek(
                            nd,
                            c,
                            m,
                            (Number(c.start) - 1 + (Number(c.src_start) || 0)) / fps,
                        );
                    }
                }

                const hov = this._hover;
                if (hov && clips[hov.i]) {
                    const r = blockRect(hov.c, s);
                    ctx.fillStyle = "#e05252";
                    ctx.font = "10px sans-serif";
                    ctx.textAlign = "right";
                    ctx.textBaseline = "top";
                    ctx.fillText("✕", r.x + r.w - 2, r.y + 2);
                    ctx.textAlign = "left";
                    ctx.fillText("📁", r.x + 1, r.y + 1);
                }

                // playhead
                if (this._play != null) {
                    const x = OFFSET_X + Math.max(0, this._play) * s;
                    ctx.fillStyle = PLAY_COLOR;
                    ctx.beginPath();
                    ctx.moveTo(x - 4, RULER_H - 7);
                    ctx.lineTo(x + 4, RULER_H - 7);
                    ctx.lineTo(x, RULER_H - 1);
                    ctx.closePath();
                    ctx.fill();
                    ctx.fillRect(x - 0.5, RULER_H - 1, 1, H - RULER_H + 1);
                    const pc = previewClip(nd);
                    if (pc) {
                        ctx.globalAlpha = 0.16;
                        const pr = blockRect(pc, s);
                        ctx.fillStyle = PLAY_COLOR;
                        roundRect(ctx, pr.x - 2, pr.y - 2, pr.w + 4, pr.h + 4, 3);
                        ctx.fill();
                        ctx.globalAlpha = 1;
                    }
                }

                const fr = this._play != null ? this._play : this._frame;
                if (fr != null) {
                    ctx.fillStyle = "#ccc";
                    ctx.font = "10px sans-serif";
                    ctx.textAlign = "right";
                    ctx.textBaseline = "bottom";
                    const txt =
                        this._unit === "s"
                            ? `${((this._play != null ? fr : fr - 1) / this._fps).toFixed(2)}s`
                            : `frame ${Math.round(fr) + (this._play != null ? 1 : 0)}`;
                    ctx.fillText(txt, WIDTH - 4, H - 6);
                }
            },
            mouse(e, pos, nd) {
                // pos space varies: canvas renderer passes node-relative
                // coords (y = row offset), Vue WidgetLegacy passes
                // widget-local coords except for the pointerdown which is
                // node-relative via processWidgetClick. _yOff is >= 4 in
                // canvas mode, exactly 1 in Vue mode (WidgetLegacy always
                // calls draw with y=1).
                let y = pos[1] - this._yOff;
                if (this._yOff <= 4 && y > HEIGHT + 4) y = pos[1] - this._rowOf;
                const p = [pos[0], y];
                const type = e.type || "";
                if (type.endsWith("down") && e.button === 0) {
                    const hit = hitTest(nd, p, this._scale);
                    if (!hit) return false;
                    e.preventDefault();
                    if (hit.zone === "in" || hit.zone === "out") {
                        const f = hit.zone === "in" ? ZOOM_STEP : 1 / ZOOM_STEP;
                        const minS = Math.max(
                            0.5,
                            Math.min(ZOOM_MIN, WIDTH / (nd._h3Span ?? SPAN)),
                        );
                        this._scale = clamp(this._scale * f, minS, ZOOM_MAX);
                        this.redraw(nd);
                        return true;
                    }
                    if (hit.zone === "unit") {
                        this._unit = this._unit === "s" ? "f" : "s";
                        writeState(nd);
                        this.redraw(nd);
                        return true;
                    }
                    if (hit.zone === "snap") {
                        this._snapEnabled = !(this._snapEnabled ?? true);
                        this.redraw(nd);
                        return true;
                    }
                    if (hit.zone === "play") {
                        togglePlay(nd);
                        return true;
                    }
                    if (hit.zone === "split") {
                        splitAt(nd);
                        this.redraw(nd);
                        return true;
                    }
                    if (hit.zone === "ruler") {
                        this._dragPlay = true;
                        const s = this._scale;
                        const v = clamp(Math.round((p[0] - OFFSET_X) / s), 0, (nd._h3Span ?? SPAN) - 1);
                        this._play = splitSnap(nd, v, s);
                        this._frame = null;
                        if (!this._playing) syncPreview(nd);
                        this.redraw(nd);
                        return true;
                    }
                    if (hit.zone === "link") {
                        hit.c.audio_link = !hit.c.audio_link;
                        writeState(nd);
                    } else if (hit.zone === "remove") {
                        this._drag = null;
                        this._hover = null;
                        removeClip(nd, hit.i);
                    } else if (hit.zone === "media") {
                        replaceClipMedia(nd, hit.c);
                    } else if (hit.c) {
                        this._drag = {
                            ...hit,
                            grab: p[0],
                            startAt: Number(
                                hit.c.kind === "video" && !hit.c.audio_link
                                    ? hit.c.audio_start ?? hit.c.start
                                    : hit.c.start,
                            ),
                            lenAt: Number(
                                hit.c.kind === "video" && !hit.c.audio_link
                                    ? hit.c.audio_len ?? hit.c.len ?? 22
                                    : hit.c.len ?? 22,
                            ),
                        };
                    } else {
                        return true;
                    }
                    return true;
                }
                if (type.includes("move")) {
                    nd._h3Hovered = true;
                    this._hover = hitTest(nd, p, this._scale);
                    const span = nd._h3Span ?? SPAN;
                    const d = this._drag;
                    if (this._dragPlay) {
                        const s = this._scale;
                        const v = clamp(Math.round((p[0] - OFFSET_X) / s), 0, span - 1);
                        this._play = splitSnap(nd, v, s);
                        if (!this._playing) syncPreview(nd);
                        this.redraw(nd);
                        return true;
                    }
                    if (d) {
                        const s = this._scale;
                        const step = Math.round((p[0] - d.grab) / s);
                        const unb = d.c.kind === "video" && !d.c.audio_link;
                        const lane = d.zone === "audio" ? 1 : laneOf(d.c.kind);
                        if (d.zone === "move" || d.zone === "audio") {
                            const img = d.c.kind === "image";
                            const len = img
                                ? 3
                                : d.zone === "audio"
                                  ? (d.c.audio_len ?? d.c.len ?? 22)
                                  : (d.c.len ?? 22);
                            const s2 = resolveMove(
                                nd,
                                d.c,
                                lane,
                                clamp(d.startAt + step, 1, span - len + 1),
                                len,
                                d.grab / s + 1,
                                s,
                            );
                            if (d.zone === "audio") d.c.audio_start = s2;
                            else d.c.start = s2;
                            this._frame = s2;
                        } else if (d.zone === "trimR") {
                            let len = clamp(d.lenAt + step, 1, span - d.startAt + 1);
                            if (nd._h3TimelineWidget?._snapEnabled !== false) {
                                const end = probeSnap(nd, d.startAt + len, s);
                                len = clamp(end - d.startAt, 1, span - d.startAt + 1);
                            }
                            for (const o of nd._h3Clips) {
                                if (o === d.c || laneOf(o.kind) !== lane) continue;
                                const r = laneRange(o, lane);
                                if (r.s >= d.startAt) len = Math.min(len, r.s - d.startAt);
                            }
                            len = Math.max(1, len);
                            if (unb) d.c.audio_len = len;
                            else d.c.len = len;
                            this._frame = d.startAt + len - 1;
                        } else if (d.zone === "trimL") {
                            let s2 = clamp(d.startAt + step, 1, d.startAt + d.lenAt - 1);
                            if (nd._h3TimelineWidget?._snapEnabled !== false) {
                                s2 = probeSnap(nd, s2, s);
                            }
                            let right = d.startAt + d.lenAt;
                            for (let guard = 0; guard < nd._h3Clips.length; guard++) {
                                if (s2 >= right) break;
                                let changed = false;
                                for (const o of nd._h3Clips) {
                                    if (o === d.c || laneOf(o.kind) !== lane) continue;
                                    const r = laneRange(o, lane);
                                    if (r.s <= s2 && r.e > s2) {
                                        s2 = r.e;
                                        changed = true;
                                    } else if (r.s > s2 && r.s < right) {
                                        right = r.s;
                                        changed = true;
                                    }
                                }
                                if (!changed) break;
                            }
                            s2 = clamp(s2, 1, d.startAt + d.lenAt - 1);
                            if (s2 >= right) right = s2 + 1;
                            const len = Math.max(1, right - s2);
                            if (unb) {
                                d.c.audio_start = s2;
                                d.c.audio_len = len;
                            } else {
                                d.c.start = s2;
                                d.c.len = len;
                            }
                            this._frame = s2;
                        }
                        writeState(nd);
                        this.redraw(nd);
                    } else {
                        this._frame = clamp(Math.round((p[0] - OFFSET_X) / this._scale + 1), 1, span);
                        this.redraw(nd);
                    }
                    return true;
                }
                if (type.includes("up")) {
                    this._drag = null;
                    this._hover = null;
                    this._dragPlay = false;
                    this._frame = null;
                    this.redraw(nd);
                    return true;
                }
                return false;
            },
        };
        widget.options = { serialize: false };
        widget.serialize = false;
        node.addCustomWidget(widget);
        node._h3TimelineWidget = widget;
    }

    if (!node._h3AddButtons) {
        node._h3AddButtons = true;
        for (const [label, kind] of [
            ["+ image", "image"],
            ["+ video", "video"],
            ["+ audio", "audio"],
        ]) {
            const b = node.addWidget("button", label, null, () => addClipWithMedia(node, kind));
            b.serialize = false;
            b.options ??= {};
            b.options.serialize = false;
        }
    }

    if (!node._h3Keys) {
        node._h3Keys = true;
        const onKey = (e) => {
            if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
            if (!node._h3TimelineWidget || !node._h3Clips || !node._h3Hovered) return;
            const k = e.key?.toLowerCase();
            if (k === "s") {
                e.preventDefault();
                splitAt(node);
            } else if (k === " ") {
                if (!e.repeat) togglePlay(node);
                e.preventDefault();
            }
        };
        document.addEventListener("keydown", onKey);
        node._h3KeyHandler = onKey;
    }

    if (!node._h3FpsWidget) {
        const w = node.addWidget(
            "number",
            "fps",
            24,
            (v) => {
                const tw = node._h3TimelineWidget;
                if (tw) {
                    tw._fps = Math.max(1, Math.round(Number(v) || 24));
                    tw.redraw?.(node);
                }
            },
            { min: 1, max: 240, step: 1 },
        );
        node._h3FpsWidget = w;
    }
    node._h3FpsWidget.value = Math.max(1, Math.round(Number(node._h3FpsWidget.value) || 24));

    if (!node._h3SpanWidget) {
        const w = node.addWidget(
            "number",
            "frames",
            SPAN,
            (v) => {
                node._h3Span = Math.max(1, Math.round(Number(v) || SPAN));
                node._h3TimelineWidget?.redraw?.(node);
            },
            { min: 1, max: 100000, step: 10 },
        );
        node._h3SpanWidget = w;
    }
    node._h3Span = Math.max(1, Math.round(Number(node._h3SpanWidget.value) || SPAN));

    const tw = node._h3TimelineWidget;
    if (tw) {
        tw._fps = node._h3FpsWidget.value;
        let unit = "f";
        try {
            const p = JSON.parse(stateWidget(node)?.value || "{}");
            if (p.unit === "s") unit = "s";
        } catch (_) {}
        tw._unit = unit;
    }

    writeState(node);
    fixNodeSize(node);
}

app.registerExtension({
    name: "seitanism.H3Timeline",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result = originalCreated?.apply(this, arguments);
            setTimeout(() => setup(this), 0);
            return result;
        };

        const originalConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result = originalConfigure?.apply(this, arguments);
            setTimeout(() => setup(this), 0);
            return result;
        };
    },
});