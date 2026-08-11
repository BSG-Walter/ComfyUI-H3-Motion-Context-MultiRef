import { app } from "../../scripts/app.js";

const NODE_NAME = "MiniMaxH3Timeline";
const STATE_NAME = "timeline_state";
const MAX_CLIPS = 32;

const SPAN = 240; // ruler length in frames
const PX = 3.5; // pixels per frame (base zoom)
const ZOOM_MIN = 2; // px/frame limits for zoom buttons
const ZOOM_MAX = 24;
const ZOOM_STEP = 1.25;
const SNAP_PX = 5; // magnet radius for clip snapping
const BTN_W = 16;
const BTN_H = 12;
const RULER_H = 18;
const LANE_H = 34;
const PAD = 4;
const WIDTH = 840;
const HEIGHT = RULER_H + 2 * LANE_H + PAD;

const COLORS = {
    image: "#7aa2f7",
    video: "#9ece6a",
    audio: "#f7768e",
};

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

function clipInputs(c) {
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
        x: (Number(c.start) - 1) * s + 2,
        y: RULER_H + laneOf(c.kind) * LANE_H + 4,
        w: Math.max(2, len * s - 4),
        h: LANE_H - 8,
    };
}

function ghostRect(c, s) {
    const start = c.audio_link ? c.start : c.audio_start ?? c.start;
    const len = c.audio_link ? c.len : c.audio_len ?? c.len ?? 22;
    return {
        x: (Number(start) - 1) * s + 2,
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

// push s so [s, s+len) never overlaps another clip in the same lane, then
// magnet-snap to the closest lane edge / ruler boundary within SNAP_PX.
function resolveMove(node, c, lane, s, len, grab, px) {
    const clips = node._h3Clips;
    const hi = (node._h3Span ?? SPAN) - len + 1;
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
    // magnet snaps to the edges of every clip in every lane
    const probeClip = (o) => {
        const r = laneRange(o, laneOf(o.kind));
        probe(r.s - len);
        probe(r.e);
        if (o.kind === "video") {
            const v = laneRange(o, 1);
            probe(v.s - len);
            probe(v.e);
        }
    };
    for (const o of clips) {
        if (o !== c) probeClip(o);
    }
    probe(1);
    probe(hi);
    if (dBest <= Math.max(0.5, SNAP_PX / px)) s = best;
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
    if (p[1] > RULER_H || p[0] < WIDTH - 64) return null;
    if (p[0] > WIDTH - 24) return { zone: "in" };
    if (p[0] > WIDTH - 44) return { zone: "out" };
    return { zone: "unit" };
}

function hitTest(node, p, s) {
    if (p[1] < RULER_H) return btnZone(p);
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
    const w = node._h3TimelineWidget;
    if (w) {
        w.triggerDraw?.();
        w.redraw?.(node);
    }
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

function addClip(node, kind) {
    if (!node._h3Clips || node._h3Clips.length >= MAX_CLIPS) return;
    const c = defaults[kind]();
    c.id = (node._h3Clips.at(-1)?.id ?? 0) + 1;
    const lastEnd = node._h3Clips.length
        ? Number(node._h3Clips.at(-1).start) + (Number(node._h3Clips.at(-1).len) || 1) - 1
        : 0;
    c.start = Math.max(1, lastEnd + 1);
    node._h3Clips.push(c);
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

function drawBlock(ctx, color, label, r, ghost) {
    ctx.globalAlpha = ghost ? 0.35 : 0.55;
    ctx.fillStyle = color;
    roundRect(ctx, r.x, r.y, r.w, r.h, 3);
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 1;
    ctx.strokeStyle = color;
    ctx.stroke();
    ctx.fillStyle = "#fff";
    ctx.font = "9px sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";
    ctx.fillText(label, r.x + (ghost ? 22 : 4), r.y + r.h / 2);
}

function drawGhost(ctx, c, s) {
    const g = ghostRect(c, s);
    drawBlock(ctx, COLORS.audio, `♪ ${c.id}`, g, true);
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

function setup(node) {
    hideStateWidget(node);
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
                    const x = (f - 1) * s;
                    ctx.fillRect(x, RULER_H - 4, 1, 4);
                }
                for (let f = label; f <= span; f += label) {
                    const x = (f - 1) * s;
                    ctx.fillRect(x, RULER_H - 9, 1, 9);
                    let txt = String(f);
                    if (sec) {
                        const v = (f - 1) / fps;
                        txt = Number.isInteger(v) ? String(v) : v.toFixed(1);
                    }
                    ctx.fillText(txt, x + 2, 2);
                }

                ctx.strokeStyle = "#888";
                ctx.fillStyle = "#333";
                const btnChars = [this._unit === "s" ? "S" : "F", "−", "+"];
                for (const [i, ch] of btnChars.entries()) {
                    const x = WIDTH - 60 + i * 20;
                    roundRect(ctx, x + 0.5, 3, BTN_W, BTN_H, 3);
                    ctx.fill();
                    ctx.stroke();
                }
                ctx.fillStyle = "#ddd";
                ctx.font = "11px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                ctx.fillText("F", WIDTH - 60 + BTN_W / 2, 3 + BTN_H / 2 + 0.5);
                ctx.fillText("−", WIDTH - 40 + BTN_W / 2, 3 + BTN_H / 2 + 0.5);
                ctx.fillText("+", WIDTH - 20 + BTN_W / 2, 3 + BTN_H / 2 + 0.5);

                ctx.fillText("video", 4, RULER_H + 14);
                ctx.fillText("audio", 4, RULER_H + LANE_H + 14);

                for (const c of clips) {
                    if (c.kind === "video") drawGhost(ctx, c, s);
                    if (c.kind === "image") {
                        drawBlock(ctx, COLORS.image, `img ${c.id}`, blockRect(c, s), false);
                    } else if (c.kind === "video") {
                        drawBlock(ctx, COLORS.video, `video ${c.id}`, blockRect(c, s), false);
                    } else {
                        drawBlock(ctx, COLORS.audio, `audio ${c.id}`, blockRect(c, s), false);
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
                }

                const fr = this._frame;
                if (fr != null) {
                    ctx.fillStyle = "#ccc";
                    ctx.font = "10px sans-serif";
                    ctx.textAlign = "right";
                    ctx.textBaseline = "bottom";
                    const txt =
                        this._unit === "s"
                            ? `${((fr - 1) / this._fps).toFixed(2)}s`
                            : `frame ${fr}`;
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
                        return true;
                    }
                    if (hit.zone === "link") {
                        hit.c.audio_link = !hit.c.audio_link;
                        writeState(nd);
                    } else if (hit.zone === "remove") {
                        this._drag = null;
                        this._hover = null;
                        removeClip(nd, hit.i);
                    } else {
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
                    }
                    return true;
                }
                if (type.includes("move")) {
                    this._hover = hitTest(nd, p, this._scale);
                    const span = nd._h3Span ?? SPAN;
                    const d = this._drag;
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
                            // the right edge stays at startAt+lenAt unless a
                            // next clip swallows it; the left edge is pushed
                            // past any clip it lands inside of.
                            let s2 = clamp(d.startAt + step, 1, d.startAt + d.lenAt - 1);
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
                    } else {
                        this._frame = clamp(Math.round(p[0] / this._scale + 1), 1, span);
                    }
                    return true;
                }
                if (type.includes("up")) {
                    this._drag = null;
                    this._hover = null;
                    this._frame = null;
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
            const b = node.addWidget("button", label, null, () => addClip(node, kind));
            b.serialize = false;
            b.options ??= {};
            b.options.serialize = false;
        }
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