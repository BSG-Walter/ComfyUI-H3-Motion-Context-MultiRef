// H3 timeline widget node: custom canvas widget + node wiring.
// Helpers live in the timeline_* modules.

import { app } from "../../scripts/app.js";
import {
    SPAN,
    PX,
    ZOOM_MIN,
    ZOOM_MAX,
    ZOOM_STEP,
    OFFSET_X,
    BTN_W,
    BTN_H,
    RULER_H,
    LANE_H,
    WIDTH,
    HEIGHT,
    SB_H,
    TOOL_X,
    SLIDER_W,
    COLORS,
    PLAY_COLOR,
    mediaKey,
    clamp,
    laneOf,
    laneRange,
    occupiesLane,
    laneFree,
    clipLen,
    blockRect,
    ghostRect,
    resolveMove,
    probeSnap,
    splitSnap,
    hitTest,
    envLen,
    envField,
    envPts,
    envNormalize,
    envStrengthAtY,
    envStrengthAt,
    envY,
    tokenSnap,
    cloneEnv,
    ENV_MAX,
} from "./timeline_core.js";
import { ensureMedia, thumbSeek, sourceFrames } from "./timeline_media.js";
import {
    stateWidget,
    readState,
    writeState,
    hideStateWidget,
    ensureInputs,
    fixNodeSize,
    removeClip,
    removeClipAudio,
    replaceClipMedia,
    addClipWithMedia,
    splitAt,
    clearAll,
    exportState,
    importState,
} from "./timeline_state.js";
import { drawBlock, drawGhost, drawEnvelope } from "./timeline_draw.js";
import { togglePlay, syncPreview, previewClip } from "./timeline_play.js";

const NODE_NAME = "MiniMaxH3Timeline";

// --- clip context menu ------------------------------------------------------

let _menu = null;
let _menuOverlay = null;

function closeClipMenu() {
    _menu?.remove?.();
    _menuOverlay?.remove?.();
    _menu = null;
    _menuOverlay = null;
}

function openClipMenu(node, widget, clip, idx, x, y, zone, envHit) {
    closeClipMenu();
    widget._menuAt = Date.now();
    // right-clicking the separated audio band of a video clip targets the
    // band only: the only destructive option is deleting that band. Deleting
    // the clip is reached from the video block itself.
    const onGhost =
        clip.kind === "video" &&
        !clip.audio_off &&
        ["audio", "trimAL", "trimAR", "link"].includes(zone);
    const items = [];
    if (onGhost) {
        items.push(["Delete audio track", () => removeClipAudio(node, clip)]);
    } else {
        items.push(["Delete clip", () => removeClip(node, idx)]);
        items.push(["Replace clip\u2026", () => replaceClipMedia(node, clip)]);
    }
    if (envHit?.pt) {
        // right-clicked an envelope point: offer to remove it
        items.push([
            "Remove strength point",
            () => {
                const env = envField(clip, envHit.ghost);
                const i = Array.isArray(env) ? env.indexOf(envHit.pt) : -1;
                if (i >= 0) env.splice(i, 1);
                writeState(node);
                widget.redraw(node);
            },
        ]);
    } else if (envHit) {
        // right-clicked the envelope line: offer to add a point there
        items.push([
            "Add strength point",
            () => {
                const s = widget._scale;
                const r = envHit.ghost ? ghostRect(clip, s) : blockRect(clip, s);
                const env = envNormalize(clip, envHit.ghost);
                const len = envLen(clip, envHit.ghost);
                const rawF = clamp(Math.round((envHit.p[0] - r.x) / s), 0, len);
                const pt = [
                    clip.kind === "video" && !envHit.ghost
                        ? tokenSnap(rawF, len)
                        : rawF,
                    envStrengthAtY(r, envHit.p[1]),
                ];
                env.push(pt);
                env.sort((a, b) => a[0] - b[0]);
                writeState(node);
                widget.redraw(node);
            },
        ]);
    }
    const overlay = document.createElement("div");
    overlay.style.cssText = "position:fixed;inset:0;z-index:3000";
    overlay.addEventListener("pointerdown", closeClipMenu);
    document.body.appendChild(overlay);
    _menuOverlay = overlay;

    const winW = typeof window !== "undefined" ? (window.innerWidth ?? 800) : 800;
    const winH = typeof window !== "undefined" ? (window.innerHeight ?? 600) : 600;
    const menu = document.createElement("div");
    menu.dataset.h3menu = "1";
    menu.style.cssText =
        "position:fixed;z-index:3001;min-width:180px;background:#222;border:1px solid #555;" +
        "border-radius:6px;padding:4px;font:13px sans-serif;color:#ddd;user-select:none;" +
        "box-shadow:0 4px 16px rgba(0,0,0,0.5);left:" +
        Math.min(x, winW - 200) + "px;top:" +
        Math.min(y, winH - items.length * 27 - 10) + "px";
    for (const [label, cb] of items) {
        const row = document.createElement("div");
        const enabled = !!cb;
        row.style.cssText =
            "padding:4px 10px;border-radius:4px;cursor:" + (enabled ? "pointer" : "default") +
            ";color:" + (enabled ? "#ddd" : "#666");
        row.textContent = label;
        if (enabled) {
            row.addEventListener("pointerdown", (e) => {
                e.stopPropagation();
                cb();
                closeClipMenu();
            });
            row.addEventListener("pointerenter", () => {
                row.style.background = "#3a5a80";
            });
            row.addEventListener("pointerleave", () => {
                row.style.background = "transparent";
            });
        }
        menu.appendChild(row);
    }
    document.body.appendChild(menu);
    _menu = menu;
    document.addEventListener("keydown", (e) => {
        if (e.key === "Escape") closeClipMenu();
    }, { once: true, capture: true });
}

function sliderZoom(nd, w, x) {
    const minS = Math.max(0.5, Math.min(ZOOM_MIN, WIDTH / (nd._h3Span ?? SPAN)));
    const tn = clamp((x - (w._sliderX ?? TOOL_X)) / SLIDER_W, 0, 1);
    w._scale = Math.exp(tn * Math.log(ZOOM_MAX / minS)) * minS;
}

// number widget that doubles as an optional input slot: Python may have
// already created it from the INT input (look it up first, and skip creation
// when it was converted to an input)
function ensureNumWidget(node, name, def, store) {
    const slot = node._h3NumWidgets ?? (node._h3NumWidgets = {});
    if (!slot[name]) slot[name] = node.widgets?.find((w) => w.name === name);
    const isInput = node.inputs?.some((i) => i.name === name);
    if (!slot[name] && !isInput) {
        const w = node.addWidget("number", name, def, (v) => {
            store(Math.max(1, Math.round(Number(v) || def)));
            node._h3TimelineWidget?.redraw?.(node);
        }, { min: 1, max: name === "fps" ? 240 : 100000, step: 1 });
        slot[name] = w;
    }
    const w = slot[name];
    if (w && (w.value == null || !Number.isFinite(Number(w.value)))) w.value = def;
    if (w) w.value = Math.max(1, Math.round(Number(w.value) || def));
    return w;
}

function setup(node) {
    hideStateWidget(node);
    node._h3Media ??= new Map();
    node._h3Thumbs ??= new Map();
    node._h3Clips = readState(node);
    if (!node._h3Clips) node._h3Clips = [];
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
            _dragEnv: null,
            _dragFlat: null,
            _hover: null,
            _hoverPos: null,
            _lastDown: 0,
            _lastDownPos: null,
            _dragPlay: false,
            _play: 0,
            _playing: false,
            _ctxs: new Set(),
            _boundCtxs: null,
            computeSize: () => [WIDTH, HEIGHT],
            draw(ctx, nd, width, y, H) {
                // y is the widget row offset inside the node. Canvas
                // renderer (1.0): widget.y (>= 4). Vue WidgetLegacy (2.0):
                // always 1, canvas is widget-local. NaN/0 -> keep last.
                if (ctx?.canvas?.isConnected) this._ctxs.add(ctx);
                this._bindMenu(ctx?.canvas, nd);
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
            _bindMenu(canvas, nd) {
                if (!canvas || this._boundCtxs?.has(canvas)) return;
                (this._boundCtxs ??= new Set()).add(canvas);
                canvas.addEventListener("contextmenu", (e) => {
                    // the mouse() right-down path already opened the menu
                    if (Date.now() - (this._menuAt || 0) < 600) return;
                    e.preventDefault();
                    e.stopImmediatePropagation();
                    const isGraph = canvas === app?.canvas?.canvas;
                    let px;
                    let py;
                    if (isGraph) {
                        try {
                            app.canvas.adjustMouseEvent?.(e);
                        } catch (_) {}
                        const n = this._node ?? nd;
                        px = (e.canvasX ?? e.offsetX) - (n.pos?.[0] ?? 0);
                        py = (e.canvasY ?? e.offsetY) - (n.pos?.[1] ?? 0) - this._yOff;
                    } else {
                        px = e.offsetX;
                        py = e.offsetY - this._yOff;
                    }
                        const hit = hitTest(nd, [px, py], this._scale, this._pan);
                        const idx = hit?.c ? (nd._h3Clips?.indexOf(hit.c) ?? -1) : -1;
                        if (idx >= 0) {
                            const envHit =
                                hit.zone === "envpt" || hit.zone === "envln"
                                    ? { ghost: hit.ghost, p: [px, py], pt: hit.pt }
                                    : null;
                            openClipMenu(nd, this, hit.c, idx, e.clientX, e.clientY, hit.zone, envHit);
                        }
                }, true);
            },
            _clear(ctx) {
                // The litegraph graph canvas repaints the whole graph each
                // frame, so stale pixels are impossible there. Any other
                // canvas (graph body, properties panel, widget canvases) is
                // NOT guaranteed to repaint itself — clear it or translucent
                // redraws smear dragged geometry into stretched trails.
                if (ctx.canvas === app?.canvas?.canvas) return;
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
                nd._h3ScrollWidget?.redraw?.(nd);
            },
            paint(ctx, nd, width, H) {
                const clips = nd._h3Clips || [];
                const s = this._scale;
                const span0 = nd._h3Span ?? SPAN;
                this._pan = clamp(this._pan ?? 0, 0, Math.max(0, OFFSET_X + span0 * s - width + 4));

                // sync fps/total_frames: spinner buttons mutate widget.value
                // without firing the callback, so poll here each paint.
                // When the widget was converted to an input slot, the widget
                // leaves node.widgets and the live value comes from the
                // connected node at graph-execution time; until then the
                // last known value persists.
                const fpsW = nd.widgets?.find((w) => w.name === "fps") || nd._h3FpsWidget;
                if (fpsW && fpsW.value != null && Number.isFinite(Number(fpsW.value))) {
                    this._fps = Math.max(1, Math.round(Number(fpsW.value) || 24));
                }
                const spanW = nd.widgets?.find((w) => w.name === "total_frames") || nd._h3SpanWidget;
                if (spanW && spanW.value != null && Number.isFinite(Number(spanW.value))) {
                    nd._h3Span = Math.max(1, Math.round(Number(spanW.value) || SPAN));
                }

                ctx.fillStyle = "rgba(0,0,0,0.25)";
                ctx.fillRect(0, 0, width, H);
                ctx.strokeStyle = "rgba(255,255,255,0.15)";
                ctx.beginPath();
                ctx.moveTo(0, RULER_H + LANE_H + 0.5);
                ctx.lineTo(width, RULER_H + LANE_H + 0.5);
                ctx.stroke();

                // the ruler labels/ticks scroll with the content; the
                // toolbar and zoom slider are canvas-fixed chrome drawn
                // over them after the restore (their backdrop hides any
                // labels that slid underneath)
                ctx.save();
                ctx.translate(-(this._pan ?? 0), 0);

                const span = nd._h3Span ?? SPAN;
                // VAE valid-length grid: the H3 video VAE only reproduces the
                // grid exactly for windows of 17k+5 frames (5, 22, 39, 56,
                // 73...) — those end on a token boundary with no virtual
                // tail. Other lengths leave the last token held-edge padded,
                // which cuts. The lines mark the valid end frames for a clip
                // starting on a chunk boundary; a clip whose right edge
                // lands on a line is clean.
                ctx.strokeStyle = "rgba(122, 162, 247, 0.4)";
                ctx.lineWidth = 1;
                ctx.beginPath();
                for (let f = 5; f <= span; f += 17) {
                    const x = OFFSET_X + (f - 1) * s + 0.5;
                    ctx.moveTo(x, RULER_H - 4);
                    ctx.lineTo(x, HEIGHT);
                }
                ctx.stroke();
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
                    ctx.fillText(txt, x + 2, RULER_H - 15);
                }
                ctx.restore();

                // opaque backdrop over the toolbar strip (buttons + slider) so ruler
// labels don't bleed through; the number band below stays clear
ctx.fillStyle = "#1a1a2a";
ctx.fillRect(TOOL_X - 4, 0, WIDTH - (TOOL_X - 4), RULER_H - 16);

                // toolbar: text buttons measured at paint time; _btns/_sliderX
                // are reused by btnZone for hit-testing
                ctx.strokeStyle = "#888";
                ctx.fillStyle = "#ddd";
                ctx.font = "10px sans-serif";
                ctx.textAlign = "center";
                ctx.textBaseline = "middle";
                const snapOn = this._snapEnabled ?? true;
                const defs = [
                    ["✂ Split", "split"],
                    ["🧲 Snap", "snap"],
                    [this._playing ? "⏹ Stop" : "▶ Play", "play"],
                    ["F", "unit"],
                    ["−", "out"],
                    ["+", "in"],
                    ["🗑 Clear", "clear"],
                    ["⤓ Export", "export"],
                    ["⤒ Import", "import"],
                ];
                this._btns = [];
                let bx = TOOL_X;
                const btnY = 3;
                for (const [label, zone] of defs) {
                    const w = Math.max(BTN_W, ctx.measureText(label).width + 10);
                    ctx.beginPath();
                    ctx.roundRect(bx + 0.5, btnY, w, BTN_H, 3);
                    ctx.fillStyle =
                        zone === "snap" && snapOn
                            ? "#3a5a80"
                            : zone === "play" && this._playing
                              ? "#3a5a80"
                              : "#333";
                    ctx.fill();
                    ctx.stroke();
                    ctx.fillStyle = "#ddd";
                    ctx.fillText(label, bx + w / 2, btnY + BTN_H / 2 + 0.5);
                    this._btns.push({ zone, x: bx, w });
                    bx += w + 3;
                }
                this._sliderX = bx + 4;

                // zoom slider: log-scale track from minS to ZOOM_MAX
                {
                    const minS = Math.max(0.5, Math.min(ZOOM_MIN, WIDTH / span));
                    const trackY = 3 + BTN_H / 2 + 0.5;
                    const t0 = this._sliderX ?? TOOL_X;
                    const t1 = t0 + SLIDER_W;
                    ctx.fillStyle = "#222";
                    ctx.beginPath();
                    ctx.roundRect(t0, trackY - 2, SLIDER_W, 4, 2);
                    ctx.fill();
                    const tn = Math.log(this._scale / minS) / Math.log(ZOOM_MAX / minS);
                    const kx = t0 + clamp(tn, 0, 1) * SLIDER_W;
                    ctx.fillStyle = "#3a5a80";
                    ctx.beginPath();
                    ctx.arc(clamp(kx, t0, t1), trackY, 4, 0, Math.PI * 2);
                    ctx.fill();
                }

                ctx.fillText("video", 2, RULER_H + 14);
                ctx.fillText("audio", 2, RULER_H + LANE_H + 14);

                ctx.save();
                ctx.translate(-(this._pan ?? 0), 0);

                const envPlayX =
                    this._play != null
                        ? OFFSET_X + Math.max(0, this._play) * s
                        : this._frame != null
                          ? OFFSET_X + (this._frame - 1) * s
                          : null;
                for (const c of clips) {
                    const media = c.file ? ensureMedia(nd, c) : null;
                    if (c.kind === "video") drawGhost(ctx, c, s, nd);
                    if (c.kind === "image") {
                        drawBlock(ctx, COLORS.image, `img ${c.id}`, blockRect(c, s), false, media, nd, c);
                        drawEnvelope(ctx, blockRect(c, s), c, s, false, envPlayX);
                    } else if (c.kind === "video") {
                        if (!c.audio_off) drawEnvelope(ctx, ghostRect(c, s), c, s, true, envPlayX);
                        drawBlock(ctx, COLORS.video, `video ${c.id}`, blockRect(c, s), false, media, nd, c);
                        drawEnvelope(ctx, blockRect(c, s), c, s, false, envPlayX);
                    } else {
                        drawBlock(ctx, COLORS.audio, `audio ${c.id}`, blockRect(c, s), false, media, nd, c);
                        drawEnvelope(ctx, blockRect(c, s), c, s, false, envPlayX);
                    }
                }

                // end-line + dim everything past it
                {
                    const ex = OFFSET_X + span * s;
                    if (ex < WIDTH) {
                        ctx.fillStyle = "rgba(0,0,0,0.45)";
                        ctx.fillRect(ex, RULER_H, WIDTH - ex, H - RULER_H);
                    }
                    ctx.strokeStyle = "#f8932b";
                    ctx.beginPath();
                    ctx.moveTo(ex + 0.5, RULER_H - 6);
                    ctx.lineTo(ex + 0.5, H);
                    ctx.stroke();
                }

                // video thumbnails: the playing clip's element runs on its
                // own (see syncPreview); every other clip seeks to the frame
                // the playhead currently sits on when over the clip,
                // otherwise to the clip's source start.
                const playFrame = this._play ?? 0;
                const playClip = previewClip(nd);
                const playing = !!this._playing;
                for (const c of clips) {
                    if (c.kind !== "video" || !c.file) continue;
                    const key = mediaKey(c.file);
                    const m = nd._h3Media?.get(key);
                    if (m?.kind !== "video") continue;
                    if (c === playClip && playing) continue;
                    let target;
                    if (c === playClip) {
                        target =
                            (playFrame - (Number(c.start) - 1) + (Number(c.src_start) || 0)) /
                            fps;
                    } else {
                        target = (Number(c.src_start) || 0) / fps;
                    }
                    thumbSeek(nd, c, m, Math.max(0, target));
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
                        ctx.beginPath();
                        ctx.roundRect(pr.x - 2, pr.y - 2, pr.w + 4, pr.h + 4, 3);
                        ctx.fill();
                        ctx.globalAlpha = 1;
                    }
                }
                ctx.restore();

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

                // envelope affordances: live value chip while dragging, a
                // "+" marker where a double-click adds a point, and a one
                // line hint for what the hovered zone does
                if (this._dragEnv) {
                    const de = this._dragEnv;
                    const r = de.ghost ? ghostRect(de.c, this._scale) : blockRect(de.c, this._scale);
                    const bx = clamp(r.x + r.w - 2, 0, WIDTH);
                    ctx.fillStyle = "rgba(0,0,0,0.65)";
                    ctx.beginPath();
                    ctx.roundRect(bx - 40, r.y + 4, 38, 13, 3);
                    ctx.fill();
                    ctx.fillStyle = "#66ff66";
                    ctx.font = "9px sans-serif";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(de.pt[1].toFixed(2), bx - 21, r.y + 11);
                }
                if (this._dragFlat) {
                    const dfl = this._dragFlat;
                    const r = dfl.ghost ? ghostRect(dfl.c, this._scale) : blockRect(dfl.c, this._scale);
                    const f = this._hoverPos ? (this._hoverPos[0] - r.x) / this._scale : 0;
                    const v = envStrengthAt(dfl.c, dfl.ghost, f);
                    const bx = clamp(r.x + r.w - 2, 0, WIDTH);
                    ctx.fillStyle = "rgba(0,0,0,0.65)";
                    ctx.beginPath();
                    ctx.roundRect(bx - 40, r.y + 4, 38, 13, 3);
                    ctx.fill();
                    ctx.fillStyle = "#66ff66";
                    ctx.font = "9px sans-serif";
                    ctx.textAlign = "center";
                    ctx.textBaseline = "middle";
                    ctx.fillText(v.toFixed(2), bx - 21, r.y + 11);
                }
                if (!this._drag && !this._dragEnv && !this._dragFlat && this._hover?.c && this._hoverPos) {
                    const h = this._hover;
                    const r = h.ghost ? ghostRect(h.c, this._scale) : blockRect(h.c, this._scale);
                    if (h.zone === "envln") {
                        const x = clamp(this._hoverPos[0], r.x + 1, r.x + r.w - 1);
                        const y = envY(r, envStrengthAt(h.c, h.ghost, (x - r.x) / this._scale));
                        ctx.strokeStyle = "#66ff66";
                        ctx.lineWidth = 2;
                        ctx.beginPath();
                        ctx.moveTo(x - 5, y - 5);
                        ctx.lineTo(x + 5, y + 5);
                        ctx.moveTo(x + 5, y - 5);
                        ctx.lineTo(x - 5, y + 5);
                        ctx.stroke();
                    }
                    const hint =
                        h.zone === "envpt"
                            ? "drag: value · right-click: remove"
                            : h.zone === "envln"
                              ? "drag: level · double-click: add point"
                              : null;
                    if (hint) {
                        ctx.fillStyle = "rgba(0,0,0,0.6)";
                        ctx.font = "9px sans-serif";
                        const tw = ctx.measureText(hint).width + 10;
                        ctx.beginPath();
                        ctx.roundRect(4, H - 18, tw, 14, 3);
                        ctx.fill();
                        ctx.fillStyle = "#ffe08a";
                        ctx.textAlign = "left";
                        ctx.textBaseline = "middle";
                        ctx.fillText(hint, 9, H - 11);
                    }
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
                // content position: the timeline scrolls under the fixed
                // toolbar, so frame math uses the panned x
                const cx = pos[0] + (this._pan ?? 0);
                const type = e.type || "";
                if (type.endsWith("down") && e.button === 2) {
                    // right click opens the clip menu; on the strength
                    // envelope it carries an add/remove point entry
                    const hit = hitTest(nd, p, this._scale, this._pan);
                    if (!hit?.c) return false;
                    e.preventDefault();
                    const idx = nd._h3Clips?.indexOf(hit.c) ?? -1;
                    if (idx >= 0) {
                        const ghost = !!(hit.ghost || hit.zone === "audio" || hit.zone === "trimAL" || hit.zone === "trimAR");
                        const envHit = { ghost, p: [cx, p[1]], pt: hit.pt };
                        openClipMenu(nd, this, hit.c, idx, e.clientX ?? pos[0], e.clientY ?? pos[1], hit.zone, envHit);
                    }
                    return true;
                }
                if (type.endsWith("down") && e.button === 0) {
                    const now = performance.now();
                    const dbl =
                        now - (this._lastDown ?? 0) < 450 &&
                        this._lastDownPos &&
                        Math.hypot(p[0] - this._lastDownPos[0], p[1] - this._lastDownPos[1]) < 12;
                    this._lastDown = now;
                    this._lastDownPos = p;
                    this._dragged = false;
                    const hit = hitTest(nd, p, this._scale, this._pan);
                    if (!hit) return false;
                    e.preventDefault();
                    if (hit.zone === "slider") {
                        this._dragSlider = true;
                        sliderZoom(nd, this, p[0]);
                        this.redraw(nd);
                        return true;
                    }
                    if (hit.zone === "in" || hit.zone === "out") {
                        const f = hit.zone === "in" ? 1 / ZOOM_STEP : ZOOM_STEP;
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
                    if (hit.zone === "clear") {
                        clearAll(nd);
                        this.redraw(nd);
                        return true;
                    }
                    if (hit.zone === "export") {
                        exportState(nd);
                        return true;
                    }
                    if (hit.zone === "import") {
                        importState(nd);
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
                        const v = Math.max(0, Math.round((cx - OFFSET_X) / s));
                        this._play = splitSnap(nd, v, s);
                        this._frame = null;
                        if (!this._playing) syncPreview(nd);
                        this.redraw(nd);
                        return true;
                    }
                    if (hit.zone === "envpt" || hit.zone === "envln" || (dbl && hit.c)) {
                        // strength envelope: double-click a point to remove
                        // it or the block/line to add one, drag a point, or drag
                        // the flat line to set the flat level when no points
                        // exist
                        const s = this._scale;
                        const c = hit.c;
                        const ghost = !!(hit.ghost || hit.zone === "audio" || hit.zone === "trimAL" || hit.zone === "trimAR");
                        const r = ghost ? ghostRect(c, s) : blockRect(c, s);
                        const len = envLen(c, ghost);
                        if (hit.zone === "envpt" && dbl) {
                            const env = envField(c, ghost);
                            const i = Array.isArray(env) ? env.indexOf(hit.pt) : -1;
                            if (i >= 0) env.splice(i, 1);
                            writeState(nd);
                            this.redraw(nd);
                            return true;
                        }
                        if (hit.zone === "envpt") {
                            this._dragEnv = { c, ghost, pt: hit.pt, len };
                        } else if (dbl) {
                            const env = envNormalize(c, ghost);
                            const rawF = clamp(Math.round((cx - r.x) / s), 0, len);
                            const pt = [
                                c.kind === "video" && !ghost
                                    ? tokenSnap(rawF, len)
                                    : rawF,
                                envStrengthAtY(r, p[1]),
                            ];
                            env.push(pt);
                            env.sort((a, b) => a[0] - b[0]);
                            this._dragEnv = { c, ghost, pt, len };
                        } else if (!envPts(c, ghost).length && hit.zone === "envln") {
                            this._dragFlat = { c, ghost };
                        } else {
                            return true;
                        }
                        writeState(nd);
                        this.redraw(nd);
                        return true;
                    }
                    if (hit.zone === "link") {
                        // unlinking freezes the ghost at its current spot so
                        // later edits to the video no longer move it; the
                        // band's strength/env are frozen as copies too, so
                        // it stops following the video's curve.
                        if (hit.c.audio_link) {
                            hit.c.audio_start = hit.c.start;
                            hit.c.audio_len = hit.c.len;
                            hit.c.audio_strength = Number.isFinite(Number(hit.c.strength))
                                ? Number(hit.c.strength)
                                : ENV_MAX;
                            hit.c.audio_env = cloneEnv(hit.c.env);
                            // the band freezes its own slice of the file, so
                            // trimming the video no longer moves the sound
                            hit.c.audio_src_start = Number(hit.c.src_start) || 0;
                        } else {
                            delete hit.c.audio_start;
                            delete hit.c.audio_len;
                            delete hit.c.audio_src_start;
                        }
                        hit.c.audio_link = !hit.c.audio_link;
                        writeState(nd);
                    } else if (hit.c) {
                        const audioEdit =
                            hit.c.kind === "video" &&
                            !hit.c.audio_link &&
                            (hit.zone === "audio" || hit.zone === "trimAL" || hit.zone === "trimAR");
                        this._drag = {
                            ...hit,
                            grab: p[0],
                            startAt: Number(
                                audioEdit ? hit.c.audio_start ?? hit.c.start : hit.c.start,
                            ),
                            lenAt: Number(
                                audioEdit ? hit.c.audio_len ?? hit.c.len ?? 22 : hit.c.len ?? 22,
                            ),
                            srcAt: Number(hit.c.src_start) || 0,
                        };
                    } else {
                        return true;
                    }
                    return true;
                }
                if (type.includes("move")) {
                    nd._h3Hovered = true;
                    this._hover = hitTest(nd, p, this._scale, this._pan);
                    this._hoverPos = [cx, p[1]];
                    if (this._drag || this._dragEnv || this._dragFlat || this._dragPlay || this._dragSlider) {
                        if (this._lastDownPos && Math.hypot(p[0] - this._lastDownPos[0], p[1] - this._lastDownPos[1]) > 8) {
                            this._dragged = true;
                        }
                    }
                    if (this._dragSlider) {
                        sliderZoom(nd, this, p[0]);
                        this.redraw(nd);
                        return true;
                    }
                    if (this._dragEnv) {
                        const s = this._scale;
                        const de = this._dragEnv;
                        const r = de.ghost ? ghostRect(de.c, s) : blockRect(de.c, s);
                        const rawF = clamp(Math.round((cx - r.x) / s), 0, de.len);
                        de.pt[0] =
                            de.c.kind === "video" && !de.ghost
                                ? tokenSnap(rawF, de.len)
                                : rawF;
                        de.pt[1] = envStrengthAtY(r, p[1]);
                        envField(de.c, de.ghost).sort((a, b) => a[0] - b[0]);
                        writeState(nd);
                        this.redraw(nd);
                        return true;
                    }
                    if (this._dragFlat) {
                        const s = this._scale;
                        const dfl = this._dragFlat;
                        const r = dfl.ghost ? ghostRect(dfl.c, s) : blockRect(dfl.c, s);
                        dfl.c[
                            dfl.ghost &&
                            (Array.isArray(dfl.c.audio_env) || !dfl.c.audio_link)
                                ? "audio_strength"
                                : "strength"
                        ] = envStrengthAtY(r, p[1]);
                        writeState(nd);
                        this.redraw(nd);
                        return true;
                    }
                    const d = this._drag;
                    if (this._dragPlay) {
                        const s = this._scale;
                        const v = Math.max(0, Math.round((p[0] - OFFSET_X) / s));
                        this._play = splitSnap(nd, v, s);
                        if (!this._playing) syncPreview(nd);
                        this.redraw(nd);
                        return true;
                    }
                    if (d) {
                        const s = this._scale;
                        const step = Math.round((p[0] - d.grab) / s);
                        const audioEdit =
                            d.zone === "audio" || d.zone === "trimAL" || d.zone === "trimAR";
                        const lane = audioEdit ? 1 : laneOf(d.c.kind);
                        if (d.zone === "move" || d.zone === "audio") {
                            const len = audioEdit
                                ? (d.c.audio_len ?? d.c.len ?? 22)
                                : clipLen(d.c);
                            const s2 = resolveMove(
                                nd,
                                d.c,
                                lane,
                                Math.max(1, d.startAt + step),
                                len,
                                d.grab / s + 1,
                                s,
                            );
                            if (d.zone === "audio") d.c.audio_start = s2;
                            else d.c.start = s2;
                            this._frame = s2;
                        } else if (d.zone === "trimR" || d.zone === "trimAR") {
                            const lanes =
                                lane === 0 && d.c.kind === "video" && d.c.audio_link && !d.c.audio_off
                                    ? [0, 1]
                                    : [lane];
                            let len = Math.max(1, d.lenAt + step);
                            if (nd._h3TimelineWidget?._snapEnabled !== false) {
                                const end = probeSnap(nd, d.startAt + len, s);
                                let snapped = Math.max(1, end - d.startAt);
                                if (lanes.every((L) => laneFree(nd, d.c, L, d.startAt, snapped))) {
                                    len = snapped;
                                }
                            }
                            // no clip can ever grow past the end of its source.
                            if (d.c.file) {
                                const srcMax = sourceFrames(nd, d.c) -
                                    (Number(d.c.src_start) || 0);
                                if (isFinite(srcMax) && srcMax > 0) len = Math.min(len, srcMax);
                            }
                            for (const L of lanes) {
                                for (const o of nd._h3Clips) {
                                    if (o === d.c || !occupiesLane(o, L)) continue;
                                    const r = laneRange(o, L);
                                    if (r.s >= d.startAt) len = Math.min(len, r.s - d.startAt);
                                    else if (r.e > d.startAt) len = Math.min(len, r.e - d.startAt);
                                }
                            }
                            len = Math.max(1, len);
                            if (d.zone === "trimAR") d.c.audio_len = len;
                            else d.c.len = len;
                            this._frame = d.startAt + len - 1;
                        } else if (d.zone === "trimL" || d.zone === "trimAL") {
                            const lanes =
                                lane === 0 && d.c.kind === "video" && d.c.audio_link && !d.c.audio_off
                                    ? [0, 1]
                                    : [lane];
                            const rawS2 = clamp(d.startAt + step, 1, d.startAt + d.lenAt - 1);
                            let s2 = rawS2;
                            if (nd._h3TimelineWidget?._snapEnabled !== false) {
                                s2 = probeSnap(nd, s2, s);
                            }
                            // dragging the left edge shifts the source window:
                            // the same source frame stays under the grab point
                            // unless src_start would fall below its floor (0
                            // for a plain clip, the split cut for halves) —
                            // past that the clip would play frames it no
                            // longer owns, so the edge stops there. The ghost
                            // band keeps its frozen source window instead.
                            const origSrc = d.srcAt;
                            if (d.zone !== "trimAL" && d.c.file) {
                                const srcFloor = Number(d.c.src_floor) || 0;
                                const minStart = d.startAt - (origSrc - srcFloor);
                                s2 = Math.max(s2, minStart);
                                const srcMax = sourceFrames(nd, d.c) - origSrc;
                                if (isFinite(srcMax) && srcMax > 0) {
                                    s2 = Math.min(s2, d.startAt + srcMax - 1);
                                }
                            }
                            let right = d.startAt + d.lenAt;
                            for (let guard = 0; guard < nd._h3Clips.length; guard++) {
                                if (s2 >= right) break;
                                let changed = false;
                                for (const L of lanes) {
                                    for (const o of nd._h3Clips) {
                                        if (o === d.c || !occupiesLane(o, L)) continue;
                                        const r = laneRange(o, L);
                                        if (r.s <= s2 && r.e > s2) {
                                            s2 = r.e;
                                            changed = true;
                                        } else if (r.s > s2 && r.s < right) {
                                            right = r.s;
                                            changed = true;
                                        }
                                    }
                                }
                                if (!changed) break;
                            }
                            s2 = Math.min(s2, Math.max(1, right - 1));
                            let len = Math.max(1, right - s2);
                            // the snap (or push) can strand the edge inside a
                            // clip that spans the whole remaining length:
                            // fall back to the un-snapped drag position.
                            if (!lanes.every((L) => laneFree(nd, d.c, L, s2, len))) {
                                s2 = rawS2;
                                len = Math.max(1, d.startAt + d.lenAt - s2);
                            }
                            if (d.zone === "trimAL") {
                                d.c.audio_start = s2;
                                d.c.audio_len = len;
                            } else {
                                d.c.start = s2;
                                d.c.len = len;
                                if (d.c.file) {
                                    d.c.src_start = Math.max(0, origSrc + (s2 - d.startAt));
                                }
                            }
                            this._frame = s2;
                        }
                        writeState(nd);
                        this.redraw(nd);
                    } else {
                        this._frame = Math.max(1, Math.round((cx - OFFSET_X) / this._scale + 1));
                        this.redraw(nd);
                    }
                    return true;
                }
                if (type.includes("up")) {
                    if (this._dragged) {
                        this._lastDown = 0;
                    }
                    this._drag = null;
                    this._hover = null;
                    this._hoverPos = null;
                    this._dragPlay = false;
                    this._dragSlider = false;
                    this._dragEnv = null;
                    this._dragFlat = null;
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

    // horizontal scrollbar under the timeline: pans the timeline content
    // (the widget's _pan) when zoomed in past the canvas width
    if (!node._h3ScrollWidget) {
        const sb = {
            name: "h3_scrollbar",
            type: "h3_scrollbar",
            width: WIDTH,
            computedHeight: SB_H,
            y: 0,
            _yOff: 1,
            _rowOf: 1,
            _dragOff: null,
            _geo: null,
            _ctxs: new Set(),
            computeSize: () => [WIDTH, SB_H],
            draw(ctx, nd, width, y, H) {
                if (ctx?.canvas?.isConnected) this._ctxs.add(ctx);
                if (Number.isFinite(y)) {
                    if (y >= 4) this._yOff = y;
                    else if (y === 1) this._yOff = 1;
                }
                this.paint(ctx, nd, WIDTH, SB_H);
            },
            redraw(nd) {
                for (const ctx of this._ctxs) {
                    if (!ctx.canvas || !ctx.canvas.isConnected) {
                        this._ctxs.delete(ctx);
                        continue;
                    }
                    this.paint(ctx, nd, WIDTH, SB_H);
                }
            },
            paint(ctx, nd, width, H) {
                const tw = nd._h3TimelineWidget;
                const s = tw?._scale ?? PX;
                const span = nd._h3Span ?? SPAN;
                const content = OFFSET_X + span * s;
                const maxPan = Math.max(0, content - width);
                if (tw) tw._pan = clamp(tw._pan ?? 0, 0, maxPan);
                ctx.clearRect(0, 0, width, H);
                ctx.fillStyle = "#1a1a2a";
                ctx.fillRect(0, 0, width, H);
                ctx.fillStyle = "#333";
                ctx.beginPath();
                ctx.roundRect(2, 2, width - 4, H - 4, 3);
                ctx.fill();
                const twW = Math.max(28, Math.min(width - 4, Math.round(width * width / content)));
                const twX = maxPan > 0 && tw
                    ? 2 + (tw._pan / maxPan) * ((width - 4) - twW)
                    : 2;
                ctx.fillStyle = "#3a5a80";
                ctx.beginPath();
                ctx.roundRect(twX, 3, twW, H - 6, 2);
                ctx.fill();
                this._geo = { maxPan, twX, twW, width };
            },
            mouse(e, pos, nd) {
                let y = pos[1] - this._yOff;
                if (this._yOff <= 4 && y > SB_H + 4) y = pos[1] - this._rowOf;
                const p = [pos[0], y];
                const type = e.type || "";
                const tw = nd._h3TimelineWidget;
                const g = this._geo;
                if (!tw || !g || g.maxPan <= 0) return false;
                if (type.endsWith("down") && e.button === 0) {
                    e.preventDefault();
                    this._dragOff = p[0] - g.twX;
                    this._setPan(nd, tw, p[0] - this._dragOff, g);
                    return true;
                }
                if (type.includes("move") && this._dragOff != null) {
                    this._setPan(nd, tw, p[0] - this._dragOff, g);
                    return true;
                }
                if (type.includes("up")) {
                    this._dragOff = null;
                    return true;
                }
                return false;
            },
            _setPan(nd, tw, x, g) {
                const range = (g.width - 4) - g.twW;
                tw._pan = clamp(Math.round(((x - 2) / range) * g.maxPan), 0, g.maxPan);
                tw.redraw?.(nd);
                this.redraw(nd);
            },
        };
        sb.options = { serialize: false };
        sb.serialize = false;
        node.addCustomWidget(sb);
        node._h3ScrollWidget = sb;
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

    // fps widget: Python may have already created it from the optional
    // INT input, so look it up first
    node._h3FpsWidget = ensureNumWidget(node, "fps", 24, (v) => {
        node._h3TimelineWidget._fps = v;
    });
    // total_frames widget: same pattern as fps
    const spanW = ensureNumWidget(node, "total_frames", SPAN, (v) => {
        node._h3Span = v;
    });
    node._h3SpanWidget = spanW;
    if (spanW) node._h3Span = Math.max(1, Math.round(Number(spanW.value) || SPAN));
    else node._h3Span = node._h3Span ?? SPAN;

    const tw = node._h3TimelineWidget;
    if (tw) {
        tw._fps = node._h3FpsWidget?.value ?? 24;
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