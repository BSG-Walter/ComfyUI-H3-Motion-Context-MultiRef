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
    TOOL_X,
    COLORS,
    PLAY_COLOR,
    mediaKey,
    clamp,
    laneOf,
    laneRange,
    blockRect,
    resolveMove,
    probeSnap,
    splitSnap,
    hitTest,
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
} from "./timeline_state.js";
import { roundRect, drawBlock, drawGhost } from "./timeline_draw.js";
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

function openClipMenu(node, widget, clip, idx, x, y, zone) {
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
    items.push(
        ["Copy clip", null],
        ["Cut clip", null],
        ["Duplicate", null],
        ["Move up", null],
        ["Move down", null],
        ["Move to playhead", null],
        ["Change strength", null],
    );
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
            _hover: null,
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
                        const hit = hitTest(nd, [px, py], this._scale);
                        const idx = hit?.c ? (nd._h3Clips?.indexOf(hit.c) ?? -1) : -1;
                        if (idx >= 0) {
                            openClipMenu(nd, this, hit.c, idx, e.clientX, e.clientY, hit.zone);
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
                        target = (Number(c.start) - 1 + (Number(c.src_start) || 0)) / fps;
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
                if (type.endsWith("down") && e.button === 2) {
                    // right click on a clip: open the clip context menu.
                    const hit = hitTest(nd, p, this._scale);
                    if (!hit?.c) return false;
                    e.preventDefault();
                    const idx = nd._h3Clips?.indexOf(hit.c) ?? -1;
                    if (idx >= 0) {
                        openClipMenu(nd, this, hit.c, idx, e.clientX ?? pos[0], e.clientY ?? pos[1], hit.zone);
                    }
                    return true;
                }
                if (type.endsWith("down") && e.button === 0) {
                    const hit = hitTest(nd, p, this._scale);
                    if (!hit) return false;
                    e.preventDefault();
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
                        // unlinking freezes the ghost at its current spot so
                        // later edits to the video no longer move it; linking
                        // drops the frozen position and follows the video again.
                        if (hit.c.audio_link) {
                            hit.c.audio_start = hit.c.start;
                            hit.c.audio_len = hit.c.len;
                        } else {
                            delete hit.c.audio_start;
                            delete hit.c.audio_len;
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
                        const audioEdit =
                            d.zone === "audio" || d.zone === "trimAL" || d.zone === "trimAR";
                        const lane = audioEdit ? 1 : laneOf(d.c.kind);
                        if (d.zone === "move" || d.zone === "audio") {
                            const img = d.c.kind === "image";
                            const len = img
                                ? 3
                                : audioEdit
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
                        } else if (d.zone === "trimR" || d.zone === "trimAR") {
                            let len = clamp(d.lenAt + step, 1, span - d.startAt + 1);
                            if (nd._h3TimelineWidget?._snapEnabled !== false) {
                                const end = probeSnap(nd, d.startAt + len, s);
                                len = clamp(end - d.startAt, 1, span - d.startAt + 1);
                            }
                            // no clip can ever grow past the end of its source.
                            if (d.c.file) {
                                const srcMax = sourceFrames(nd, d.c) -
                                    (Number(d.c.src_start) || 0);
                                if (isFinite(srcMax) && srcMax > 0) len = Math.min(len, srcMax);
                            }
                            for (const o of nd._h3Clips) {
                                if (o === d.c || laneOf(o.kind) !== lane) continue;
                                const r = laneRange(o, lane);
                                if (r.s >= d.startAt) len = Math.min(len, r.s - d.startAt);
                            }
                            len = Math.max(1, len);
                            if (d.zone === "trimAR") d.c.audio_len = len;
                            else d.c.len = len;
                            this._frame = d.startAt + len - 1;
                        } else if (d.zone === "trimL" || d.zone === "trimAL") {
                            let s2 = clamp(d.startAt + step, 1, d.startAt + d.lenAt - 1);
                            if (nd._h3TimelineWidget?._snapEnabled !== false) {
                                s2 = probeSnap(nd, s2, s);
                            }
                            // dragging the left edge shifts the source window:
                            // the same source frame stays under the grab point
                            // unless src_start would go negative. The ghost
                            // keeps its shared source window fixed instead.
                            const origSrc = Number(d.c.src_start) || 0;
                            if (d.zone !== "trimAL" && d.c.file) {
                                const minStart = d.startAt - origSrc;
                                s2 = Math.max(s2, minStart);
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
                            let len = Math.max(1, right - s2);
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