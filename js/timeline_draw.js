// Canvas painting helpers for the H3 timeline widget.

import { bandSrc, bandGeom, clamp, COLORS, ghostRect, WIDTH, envFlat, envLen, envPts, envStrengthAt, envX, envY, videoTokenStarts } from "./timeline_core.js";
import { ensureMedia } from "./timeline_media.js";

function paintCover(ctx, el, r) {
    if (!el || (!el.videoWidth && !el.naturalWidth)) return;
    const w = el.videoWidth || el.naturalWidth;
    const h = el.videoHeight || el.naturalHeight;
    const s = Math.min(r.w / w, r.h / h);
    const dw = w * s;
    const dh = h * s;
    ctx.drawImage(el, r.x + (r.w - dw) / 2, r.y + (r.h - dh) / 2, dw, dh);
}

// draws the peaks of the SOURCE window [src_start, src_start+len) across
// the block. Peaks are re-bucketed from the decoded buffer for the window
// itself (one bucket per block pixel), so a short clip over a long file
// still draws a dense waveform and trimming crops cleanly instead of
// zooming the whole-file peaks into nothing.
export function paintWaveform(ctx, m, r, ghost, node, clip) {
    const buf = m?.buffer;
    if (!buf || !clip) return;
    const { len } = bandGeom(clip);
    if (len < 1) return;
    const srcStart = bandSrc(clip);
    const fps = node?._h3TimelineWidget?._fps || 24;
    const sr = buf.sampleRate || 48000;
    const s0 = Math.floor((srcStart / fps) * sr);
    const s1 = Math.ceil(((srcStart + len) / fps) * sr);
    const data = buf.getChannelData(0).subarray(
        Math.max(0, s0),
        Math.min(buf.length, s1),
    );
    if (data.length < 2) return;
    const n = Math.max(2, Math.min(1024, Math.floor(r.w) || 64));
    const step = Math.max(1, Math.floor(data.length / n));
    ctx.fillStyle = "rgba(0,0,0," + (ghost ? 0.25 : 0.35) + ")";
    ctx.fillRect(r.x, r.y, r.w, r.h);
    ctx.fillStyle = ghost ? "rgba(255,255,255,0.5)" : "#fff";
    const cw = (r.w - 4) / (n - 1);
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
        const amp = Math.min(1, Math.max(0.02, mx - mn));
        const bh = Math.max(1.5, amp * (r.h - 6));
        const x = r.x + 2 + i * cw;
        ctx.fillRect(x, r.y + (r.h - bh) / 2, Math.max(1, cw * 0.7), bh);
    }
}

// clip the block and draw a media element cover-fit inside it
function paintThumb(ctx, el, r) {
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(r.x, r.y, r.w, r.h, 3);
    ctx.clip();
    ctx.fillStyle = "#000";
    ctx.fillRect(r.x, r.y, r.w, r.h);
    paintCover(ctx, el, r);
    ctx.restore();
}

export function drawBlock(ctx, color, label, r, ghost, media, node, clip) {
    ctx.globalAlpha = ghost ? 0.35 : 0.55;
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.roundRect(r.x, r.y, r.w, r.h, 3);
    ctx.fill();
    ctx.globalAlpha = 1;
    if (media?.kind === "audio") {
        paintWaveform(ctx, media, r, ghost, node, clip);
    } else if (!ghost && media?.kind === "image" && media.img) {
        paintThumb(ctx, media.img, r);
    } else if (!ghost && media?.kind === "video" && node?._h3Thumbs?.get(clip?.id)?.el) {
        paintThumb(ctx, node._h3Thumbs.get(clip.id).el, r);
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

export function drawGhost(ctx, c, s, node) {
    if (c.audio_off) return; // the separated audio band was deleted
    const g = ghostRect(c, s);
    drawBlock(ctx, COLORS.audio, `♪ ${c.id}`, g, true, null, node);
    const m = c.file ? ensureMedia(node, c) : null;
    if (m?.buffer) paintWaveform(ctx, m, g, true, node, c);
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

const ENV_COLOR = "#66ff66";

function envChip(ctx, x, y, text) {
    const font = "8px sans-serif";
    ctx.font = font;
    const tw = ctx.measureText(text).width + 8;
    const cx = clamp(x - tw, 0, WIDTH - tw);
    ctx.fillStyle = "rgba(0,0,0,0.65)";
    ctx.beginPath();
    ctx.roundRect(cx, y - 6, tw, 11, 3);
    ctx.fill();
    ctx.fillStyle = ENV_COLOR;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(text, cx + tw / 2, y + 0.5);
}

// the block's strength envelope as a green automation line: flat at the
// block's own flat strength when no points, a polyline through the points
// otherwise, with each point labeled and a live value chip at the right
// end. When `playX` (pixel x) sits across the block, a dot on the curve
// plus a chip show the strength the playhead is currently touching. A dark
// backing stroke keeps it readable over thumbnails and waveforms; the
// ghost band draws it dimmed.
export function drawEnvelope(ctx, r, c, s, ghost, playX) {
    const flat = envFlat(c, ghost);
    const L = envLen(c, ghost);
    const pts = envPts(c, ghost).filter((p) => p[0] >= 0 && p[0] <= L);
    // video blocks: the node samples the strength once per latent token at
    // the token's first frame (5 tokens per 17 frames), so draw the applied
    // stepped curve instead of a smooth polyline; everything else (image
    // clips, audio bands) is truly linear between points.
    const stepped = c.kind === "video" && !ghost;
    const seg = [];
    if (!pts.length) {
        seg.push([r.x + 0.5, envY(r, flat)], [r.x + r.w - 0.5, envY(r, flat)]);
    } else if (stepped) {
        let lastY = envY(r, envStrengthAt(c, ghost, 0));
        seg.push([r.x + 0.5, lastY]);
        for (const t of videoTokenStarts(L + 1).slice(1)) {
            const x = envX(r, t, s);
            const y = envY(r, envStrengthAt(c, ghost, t));
            if (y !== lastY) seg.push([x, lastY], [x, y]);
            lastY = y;
        }
        seg.push([r.x + r.w - 0.5, lastY]);
    } else {
        seg.push([r.x + 0.5, envY(r, pts[0][1])]);
        for (const [f, v] of pts) seg.push([envX(r, f, s), envY(r, v)]);
        seg.push([r.x + r.w - 0.5, envY(r, pts[pts.length - 1][1])]);
    }
    const trace = () => {
        ctx.beginPath();
        seg.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
        ctx.stroke();
    };
    ctx.save();
    ctx.globalAlpha = ghost ? 0.45 : 0.95;
    ctx.lineWidth = 4;
    ctx.strokeStyle = "rgba(0,0,0,0.45)";
    trace();
    ctx.lineWidth = ghost ? 1 : 1.5;
    ctx.strokeStyle = ENV_COLOR;
    trace();
    ctx.font = "8px sans-serif";
    ctx.textAlign = "center";
    ctx.textBaseline = "bottom";
    for (const [f, v] of pts) {
        ctx.beginPath();
        ctx.arc(envX(r, f, s), envY(r, v), ghost ? 2.5 : 3.5, 0, Math.PI * 2);
        ctx.fillStyle = ENV_COLOR;
        ctx.fill();
        ctx.fillStyle = ghost ? "#aafaa0" : "#d8ffd4";
        ctx.fillText(v.toFixed(2), envX(r, f, s), envY(r, v) - 5);
    }
    if (playX != null && playX >= r.x && playX <= r.x + r.w) {
        const v = envStrengthAt(c, ghost, (playX - r.x) / s);
        const y = envY(r, v);
        ctx.beginPath();
        ctx.arc(playX, y, ghost ? 3 : 4, 0, Math.PI * 2);
        ctx.fillStyle = "#ffd166";
        ctx.fill();
        envChip(ctx, playX + 6, y, v.toFixed(2));
    }
    const endV = pts.length ? pts[pts.length - 1][1] : flat;
    if (r.w > 40) envChip(ctx, r.x + r.w - 2, envY(r, endV), endV.toFixed(2));
    ctx.restore();
}
