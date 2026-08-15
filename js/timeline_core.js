// Shared constants and pure geometry for the H3 timeline widget.

import { api } from "../../../scripts/api.js";

export const STATE_NAME = "timeline_state";
export const MAX_CLIPS = 32;

export const SPAN = 240; // ruler length in frames
export const PX = 3.5; // pixels per frame (base zoom)
export const ZOOM_MIN = 1; // px/frame limits for zoom buttons
export const ZOOM_MAX = 24;
export const ZOOM_STEP = 1.25;
export const SNAP_PX = 12; // magnet radius for clip snapping
export const SNAP_PLAY_PX = 24; // wider magnet radius when clips snap to the playhead
export const OFFSET_X = 16; // left margin offset
export const BTN_W = 16;
export const BTN_H = 14;
export const RULER_H = 34;
export const LANE_H = 68;
export const PAD = 4;
export const WIDTH = 840;
export const HEIGHT = RULER_H + 2 * LANE_H + PAD;
export const SB_H = 16; // horizontal scrollbar height under the timeline
export const TOOL_X = WIDTH - 540; // first toolbar button x (measured widths, so slack)
export const SLIDER_W = 74; // zoom slider track width

export const COLORS = {
    image: "#7aa2f7",
    video: "#9ece6a",
    audio: "#f7768e",
};
export const PLAY_COLOR = "#ff5252";

export const defaults = {
    image: () => ({ kind: "image", start: 1, strength: 1, len: 22 }),
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

const IMAGE_EXT = ["png", "jpg", "jpeg", "webp", "bmp"];
const VIDEO_EXT = ["mp4", "mov", "webm", "mkv", "m4v", "avi", "gif"];
const AUDIO_EXT = ["mp3", "wav", "flac", "m4a", "aac", "ogg", "opus", "wma"];

export function kindOfFile(m) {
    const ext = (m?.name || "").split(".").pop().toLowerCase();
    if (IMAGE_EXT.includes(ext)) return "image";
    if (VIDEO_EXT.includes(ext)) return "video";
    if (AUDIO_EXT.includes(ext)) return "audio";
    return null;
}

export function mediaKey(m) {
    return (m?.subfolder ? m.subfolder + "/" : "") + (m?.name || "");
}

export function mediaURL(m) {
    return api.apiURL(
        "/view?" +
            new URLSearchParams({
                filename: m.name,
                type: m.type || "input",
                subfolder: m.subfolder || "",
            }),
    );
}

// --- geometry and snapping -------------------------------------------------

export function clamp(v, lo, hi) {
    return Math.min(hi, Math.max(lo, v));
}

export function laneOf(kind) {
    return kind === "audio" ? 1 : 0;
}

// a clip's length in frames: `len` when set, 3 for legacy images that were
// placed before images were stretchable, 22 for everything else
export function clipLen(c) {
    return Number(c.len) || (c.kind === "image" ? 3 : 22);
}

// whether a clip occupies the given lane: video clips also occupy the audio
// lane through their sound band (unless the band was deleted), so audio
// clips must never overlap a linked band.
export function occupiesLane(c, lane) {
    if (lane === 1) return c.kind === "audio" || (c.kind === "video" && !c.audio_off);
    return c.kind !== "audio";
}

export function blockRect(c, s) {
    const len = clipLen(c);
    return {
        x: OFFSET_X + (Number(c.start) - 1) * s + 1,
        y: RULER_H + laneOf(c.kind) * LANE_H + 4,
        w: Math.max(2, len * s - 2),
        h: LANE_H - 8,
    };
}

// audio-lane position/length of a clip: audio clips use their own start/len,
// a video's band follows the video while linked and its own audio_* fields
// once separated
export function bandGeom(c) {
    const audio = c.kind === "audio";
    const start = audio ? c.start : c.audio_link ? c.start : c.audio_start ?? c.start;
    const len = audio ? c.len : c.audio_link ? c.len : c.audio_len ?? c.len ?? 22;
    return { start: Number(start), len: Number(len) || 22 };
}

export function ghostRect(c, s) {
    const { start, len } = bandGeom(c);
    return {
        x: OFFSET_X + (Number(start) - 1) * s + 1,
        y: RULER_H + LANE_H + 4,
        w: Math.max(2, len * s - 2),
        h: LANE_H - 8,
    };
}

// occupied [s, e) frame range of a clip inside a given lane
export function laneRange(c, lane) {
    if (lane === 0) {
        const len = clipLen(c);
        return { s: Number(c.start), e: Number(c.start) + len };
    }
    if (c.audio_off) return { s: 0, e: 0 };
    const { start, len } = bandGeom(c);
    return { s: start, e: start + len };
}

// the time range the clip's sound actually plays in (the audio-lane range:
// follows the video while linked, the ghost's own position when unlinked).
// audio_off clips have no sound at all.
export function soundRange(c) {
    if (c.audio_off) return { s: -1, e: -1 };
    const { start, len } = bandGeom(c);
    return { s: start, e: start + len };
}

// the source frame where a clip's sound content begins: a separated
// (unlinked) band froze its own slice of the file at unlink time
// (audio_src_start), so later trims to the video no longer shift the sound;
// otherwise it follows the clip's own src_start.
export function bandSrc(c) {
    if (!c) return 0;
    return Number.isFinite(Number(c.audio_src_start))
        ? Number(c.audio_src_start)
        : Number(c.src_start) || 0;
}

// magnet-snap a value to the playhead boundary or the end-line (span+1).
// Used by the trim handlers so that resizing a clip "kisses" the playhead
// or the timeline end without yanking it toward every other clip edge —
// the collision guard already keeps clips from overlapping, so the only
// free magnets the user wants are the playhead and the end-line.
export function probeSnap(node, value, scale) {
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
    if (pl != null) probe(Math.round(pl) + 1);
    if (dBest <= Math.max(0.5, SNAP_PLAY_PX / scale)) return best;
    return value;
}

// whether [s, s+len) overlaps any clip occupying `lane` (excluding `c`)
export function laneFree(node, c, lane, s, len) {
    for (const o of node._h3Clips) {
        if (o === c || !occupiesLane(o, lane)) continue;
        const r = laneRange(o, lane);
        if (s < r.e && s + len > r.s) return false;
    }
    return true;
}

// magnet-snap to the playhead boundary (and span ends) within SNAP_PLAY_PX.
// Snapping to other clips is handled by the collision push above: the clip
// is shoved flush against whichever neighbour it would overlap, so there is
// no need to probe arbitrary clip edges here — that only made the magnet
// feel like it was fighting the user at every position.
export function resolveMove(node, c, lane, s, len, grab, px) {
    const clips = node._h3Clips;
    const span = node._h3Span ?? SPAN;
    const endLine = span + 1;
    // a linked video drags its sound band along, so it must not plow
    // through audio clips either: both lanes block the whole clip.
    const lanes =
        lane === 0 && c.kind === "video" && c.audio_link && !c.audio_off
            ? [0, 1]
            : [lane];
    const free = (start) => lanes.every((L) => laneFree(node, c, L, start, len));
    // collision: push the clip flush against the neighbour it would overlap.
    // this doubles as the clip-to-clip snap: once resolved the clip sits
    // edge-to-edge with the neighbour, regardless of how fast you dragged.
    for (let guard = 0; guard < clips.length; guard++) {
        let hit = null;
        for (const L of lanes) {
            for (const o of clips) {
                if (o === c || !occupiesLane(o, L)) continue;
                const r = laneRange(o, L);
                if (s < r.e && s + len > r.s) {
                    hit = r;
                    break;
                }
            }
            if (hit) break;
        }
        if (!hit) break;
        s = grab < (hit.s + hit.e) / 2 ? hit.s - len : hit.e;
        if (s < 1) break;
    }
    s = Math.max(1, s);

    if (node._h3TimelineWidget?._snapEnabled !== false) {
        let best = s;
        let dBest = Infinity;
        const probe = (v) => {
            if (v < 1 || !free(v)) return;
            const d = Math.abs(v - s);
            if (d < dBest) {
                dBest = d;
                best = v;
            }
        };
        probe(1);
        probe(endLine);
        probe(endLine - len);
        const pl = node._h3TimelineWidget?._play;
        if (pl != null) {
            const f = Math.round(pl) + 1;
            probe(f);
            probe(f - len);
        }
        if (dBest <= Math.max(0.5, SNAP_PLAY_PX / px)) s = best;
    }
    return Math.max(1, s);
}

export function resolveMultiMove(node, targets, desiredStep, scale) {
    if (!targets || !targets.length) return 0;
    const minStart = Math.min(...targets.map((t) => t.startAt));
    const span = node._h3Span ?? SPAN;

    const obstaclesByLane = { 0: [], 1: [] };
    const targetVideoClips = new Set(targets.filter((t) => !t.audioEdit).map((t) => t.clip));
    const targetAudioTracks = new Set(
        targets.map((t) => (t.audioEdit ? `a_${t.clip.id}` : t.clip.id)),
    );

    for (const c of node._h3Clips) {
        if (occupiesLane(c, 0)) {
            if (!targetVideoClips.has(c)) {
                obstaclesByLane[0].push(laneRange(c, 0));
            }
        }
        if (occupiesLane(c, 1)) {
            const key = c.kind === "video" && !c.audio_link ? `a_${c.id}` : c.id;
            if (!targetAudioTracks.has(key) && !targetVideoClips.has(c)) {
                obstaclesByLane[1].push(laneRange(c, 1));
            }
        }
    }

    let minStep = 1 - minStart;
    let maxStep = Infinity;

    for (const t of targets) {
        const lanes = t.audioEdit
            ? [1]
            : t.clip.kind === "video" && t.clip.audio_link && !t.clip.audio_off
              ? [0, 1]
              : [laneOf(t.clip.kind)];

        const tStart = t.startAt;
        const tEnd = tStart + t.lenAt;

        for (const L of lanes) {
            for (const obs of obstaclesByLane[L]) {
                if (obs.e <= tStart) {
                    minStep = Math.max(minStep, obs.e - tStart);
                } else if (obs.s >= tEnd) {
                    maxStep = Math.min(maxStep, obs.s - tEnd);
                } else {
                    if (tStart >= obs.s) minStep = Math.max(minStep, 0);
                    if (tEnd <= obs.e) maxStep = Math.min(maxStep, 0);
                }
            }
        }
    }

    if (minStep > maxStep) return 0;

    let actualStep = clamp(desiredStep, minStep, maxStep);

    if (node._h3TimelineWidget?._snapEnabled !== false && scale > 0) {
        const snapThreshold = Math.max(0.5, SNAP_PLAY_PX / scale);
        let bestStep = actualStep;
        let dBest = Infinity;

        const probeCandidate = (candStep) => {
            if (candStep < minStep || candStep > maxStep) return;
            const diff = Math.abs(candStep - desiredStep);
            if (diff <= snapThreshold && diff < dBest) {
                dBest = diff;
                bestStep = candStep;
            }
        };

        const snapLines = [1, span + 1];
        const pl = node._h3TimelineWidget?._play;
        if (pl != null) snapLines.push(Math.round(pl) + 1);

        for (const L of [0, 1]) {
            for (const obs of obstaclesByLane[L]) {
                snapLines.push(obs.s, obs.e);
            }
        }

        for (const line of snapLines) {
            for (const t of targets) {
                probeCandidate(line - t.startAt);
                probeCandidate(line - t.lenAt - t.startAt);
            }
        }

        actualStep = bestStep;
    }

    return actualStep;
}

export function inRect(p, r, padX = 0, padY = 0) {
    const px = typeof padX === "number" ? padX : 0;
    const py = typeof padY === "number" ? padY : px;
    return (
        p[0] >= r.x - px &&
        p[0] <= r.x + r.w + px &&
        p[1] >= r.y - py &&
        p[1] <= r.y + r.h + py
    );
}

export function edgeZone(p, r) {
    // If the clip is too narrow (e.g. 1 frame on zoom out, or width < 18px),
    // do not trigger trim handles inside the body so the user can easily grab and move it.
    if (r.w < 18) {
        if (p[0] < r.x - 2) return "trimL";
        if (p[0] > r.x + r.w + 2) return "trimR";
        return null;
    }
    const trimMargin = Math.min(6, Math.max(3, Math.floor(r.w * 0.25)));
    if (p[0] < r.x + trimMargin) return "trimL";
    if (p[0] > r.x + r.w - trimMargin) return "trimR";
    return null;
}

export function btnZone(node, p) {
    if (p[1] > RULER_H) return null;
    // toolbar (buttons + slider) only spans the top strip; the ruler-number
    // band below it (drawn at RULER_H - 15) must stay ruler hits even when
    // its x overlaps a button
    if (p[1] > RULER_H - 16) return "ruler";
    const w = node._h3TimelineWidget;
    const btns = Array.isArray(w?._btns) ? w._btns : [];
    const s = w?._sliderX;
    if (btns.length && p[0] < btns[0].x - 4) return null;
    if (s != null && p[0] >= s && p[0] <= s + SLIDER_W) return "slider";
    for (const b of btns) if (p[0] >= b.x && p[0] <= b.x + b.w) return b.zone;
    return "ruler";
}

export function playHeadBoundary(node) {
    const w = node._h3TimelineWidget;
    const v = w?._play;
    if (v != null) return Math.max(1, Math.round(v) + 1);
    return Math.max(1, w?._frame ?? 1);
}

// strength envelope (flat strength per clip)
export const ENV_MIN = 0.0;
export const ENV_MAX = 1.0;

export function envFlat(c, ghost) {
    if (ghost) {
        const own = Number(c.audio_strength);
        if (Number.isFinite(own)) return clamp(own, ENV_MIN, ENV_MAX);
        return ENV_MAX;
    }
    const own = Number(c.strength);
    if (Number.isFinite(own)) return clamp(own, ENV_MIN, ENV_MAX);
    return ENV_MAX;
}

export function envY(r, v) {
    return r.y + r.h - 6 - ((clamp(v, ENV_MIN, ENV_MAX) - ENV_MIN) / (ENV_MAX - ENV_MIN)) * (r.h - 12);
}

export function envStrengthAtY(r, y) {
    return clamp(ENV_MIN + ((r.y + r.h - 6 - y) / (r.h - 12)) * (ENV_MAX - ENV_MIN), ENV_MIN, ENV_MAX);
}

export function envZone(c, p, s) {
    const rects = [];
    if (c.kind === "video") {
        if (!c.audio_off) rects.push([ghostRect(c, s), true]);
        rects.push([blockRect(c, s), false]);
    } else {
        rects.push([blockRect(c, s), false]);
    }
    for (const [r, ghost] of rects) {
        if (r.w < 16) continue;
        if (p[0] < r.x || p[0] > r.x + r.w || p[1] < r.y || p[1] > r.y + r.h) continue;
        if (ghost) {
            const bx = r.x + 8;
            const by = r.y + r.h / 2;
            if (Math.hypot(p[0] - bx, p[1] - by) < 9) continue;
        }
        if (p[0] < r.x + 5 || p[0] > r.x + r.w - 5) continue;
        if (Math.abs(p[1] - envY(r, envFlat(c, ghost))) <= 6) {
            return { zone: "envln", ghost };
        }
    }
    return null;
}

// p is in canvas space; `pan` (px) shifts the timeline content: chrome
// (buttons, slider) stays canvas-fixed, everything else compares against
// the panned content position.
export function hitTest(node, p, s, pan = 0) {
    const q = pan ? [p[0] + pan, p[1]] : p;
    if (p[1] < RULER_H) {
        const b = btnZone(node, p);
        return b ? { zone: b } : { zone: "ruler" };
    }
    for (let i = node._h3Clips.length - 1; i >= 0; i--) {
        const c = node._h3Clips[i];
        const ez = envZone(c, q, s);
        if (ez) return { i, c, ...ez };
        if (c.kind === "video" && !c.audio_off) {
            const g = ghostRect(c, s);
            const aLen = Number(c.audio_link ? c.len : c.audio_len ?? c.len ?? 22) || 1;
            const padX = g.w < 16 ? 6 : 4;
            if (inRect(q, g, padX, 4)) {
                // edge trims win over the link toggle so the ghost's
                // borders stay grabbable; the toggle keeps the middle.
                if (!c.audio_link) {
                    if (aLen === 1) {
                        if (q[0] > g.x + g.w + 1) return { i, c, zone: "trimAR" };
                        return { i, c, zone: "audio" };
                    }
                    const ez = edgeZone(q, g);
                    if (ez === "trimL") return { i, c, zone: "trimAL" };
                    if (ez === "trimR") return { i, c, zone: "trimAR" };
                }
                const bx = g.x + 8;
                const by = g.y + g.h / 2;
                if (g.w >= 20 && Math.hypot(q[0] - bx, q[1] - by) < 9) {
                    return { i, c, zone: "link" };
                }
                if (!c.audio_link) return { i, c, zone: "audio" };
                return null; // linked: the sound moves with the video
            }
        }
        const r = blockRect(c, s);
        const len = clipLen(c);
        const padX = r.w < 16 ? 6 : 2;
        if (inRect(q, r, padX, 2)) {
            if (len === 1) {
                if (q[0] > r.x + r.w + 1) return { i, c, zone: "trimR" };
                return { i, c, zone: "move" };
            }
            return { i, c, zone: edgeZone(q, r) || "move" };
        }
    }
    return null;
}

// magnet-snap a playhead boundary to clip edges and the end-line;
// p is 0-based play position, clip boundaries are 1-based frame edges
export function splitSnap(node, p, s) {
    if (node._h3TimelineWidget?._snapEnabled === false) return p;
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
    probe(span); // end-line
    if (dBest <= Math.max(0.5, SNAP_PX / s)) return best;
    return p;
}