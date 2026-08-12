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
export const BTN_H = 12;
export const RULER_H = 18;
export const LANE_H = 68;
export const PAD = 4;
export const WIDTH = 840;
export const HEIGHT = RULER_H + 2 * LANE_H + PAD;
export const TOOL_X = WIDTH - 230; // first toolbar column
export const SLIDER_X = TOOL_X + 6 * 20 + 6; // zoom slider start (6px gap after last button)
export const SLIDER_W = 74; // zoom slider track width

export const COLORS = {
    image: "#7aa2f7",
    video: "#9ece6a",
    audio: "#f7768e",
};
export const PLAY_COLOR = "#ff5252";

export const defaults = {
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

// whether a clip occupies the given lane: video clips also occupy the audio
// lane through their sound band (unless the band was deleted), so audio
// clips must never overlap a linked band.
export function occupiesLane(c, lane) {
    if (lane === 1) return c.kind === "audio" || (c.kind === "video" && !c.audio_off);
    return c.kind !== "audio";
}

export function blockRect(c, s) {
    const len = c.kind === "image" ? 3 : Number(c.len) || 22;
    return {
        x: OFFSET_X + (Number(c.start) - 1) * s + 2,
        y: RULER_H + laneOf(c.kind) * LANE_H + 4,
        w: Math.max(2, len * s - 4),
        h: LANE_H - 8,
    };
}

export function ghostRect(c, s) {
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
export function laneRange(c, lane) {
    if (lane === 0) {
        const len = c.kind === "image" ? 3 : Number(c.len) || 22;
        return { s: Number(c.start), e: Number(c.start) + len };
    }
    if (c.audio_off) return { s: 0, e: 0 };
    const s =
        c.kind === "audio"
            ? Number(c.start)
            : c.audio_link
              ? Number(c.start)
              : Number(c.audio_start ?? c.start);
    const len =
        c.kind === "audio"
            ? Number(c.len) || 22
            : c.audio_link
              ? Number(c.len) || 22
              : Number(c.audio_len ?? c.len ?? 22);
    return { s, e: s + len };
}

// the time range the clip's sound actually plays in (the audio-lane range:
// follows the video while linked, the ghost's own position when unlinked).
// audio_off clips have no sound at all.
export function soundRange(c) {
    if (c.audio_off) return { s: -1, e: -1 };
    const start = c.audio_link ? c.start : c.audio_start ?? c.start;
    const len = c.audio_link ? c.len : c.audio_len ?? c.len ?? 22;
    return { s: Number(start), e: Number(start) + (Number(len) || 22) };
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

export function inRect(p, r, pad = 0) {
    return (
        p[0] >= r.x - pad &&
        p[0] <= r.x + r.w + pad &&
        p[1] >= r.y - pad &&
        p[1] <= r.y + r.h + pad
    );
}

export function edgeZone(p, r) {
    if (p[0] < r.x + 5) return "trimL";
    if (p[0] > r.x + r.w - 5) return "trimR";
    return null;
}

export function btnZone(p) {
    if (p[1] > RULER_H || p[0] < TOOL_X) return null;
    if (p[0] >= SLIDER_X && p[0] <= SLIDER_X + SLIDER_W) return "slider";
    const col = Math.floor((p[0] - TOOL_X) / 20);
    return ["split", "snap", "play", "unit", "in", "out"][col] || null;
}

export function playHeadBoundary(node) {
    const w = node._h3TimelineWidget;
    const v = w?._play;
    if (v != null) return Math.max(1, Math.round(v) + 1);
    return Math.max(1, w?._frame ?? 1);
}

export function hitTest(node, p, s) {
    if (p[1] < RULER_H) {
        const b = btnZone(p);
        return b ? { zone: b } : { zone: "ruler" };
    }
    for (let i = node._h3Clips.length - 1; i >= 0; i--) {
        const c = node._h3Clips[i];
        if (c.kind === "video" && !c.audio_off) {
            const g = ghostRect(c, s);
            if (inRect(p, g, 4)) {
                // edge trims win over the link toggle so the ghost's
                // borders stay grabbable; the toggle keeps the middle.
                if (!c.audio_link) {
                    const ez = edgeZone(p, g);
                    if (ez === "trimL") return { i, c, zone: "trimAL" };
                    if (ez === "trimR") return { i, c, zone: "trimAR" };
                }
                const bx = g.x + 8;
                const by = g.y + g.h / 2;
                if (Math.hypot(p[0] - bx, p[1] - by) < 9) {
                    return { i, c, zone: "link" };
                }
                if (!c.audio_link) return { i, c, zone: "audio" };
                return null; // linked: the sound moves with the video
            }
        }
        const r = blockRect(c, s);
        if (inRect(p, r, 2)) {
            if (c.kind === "image") return { i, c, zone: "move" };
            return { i, c, zone: edgeZone(p, r) || "move" };
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