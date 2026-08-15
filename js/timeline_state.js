// Timeline state persistence and clip operations for the H3 timeline widget.

import { app } from "../../scripts/app.js";
import {
    STATE_NAME,
    WIDTH,
    HEIGHT,
    MAX_CLIPS,
    SPAN,
    bandSrc,
    defaults,
    kindOfFile,
    clipLen,
    laneOf,
    laneRange,
    occupiesLane,
    playHeadBoundary,
    cloneEnv,
} from "./timeline_core.js";
import { pickFile, probeVideoFrames, redrawNode, uploadMedia } from "./timeline_media.js";
import { togglePlay } from "./timeline_play.js";

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

export function stateWidget(node) {
    return node.widgets?.find((w) => w.name === STATE_NAME);
}

export function readState(node) {
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

export function writeState(node) {
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

const MAX_HISTORY = 50;

function snapshotState(node) {
    return JSON.stringify({
        clips: node._h3Clips || [],
        unit: node._h3TimelineWidget?._unit ?? "f",
    });
}

export function recordHistory(node) {
    if (!node) return;
    node._h3Undo ??= [];
    node._h3Redo ??= [];
    const snap = snapshotState(node);
    if (node._h3Undo.length && node._h3Undo[node._h3Undo.length - 1] === snap) return;
    node._h3Undo.push(snap);
    if (node._h3Undo.length > MAX_HISTORY) node._h3Undo.shift();
    node._h3Redo.length = 0;
}

export function undo(node) {
    if (!node) return false;
    node._h3Undo ??= [];
    node._h3Redo ??= [];
    if (!node._h3Undo.length) return false;
    const currentSnap = snapshotState(node);
    let targetSnap = node._h3Undo.pop();
    if (targetSnap === currentSnap && node._h3Undo.length) {
        targetSnap = node._h3Undo.pop();
    }
    if (targetSnap === currentSnap) return false;
    node._h3Redo.push(currentSnap);
    restoreSnapshot(node, targetSnap);
    return true;
}

export function redo(node) {
    if (!node) return false;
    node._h3Undo ??= [];
    node._h3Redo ??= [];
    if (!node._h3Redo.length) return false;
    const currentSnap = snapshotState(node);
    const nextSnap = node._h3Redo.pop();
    node._h3Undo.push(currentSnap);
    restoreSnapshot(node, nextSnap);
    return true;
}

function restoreSnapshot(node, snap) {
    try {
        const parsed = JSON.parse(snap);
        node._h3Clips = Array.isArray(parsed.clips) ? parsed.clips : [];
        if (node._h3TimelineWidget && parsed.unit) {
            node._h3TimelineWidget._unit = parsed.unit;
        }
        node._h3Selected?.clear();
        ensureInputs(node);
        writeState(node);
        fixNodeSize(node);
        node._h3TimelineWidget?.redraw?.(node);
    } catch (_) {}
}

export function hideStateWidget(node) {
    const widget = stateWidget(node);
    if (!widget || widget._h3Hidden) return;
    widget._h3Hidden = true;
    widget.hidden = true;
    widget.options ??= {};
    widget.options.hidden = true;
    widget.computeSize = () => [0, -4];
}

// wipe every clip off the timeline (thumbs too; the decoded media cache is
// kept — the same files are likely re-imported soon)
export function clearAll(node) {
    if (!node._h3Clips?.length) return;
    if (typeof window !== "undefined" && !window.confirm("Clear all timeline clips?")) return;
    recordHistory(node);
    if (node._h3TimelineWidget?._playing) togglePlay(node);
    node._h3Clips.length = 0;
    node._h3Thumbs?.clear();
    node._h3Selected?.clear();
    ensureInputs(node);
    writeState(node);
    fixNodeSize(node);
}

// download the timeline (clips + unit) as a JSON file
export function exportState(node) {
    const data = JSON.stringify(
        {
            clips: node._h3Clips ?? [],
            unit: node._h3TimelineWidget?._unit ?? "f",
        },
        null,
        2,
    );
    const blob = new Blob([data], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "h3_timeline.json";
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
}

// load a timeline JSON produced by exportState (or any {clips:[...]} shape)
export function importState(node) {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "application/json,.json";
    input.onchange = () => {
        const file = input.files?.[0];
        if (!file) return;
        const reader = new FileReader();
        reader.onload = () => {
            try {
                const parsed = JSON.parse(String(reader.result));
                if (!Array.isArray(parsed?.clips)) {
                    throw new Error("no clips array");
                }
                recordHistory(node);
                node._h3Clips = parsed.clips.filter(
                    (c) => c && c.kind && Number.isFinite(Number(c.start)),
                );
                if (node._h3TimelineWidget) {
                    node._h3TimelineWidget._unit =
                        parsed.unit === "s" ? "s" : "f";
                }
                node._h3Thumbs?.clear();
                node._h3Selected?.clear();
                ensureInputs(node);
                writeState(node);
                fixNodeSize(node);
            } catch (err) {
                window.alert?.(`Import failed: ${err.message}`);
            }
        };
        reader.readAsText(file);
    };
    input.click();
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

export function fixNodeSize(node) {
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
    const sb = node._h3ScrollWidget;
    if (sb) {
        const off = widgetYOffset(node, w?.y ?? 0) + HEIGHT;
        sb.y = off;
        sb._rowOf = off;
    }
}

export function ensureInputs(node) {
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

export function removeClip(node, i) {
    recordHistory(node);
    const [clip] = node._h3Clips.splice(i, 1);
    if (clip) {
        node._h3Selected?.delete(clip.id);
        node._h3Selected?.delete(`a_${clip.id}`);
    }
    // a separated (unlinked) audio band outlives its video: promote the band
    // to its own audio clip before deleting the video block, so deleting the
    // video never kills the detached audio.
    if (clip.kind === "video" && !clip.audio_link && !clip.audio_off) {
        const a = defaults.audio();
        a.id = (node._h3Clips.at(-1)?.id ?? 0) + 1;
        a.start = Number(clip.audio_start ?? clip.start);
        a.len = Number(clip.audio_len ?? clip.len) || 22;
        a.strength = clip.audio_strength ?? clip.strength ?? 1;
        a.align = clip.audio_align ?? "head";
        a.env = cloneEnv(clip.audio_env ?? clip.env);
        if (clip.file) {
            a.file = clip.file;
            a.src_start = bandSrc(clip);
        }
        node._h3Clips.push(a);
    }
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

export function removeSelectedClips(node) {
    if (!node._h3Selected?.size || !node._h3Clips?.length) return;
    recordHistory(node);
    let changed = false;
    for (const key of Array.from(node._h3Selected)) {
        if (typeof key === "string" && key.startsWith("a_")) {
            const id = Number(key.slice(2));
            const clip = node._h3Clips.find((c) => c.id === id);
            if (clip && clip.kind === "video" && !clip.audio_link) {
                clip.audio_off = true;
                changed = true;
            }
        }
    }
    const toRemove = node._h3Clips.filter((c) => node._h3Selected.has(c.id));
    if (toRemove.length) {
        changed = true;
        for (const clip of toRemove) {
            const idx = node._h3Clips.indexOf(clip);
            if (idx >= 0) node._h3Clips.splice(idx, 1);
            node._h3Thumbs?.delete(clip.id);
            for (const [name] of clipInputs(clip)) {
                const slot = node.inputs?.findIndex((inp) => inp.name === name);
                if (slot >= 0) {
                    if (node.inputs[slot].link != null) node.disconnectInput(slot);
                    node.removeInput(slot);
                }
            }
        }
    }
    node._h3Selected.clear();
    if (changed) {
        ensureInputs(node);
        writeState(node);
        fixNodeSize(node);
    }
}

let _clipboard = [];

export function copySelectedClips(node, targetClip = null) {
    let clipsToCopy = [];
    if (node._h3Selected?.size) {
        clipsToCopy = node._h3Clips.filter((c) => node._h3Selected.has(c.id) || node._h3Selected.has(`a_${c.id}`));
    }
    if (!clipsToCopy.length && targetClip) {
        clipsToCopy = [targetClip];
    }
    if (!clipsToCopy.length) return false;
    _clipboard = clipsToCopy.map((c) => ({
        ...JSON.parse(JSON.stringify(c)),
        env: cloneEnv(c.env),
        audio_env: cloneEnv(c.audio_env),
    }));
    return true;
}

export function hasClipboard() {
    return _clipboard.length > 0;
}

function isClipsRangeFree(existingClips, clipsToPlace, baseStart) {
    for (const c of clipsToPlace) {
        const cStart = baseStart + c.offset;
        const cLen = c.len;
        const lanes =
            c.kind === "video" && c.audio_link && !c.audio_off
                ? [0, 1]
                : [laneOf(c.kind)];
        for (const L of lanes) {
            for (const o of existingClips) {
                if (!occupiesLane(o, L)) continue;
                const r = laneRange(o, L);
                if (cStart < r.e && cStart + cLen > r.s) {
                    return { free: false, nextBase: r.e - c.offset };
                }
            }
        }
        if (c.kind === "video" && !c.audio_link && !c.audio_off && c.audio_len != null) {
            const aStart = baseStart + c.audio_offset;
            const aLen = c.audio_len;
            for (const o of existingClips) {
                if (!occupiesLane(o, 1)) continue;
                const r = laneRange(o, 1);
                if (aStart < r.e && aStart + aLen > r.s) {
                    return { free: false, nextBase: r.e - c.audio_offset };
                }
            }
        }
    }
    return { free: true, base: baseStart };
}

function findNextFreeBase(existingClips, clipsToPlace, startFrom) {
    let base = Math.max(1, startFrom);
    for (let guard = 0; guard < 2000; guard++) {
        const res = isClipsRangeFree(existingClips, clipsToPlace, base);
        if (res.free) return base;
        base = Math.max(base + 1, res.nextBase);
    }
    return base;
}

export function pasteClips(node, targetFrame = null) {
    if (!_clipboard.length) return false;
    recordHistory(node);
    const minOrig = Math.min(..._clipboard.map((c) => Number(c.start) || 1));
    const startFrom = targetFrame != null ? targetFrame : playHeadBoundary(node);

    const clipsToPlace = _clipboard.map((c) => {
        const offset = (Number(c.start) || 1) - minOrig;
        const len = clipLen(c);
        const aOffset = (Number(c.audio_start) || Number(c.start) || 1) - minOrig;
        const aLen = Number(c.audio_len ?? c.len) || 22;
        return {
            kind: c.kind,
            audio_link: c.audio_link,
            audio_off: c.audio_off,
            offset,
            len,
            audio_offset: aOffset,
            audio_len: aLen,
            raw: c,
        };
    });

    const baseF = findNextFreeBase(node._h3Clips, clipsToPlace, startFrom);
    let nextId = (node._h3Clips.reduce((max, c) => Math.max(max, Number(c.id) || 0), 0) || 0) + 1;
    const newSelected = new Set();

    for (const item of clipsToPlace) {
        const c = item.raw;
        const newStart = baseF + item.offset;
        const cloned = {
            ...JSON.parse(JSON.stringify(c)),
            id: nextId++,
            start: newStart,
            env: cloneEnv(c.env),
            audio_env: cloneEnv(c.audio_env),
        };
        if (cloned.audio_start != null) {
            cloned.audio_start = baseF + item.audio_offset;
        }
        node._h3Clips.push(cloned);
        newSelected.add(cloned.id);
    }
    node._h3Selected = newSelected;
    ensureInputs(node);
    writeState(node);
    fixNodeSize(node);
    return true;
}

// delete only the separated audio band of a video clip: the band goes
// silent and stops being drawn/colliding, the video block stays.
export function removeClipAudio(node, c) {
    if (c?.kind !== "video") return;
    recordHistory(node);
    c.audio_off = true;
    writeState(node);
    fixNodeSize(node);
}

export async function addClipWithMedia(node, kind, startFrom = null) {
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
    const useKind = detected || kind;
    let len = null;
    if (useKind === "video" || useKind === "audio") {
        len = await probeVideoFrames(node, info, file);
    }
    addClip(node, useKind, info, len, startFrom);
}

function placeNewClip(node, newClip, startFrom = null) {
    const existingClips = node._h3Clips.filter((c) => c !== newClip);
    const targetStart = startFrom != null ? Math.max(1, startFrom) : playHeadBoundary(node);
    const newLen = clipLen(newClip);
    const span = node._h3Span ?? SPAN;
    const clipSpec = {
        kind: newClip.kind,
        audio_link: newClip.audio_link,
        audio_off: newClip.audio_off,
        offset: 0,
        len: newLen,
        audio_offset: 0,
        audio_len: newClip.audio_len ?? newLen,
    };

    // 1. Try targetStart if it fits in span and is free
    const atTarget = isClipsRangeFree(existingClips, [clipSpec], targetStart);
    if (atTarget.free && targetStart + newLen <= span + 1) {
        newClip.start = targetStart;
        return;
    }

    // 2. Try after targetStart
    const nextAfter = findNextFreeBase(existingClips, [clipSpec], targetStart);
    if (nextAfter + newLen <= span + 1) {
        newClip.start = nextAfter;
        return;
    }

    // 3. Try from frame 1
    const firstGap = findNextFreeBase(existingClips, [clipSpec], 1);
    if (firstGap + newLen <= span + 1) {
        newClip.start = firstGap;
        return;
    }

    // 4. If it doesn't fit in span at all, place at next available slot after last clip
    newClip.start = nextAfter;
}

export function addClip(node, kind, info, lenOverride, startFrom = null) {
    if (!node._h3Clips || node._h3Clips.length >= MAX_CLIPS) return;
    recordHistory(node);
    const c = defaults[kind]();
    c.id = (node._h3Clips.at(-1)?.id ?? 0) + 1;
    if (info) {
        c.file = info;
        c.src_start = 0;
        if ((info.name || "").toLowerCase().endsWith(".gif")) {
            c.audio_off = true;
        }
    }
    if (lenOverride && Number.isFinite(lenOverride) && lenOverride > 0) {
        c.source_len = lenOverride;
        c.len = lenOverride;
    }
    node._h3Clips.push(c);
    placeNewClip(node, c, startFrom);
    ensureInputs(node);
    writeState(node);
    fixNodeSize(node);
}

export async function replaceClipMedia(node, c) {
    const file = await pickFile();
    if (!file) return;
    const info = await uploadMedia(file);
    recordHistory(node);
    const kind = kindOfFile(info);
    if (kind) c.kind = kind;
    c.file = info;
    c.src_start = 0;
    delete c.source_len;
    delete c.source_end;
    delete c.src_floor;
    if ((info.name || "").toLowerCase().endsWith(".gif")) {
        c.audio_off = true;
    }
    if (c.kind === "video" || c.kind === "audio") {
        const frames = await probeVideoFrames(node, info);
        if (frames && Number.isFinite(frames) && frames > 0) {
            c.source_len = frames;
            c.len = Math.min(Number(c.len) || frames, frames);
        }
    }
    ensureInputs(node);
    writeState(node);
    fixNodeSize(node);
}

// envelope points of the right half of a split: content shifts by `cut`
// frames, so points before the cut stay with the left half and the rest
// move with the right half
function spliceEnv(env, cut) {
    if (!Array.isArray(env)) return env;
    const out = [];
    for (const p of env) {
        const f = Number(p[0]);
        if (f >= cut) out.push([f - cut, Number(p[1])]);
    }
    return out;
}

export function splitAt(node) {
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
    recordHistory(node);
    const c = clips[i];
    const cut = f - Number(c.start);
    const src = Number(c.src_start) || 0;
    // each half becomes a brand-new file of its own window: source_end caps
    // growth at the original window's end so a half can never reclaim the
    // part the split cut off.
    const left = {
        ...c,
        len: cut,
        env: cloneEnv(c.env),
        audio_env: cloneEnv(c.audio_env),
        audio_len: Math.min(Number(c.audio_len ?? c.len ?? 22) || 1, cut),
        source_end: src + cut,
        src_floor: src,
    };
    const right = {
        ...c,
        start: f,
        len: (Number(c.len) || 1) - cut,
        src_start: src + cut,
        src_floor: src + cut,
        id: (clips.at(-1)?.id ?? 0) + 1,
        env: spliceEnv(c.env, cut),
        audio_env: spliceEnv(c.audio_env, cut),
        audio_start: (Number(c.audio_start ?? c.start) || 1) + cut,
        audio_len: Math.max(1, (Number(c.audio_len ?? c.len ?? 22) || 1) - cut),
        audio_src_start: (Number(c.audio_src_start ?? c.src_start) || 0) + cut,
        source_end: src + (Number(c.len) || 1),
    };
    clips.splice(i, 1, left, right);
    writeState(node);
    fixNodeSize(node);
}