// Timeline state persistence and clip operations for the H3 timeline widget.

import { app } from "../../scripts/app.js";
import {
    STATE_NAME,
    WIDTH,
    HEIGHT,
    MAX_CLIPS,
    defaults,
    kindOfFile,
    laneOf,
    playHeadBoundary,
} from "./timeline_core.js";
import { pickFile, probeVideoFrames, redrawNode, uploadMedia } from "./timeline_media.js";

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

export function hideStateWidget(node) {
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
    const [clip] = node._h3Clips.splice(i, 1);
    // a separated (unlinked) audio band outlives its video: promote the band
    // to its own audio clip before deleting the video block, so deleting the
    // video never kills the detached audio.
    if (clip.kind === "video" && !clip.audio_link && !clip.audio_off) {
        const a = defaults.audio();
        a.id = (node._h3Clips.at(-1)?.id ?? 0) + 1;
        a.start = Number(clip.audio_start ?? clip.start);
        a.len = Number(clip.audio_len ?? clip.len) || 22;
        a.strength = Number(clip.strength) || 1;
        a.align = clip.audio_align ?? "head";
        if (clip.file) {
            a.file = clip.file;
            a.src_start = Number(clip.src_start) || 0;
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

// delete only the separated audio band of a video clip: the band goes
// silent and stops being drawn/colliding, the video block stays.
export function removeClipAudio(node, c) {
    if (c?.kind !== "video") return;
    c.audio_off = true;
    writeState(node);
    fixNodeSize(node);
}

export async function addClipWithMedia(node, kind) {
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
    if (useKind === "video") {
        len = await probeVideoFrames(node, info, file);
    }
    addClip(node, useKind, info, len);
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

export function addClip(node, kind, info, lenOverride) {
    if (!node._h3Clips || node._h3Clips.length >= MAX_CLIPS) return;
    const c = defaults[kind]();
    c.id = (node._h3Clips.at(-1)?.id ?? 0) + 1;
    if (info) {
        c.file = info;
        c.src_start = 0;
    }
    if (lenOverride && Number.isFinite(lenOverride) && lenOverride > 0) {
        c.source_len = lenOverride;
        c.len = lenOverride;
    }
    node._h3Clips.push(c);
    placeAndPushClip(node, c);
    ensureInputs(node);
    writeState(node);
    fixNodeSize(node);
}

export async function replaceClipMedia(node, c) {
    const file = await pickFile();
    if (!file) return;
    const info = await uploadMedia(file);
    const kind = kindOfFile(info);
    if (kind) c.kind = kind;
    c.file = info;
    c.src_start = 0;
    delete c.source_len;
    if (c.kind === "video") {
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