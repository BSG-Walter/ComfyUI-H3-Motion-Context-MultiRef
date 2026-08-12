// Playback (playhead, preview, sound) for the H3 timeline widget.

import { mediaKey, soundRange, SPAN } from "./timeline_core.js";
import { ensureMedia, redrawNode } from "./timeline_media.js";

export function togglePlay(node) {
    const w = node._h3TimelineWidget;
    if (!w) return;
    w._playing = !w._playing;
    if (w._playing) {
        if (!node._h3AudioCtx) {
            node._h3AudioCtx = new AudioContext();
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
                stopPlay(node);
            }
            redrawNode(node);
            requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
    } else {
        stopPlay(node);
    }
    redrawNode(node);
}

export function stopPlay(node) {
    stopSound(node);
    for (const [, t] of (node._h3Thumbs ?? []).entries()) {
        try {
            t.el.pause();
            t.el.muted = true;
        } catch (_) {}
    }
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

export function previewClip(node) {
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

// clip whose SOUND is under the playhead (audio-lane position: a video's
// ghost when unlinked, its own block otherwise)
function soundClip(node) {
    const w = node._h3TimelineWidget;
    const f = w?._play;
    if (f == null) return null;
    return (
        node._h3Clips?.find((c) => {
            if (c.kind === "image" || !c.file) return false;
            const r = soundRange(c);
            return f >= r.s - 1 && f < r.e - 1;
        }) || null
    );
}

export function syncPreview(node) {
    const w = node._h3TimelineWidget;
    const vc = previewClip(node);
    const sc = soundClip(node);
    const play = w?._play ?? 0;
    const playing = !!w?._playing;
    const fps = w?._fps || 24;

    for (const [id, t] of (node._h3Thumbs ?? []).entries()) {
        const active =
            playing &&
            vc?.kind === "video" &&
            vc.id === id &&
            vc.file;
        try {
            if (!active) {
                t.el.pause();
                t.el.muted = true;
                continue;
            }
            // the active clip's element PLAYS muted (visuals) while the
            // decoded WebAudio buffer provides the sound (startSound), so
            // the picture advances smoothly instead of seeking every tick
            // (which flickered). Drift is corrected by snapping back when
            // the element falls more than a quarter second behind.
            const m = ensureMedia(node, vc);
            if (m?.kind !== "video") continue;
            const off = (play - (Number(vc.start) - 1) + (Number(vc.src_start) || 0)) / fps;
            if (t.el.paused) {
                t.el.muted = true;
                t.el.currentTime = Math.max(0, off);
                t.el.play().catch(() => {});
            } else if (Math.abs(t.el.currentTime - off) > 0.25) {
                t.el.currentTime = Math.max(0, off);
            }
        } catch (_) {}
    }

    // Audio is played via WebAudio using the decoded buffer, which unlike
    // .play() on a detached <video> with sound is not blocked by autoplay
    // policy once the AudioContext was resumed by the click.
    if (playing && sc) startSound(node, sc);
    else stopSound(node);
}

function startSound(node, c) {
    const key = mediaKey(c.file);
    if (node._h3Sound && node._h3SoundClip !== key) stopSound(node);
    if (node._h3Sound) return; // already playing this track
    const m = ensureMedia(node, c);
    if (!m?.buffer || !node._h3AudioCtx) return;
    const play = node._h3TimelineWidget?._play ?? 0;
    const fps = node._h3TimelineWidget?._fps || 24;
    const r = soundRange(c);
    const off = (play - (r.s - 1)) / fps + (Number(c.src_start) || 0) / fps;
    try {
        const src = node._h3AudioCtx.createBufferSource();
        src.buffer = m.buffer;
        src.connect(node._h3AudioCtx.destination);
        src.start(0, Math.max(0, off));
        node._h3Sound = src;
        node._h3SoundClip = key;
    } catch (_) {}
}