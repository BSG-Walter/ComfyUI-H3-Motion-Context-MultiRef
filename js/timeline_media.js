// Media loading and file helpers for the H3 timeline widget.

import { api } from "../../../scripts/api.js";
import { kindOfFile, mediaKey, mediaURL } from "./timeline_core.js";

export async function uploadMedia(file) {
    const fd = new FormData();
    fd.append("image", file);
    fd.append("type", "input");
    fd.append("overwrite", "true");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body: fd });
    if (!resp.ok) throw new Error("upload failed: " + resp.status);
    const info = await resp.json();
    return { name: info.name, subfolder: info.subfolder || "", type: info.type || "input" };
}

export function pickFile() {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*,video/*,audio/*";
    return new Promise((resolve) => {
        input.onchange = () => resolve(input.files?.[0] || null);
        input.click();
    });
}

export function computePeaks(buf) {
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

export function redrawNode(node) {
    const w = node._h3TimelineWidget;
    if (w) w.redraw?.(node);
}

export function ensureMedia(node, c) {
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

function decodeAudio(node, m) {
    return fetch(m.url)
        .then((r) => r.arrayBuffer())
        .then((buf) => {
            const actx =
                node._h3AudioCtx ?? (node._h3AudioCtx = new AudioContext());
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
        return decodeAudio(node, m);
    }
    if (m.kind === "video") {
        // thumbnails come from the per-clip <video> elements (seeked on
        // demand in paint). The audio track is decoded into a WebAudio
        // buffer so it can be played back via startSound, the same path
        // used by audio clips — this sidesteps the browser autoplay-with-
        // sound restriction that blocks .play() on a detached <video> once
        // the user gesture has expired.
        decodeAudio(node, m);
        return Promise.resolve();
    }
    return Promise.resolve();
}

// per-clip <video> elements render each clip's own thumbnail frame and,
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
            m.duration = t.el.duration;
            try {
                t.el.currentTime = t.t;
            } catch (_) {}
            redrawNode(node);
        };
        t.el.onloadedmetadata = () => {
            m.duration = t.el.duration;
        };
        t.el.onended = () => {
            t.el.currentTime = 0;
        };
    }
    return t;
}

// source length in frames for any file-backed clip (video metadata or the
// decoded audio buffer), Infinity while unknown so loading is non-blocking.
export function sourceFrames(node, c) {
    const saved = Number(c?.source_len);
    if (saved > 0 && isFinite(saved)) return saved;
    const m = ensureMedia(node, c);
    if (!m) return Infinity;
    const dur =
        (typeof m.duration === "number" && isFinite(m.duration) ? m.duration : m.buffer?.duration) ||
        0;
    if (dur <= 0) return Infinity;
    return Math.max(1, Math.floor(dur * (node._h3TimelineWidget?._fps || 24)));
}

export function thumbSeek(node, c, m, target) {
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

// read the duration of an uploaded video so new clips default to the full
// source length. Resolves to null if the duration can't be probed quickly.
export function probeVideoFrames(node, info) {
    return new Promise((resolve) => {
        const url = mediaURL(info);
        const v = document.createElement("video");
        v.preload = "metadata";
        v.muted = true;
        let done = false;
        const finish = (val) => {
            if (done) return;
            done = true;
            v.removeAttribute("src");
            v.load();
            resolve(val);
        };
        v.onloadedmetadata = () => {
            const fps = node._h3TimelineWidget?._fps || 24;
            const frames = Math.max(1, Math.floor(v.duration * fps));
            finish(frames);
        };
        v.onerror = () => finish(null);
        setTimeout(() => finish(null), 4000);
        v.src = url;
    });
}