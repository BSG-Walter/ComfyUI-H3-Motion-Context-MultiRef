// Media loading and file helpers for the H3 timeline widget.

import { api } from "../../../scripts/api.js";
import { kindOfFile, mediaKey, mediaURL } from "./timeline_core.js";

export async function uploadMedia(file) {
    const fd = new FormData();
    fd.append("image", file);
    fd.append("type", "input");
    fd.append("overwrite", "true");
    const resp = await api.fetchApi("/upload/image", { method: "POST", body: fd });
    if (!resp.ok) {
        if (resp.status === 413) {
            const sizeMB = file?.size ? (file.size / (1024 * 1024)).toFixed(1) : "?";
            throw new Error(
                `File "${file.name}" (${sizeMB} MB) exceeds ComfyUI's upload limit (default 100 MB, HTTP 413).\n\n` +
                `Solutions:\n` +
                `• Compress or trim the video to under 100 MB.\n` +
                `• Or start ComfyUI with the argument: --max-upload-size 1000 (allows up to 1 GB).`
            );
        }
        let detail = "";
        try {
            const txt = await resp.text();
            if (txt) detail = `: ${txt}`;
        } catch (_) {}
        throw new Error(`Upload failed (${resp.status})${detail}`);
    }
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
            m.loaded = true;
            redrawNode(node);
        })
        .catch(() => {});
}

export async function decodeGifFrames(url) {
    if (typeof ImageDecoder !== "undefined") {
        try {
            const resp = await fetch(url);
            const buf = await resp.arrayBuffer();
            const decoder = new ImageDecoder({ data: buf, type: "image/gif" });
            await decoder.tracks.ready;
            const track = decoder.tracks.selectedTrack;
            const count = track.frameCount;
            const frames = [];
            let totalDur = 0;
            for (let i = 0; i < count; i++) {
                const res = await decoder.decode({ frameIndex: i });
                const dur = (res.image.duration || 100000) / 1000000;
                const bmp = await createImageBitmap(res.image);
                frames.push({ img: bmp, timestamp: totalDur, duration: dur });
                totalDur += dur;
                res.image.close();
            }
            return { frames, duration: totalDur || frames.length * 0.1, count };
        } catch (e) {
            console.warn("h3 timeline: ImageDecoder failed for gif", e);
        }
    }
    return null;
}

export function gifFrameAtTime(m, target) {
    if (!m?.frames?.length) return null;
    if (m.frames.length === 1) return m.frames[0].img;
    const total = m.duration || m.frames.length * 0.1;
    if (total <= 0) return m.frames[0].img;
    const t = Math.max(0, target) % total;
    let acc = 0;
    for (const f of m.frames) {
        acc += f.duration;
        if (t <= acc) return f.img;
    }
    return m.frames[m.frames.length - 1].img;
}

export function getClipGifFrame(node, clip, media) {
    if (!media?.isGif || !media?.frames?.length) return null;
    const w = node?._h3TimelineWidget;
    const fps = w?._fps || 24;
    const play = w?._play;
    const frame = w?._frame;
    const f = play != null ? play : frame != null ? frame - 1 : null;
    let target = (Number(clip.src_start) || 0) / fps;
    if (f != null) {
        const s = Number(clip.start) - 1;
        const len = Number(clip.len) || 22;
        if (f >= s && f < s + len) {
            target = (f - s + (Number(clip.src_start) || 0)) / fps;
        }
    }
    return gifFrameAtTime(media, Math.max(0, target));
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
        if ((m.url || "").toLowerCase().includes(".gif") || (_c?.file?.name || "").toLowerCase().endsWith(".gif")) {
            m.isGif = true;
            return decodeGifFrames(m.url).then((res) => {
                if (res) {
                    m.frames = res.frames;
                    m.duration = res.duration;
                    m.loaded = true;
                    redrawNode(node);
                }
            });
        }
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

// source length in frames for any file-backed clip: `source_end` (set by
// split: an absolute source frame the clip's content may not pass, turning
// each half into its own file) first, then video metadata or the decoded
// audio buffer, Infinity while unknown so loading is non-blocking.
export function sourceFrames(node, c) {
    const end = Number(c?.source_end);
    if (end > 0 && isFinite(end)) return end;
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
    if (m?.isGif) {
        let t = node._h3Thumbs.get(c.id);
        if (!t) {
            t = {
                isGif: true,
                t: -1,
                currentFrame: null,
            };
            node._h3Thumbs.set(c.id, t);
        }
        t.isGif = true;
        t.t = target;
        t.currentFrame = gifFrameAtTime(m, target);
        return t;
    }
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
export async function probeVideoFrames(node, info) {
    const url = mediaURL(info);
    if ((info?.name || "").toLowerCase().endsWith(".gif")) {
        const data = await decodeGifFrames(url);
        if (data) {
            const fps = node._h3TimelineWidget?._fps || 24;
            return Math.max(1, Math.round(data.duration * fps));
        }
    }
    return new Promise((resolve) => {
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