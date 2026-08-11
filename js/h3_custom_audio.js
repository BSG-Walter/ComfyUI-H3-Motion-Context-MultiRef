import { app } from "../../scripts/app.js";

const NODE_NAME = "MiniMaxH3CustomAudio";
const DEFAULT_POSITIONS = [1];
const MIN_AUDIOS = 1;
const MAX_AUDIOS = 16;

function stateWidget(node) {
    return node.widgets?.find((w) => w.name === "audio_state");
}

function readState(node) {
    const raw = stateWidget(node);
    let state = {
        count: 1,
        positions: [...DEFAULT_POSITIONS],
        strengths: [1],
    };

    try {
        const parsed = JSON.parse(raw?.value || "");
        if (Number.isInteger(parsed?.count)) {
            state.count = Math.min(
                MAX_AUDIOS,
                Math.max(MIN_AUDIOS, parsed.count),
            );
        }
        if (Array.isArray(parsed?.positions)) {
            state.positions = parsed.positions.map(
                (v) => Math.trunc(Number(v)),
            );
        }
        if (Array.isArray(parsed?.strengths)) {
            state.strengths = parsed.strengths.map((v) => {
                const s = Number(v);
                if (!Number.isFinite(s)) return 1;
                return Math.min(1, Math.max(0.05, s));
            });
        }
    } catch (_) {}

    while (state.positions.length < state.count) {
        const previous = state.positions.at(-1) ?? 1;
        state.positions.push(previous + 17);
    }
    while (state.strengths.length < state.count) {
        state.strengths.push(1);
    }

    state.positions = state.positions.slice(0, state.count);
    state.strengths = state.strengths.slice(0, state.count);
    return state;
}

function hideStateWidget(node) {
    const widget = stateWidget(node);
    if (!widget || widget._h3CustomHidden) return;

    widget._h3CustomHidden = true;
    widget.computeSize = () => [0, -4];
}

function audioInputName(i) {
    return `audio_${i}`;
}

function positionWidgetName(i) {
    return `audio ${i} position`;
}

function strengthWidgetName(i) {
    return `audio ${i} strength`;
}

function findInput(node, name) {
    return node.inputs?.findIndex(
        (input) => input.name === name,
    ) ?? -1;
}

function ensureAudioInput(node, i) {
    const name = audioInputName(i);
    if (findInput(node, name) >= 0) return;

    node.addInput(name, "AUDIO", {
        label: `audio ${i}`,
    });
}

function removeAudioInput(node, i) {
    const slot = findInput(node, audioInputName(i));
    if (slot < 0) return;

    if (node.inputs?.[slot]?.link != null) {
        node.disconnectInput(slot);
    }
    node.removeInput(slot);
}

function findPositionWidget(node, i) {
    return node.widgets?.find(
        (w) => w.name === positionWidgetName(i),
    );
}

function ensurePositionWidget(node, i, initialValue) {
    let widget = findPositionWidget(node, i);

    if (widget) {
        widget.value = initialValue;
        return widget;
    }

    widget = node.addWidget(
        "number",
        positionWidgetName(i),
        initialValue,
        (value) => {
            widget.value = Math.trunc(Number(value));
            writeState(node);
        },
        {
            min: 0,
            max: 99999,
            step: 1,
            precision: 0,
        },
    );

    // The hidden server-declared audio_state widget is the durable source
    // of truth for the position list.
    widget.serialize = false;
    widget.options ??= {};
    widget.options.serialize = false;
    return widget;
}

function removePositionWidget(node, i) {
    const widget = findPositionWidget(node, i);
    if (!widget || !node.widgets) return;

    const index = node.widgets.indexOf(widget);
    if (index >= 0) {
        node.widgets.splice(index, 1);
    }
}

function findStrengthWidget(node, i) {
    return node.widgets?.find(
        (w) => w.name === strengthWidgetName(i),
    );
}

function ensureStrengthWidget(node, i, initialValue) {
    let widget = findStrengthWidget(node, i);

    if (widget) {
        widget.value = initialValue;
        return widget;
    }

    widget = node.addWidget(
        "number",
        strengthWidgetName(i),
        initialValue,
        (value) => {
            widget.value = Math.min(
                1,
                Math.max(0.05, Number(value)),
            );
            writeState(node);
        },
        {
            min: 0.05,
            max: 1,
            step: 0.01,
            precision: 2,
            tooltip:
                "Clip influence on that zone: 1.0 pins it exactly; 0.5 is " +
                "half clip / half model generation; 0.1 is a light hint " +
                "(the model creates most of the sound). Continuous, no " +
                "cutoffs.",
        },
    );

    widget.serialize = false;
    widget.options ??= {};
    widget.options.serialize = false;
    return widget;
}

function removeStrengthWidget(node, i) {
    const widget = findStrengthWidget(node, i);
    if (!widget || !node.widgets) return;

    const index = node.widgets.indexOf(widget);
    if (index >= 0) {
        node.widgets.splice(index, 1);
    }
}

function writeState(node) {
    const raw = stateWidget(node);
    if (!raw) return;

    const positions = [];
    const strengths = [];
    for (
        let i = 1;
        i <= node._h3CustomAudioCount;
        i++
    ) {
        positions.push(
            Math.trunc(
                Number(findPositionWidget(node, i)?.value ?? 1),
            ),
        );
        strengths.push(
            Math.min(
                1,
                Math.max(
                    0.05,
                    Number(
                        findStrengthWidget(node, i)?.value ?? 1,
                    ),
                ),
            ),
        );
    }

    raw.value = JSON.stringify({
        count: node._h3CustomAudioCount,
        positions,
        strengths,
    });
}

function ensureButtons(node) {
    if (
        node.widgets?.some(
            (w) => w.name === "+ Add audio",
        )
    ) {
        return;
    }

    const add = node.addWidget(
        "button",
        "+ Add audio",
        null,
        () => {
            if (
                node._h3CustomAudioCount >= MAX_AUDIOS
            ) {
                return;
            }

            const current = readState(node);
            const i = node._h3CustomAudioCount + 1;
            const previous =
                current.positions.at(-1) ?? 1;

            node._h3CustomAudioCount = i;
            ensureAudioInput(node, i);
            ensurePositionWidget(
                node,
                i,
                previous + 17,
            );
            ensureStrengthWidget(node, i, 1);
            writeState(node);
            refreshNode(node);
        },
    );
    add.serialize = false;

    const remove = node.addWidget(
        "button",
        "- Remove audio",
        null,
        () => {
            if (
                node._h3CustomAudioCount <= MIN_AUDIOS
            ) {
                return;
            }

            const i = node._h3CustomAudioCount;
            removeAudioInput(node, i);
            removePositionWidget(node, i);
            removeStrengthWidget(node, i);
            node._h3CustomAudioCount -= 1;
            writeState(node);
            refreshNode(node);
        },
    );
    remove.serialize = false;
}

function reorderWidgets(node) {
    if (!node.widgets) return;

    const raw = stateWidget(node);
    const normal = [];
    const slots = [];
    const buttons = [];

    for (const widget of node.widgets) {
        if (widget === raw) continue;

        if (/^audio \d+ (position|strength)$/.test(widget.name)) {
            slots.push(widget);
        } else if (
            widget.name === "+ Add audio" ||
            widget.name === "- Remove audio"
        ) {
            buttons.push(widget);
        } else {
            normal.push(widget);
        }
    }

    slots.sort((a, b) => {
        const ai = Number(
            a.name.match(/\d+/)?.[0] ?? 0,
        );
        const bi = Number(
            b.name.match(/\d+/)?.[0] ?? 0,
        );
        return ai - bi;
    });

    node.widgets = [
        ...(raw ? [raw] : []),
        ...normal,
        ...slots,
        ...buttons,
    ];
}

function refreshNode(node) {
    reorderWidgets(node);

    const size = node.computeSize?.();
    if (size) {
        node.setSize(size);
    }

    app.graph?.setDirtyCanvas?.(true, true);
}

function buildUI(node) {
    hideStateWidget(node);

    const state = readState(node);
    node._h3CustomAudioCount = state.count;

    // Remove only stale dynamic slots beyond the serialized count.
    for (
        let i = MAX_AUDIOS;
        i > state.count;
        i--
    ) {
        removeAudioInput(node, i);
        removePositionWidget(node, i);
        removeStrengthWidget(node, i);
    }

    for (let i = 1; i <= state.count; i++) {
        ensureAudioInput(node, i);
        ensurePositionWidget(
            node,
            i,
            state.positions[i - 1],
        );
        ensureStrengthWidget(
            node,
            i,
            state.strengths[i - 1],
        );
    }

    ensureButtons(node);
    writeState(node);
    refreshNode(node);
}

app.registerExtension({
    name: "seitanism.H3CustomAudio",

    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_NAME) return;

        const originalCreated =
            nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            const result =
                originalCreated?.apply(this, arguments);
            setTimeout(() => buildUI(this), 0);
            return result;
        };

        const originalConfigure =
            nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            const result =
                originalConfigure?.apply(this, arguments);
            setTimeout(() => buildUI(this), 0);
            return result;
        };
    },
});