import { app } from "../../scripts/app.js";

export function makeDynamicNodeExtension(nodeName, cfg) {
    const {
        stateWidgetName,
        defaultState,
        slotPrefix,
        slotType,
        slotLabel,
        positionLabel,
        minSlots,
        maxSlots,
        countProp,
        addLabel,
        removeLabel,
        positionRegex,
        extName,
        hasStrength = false,
        strengthLabel = (i) => `audio ${i} strength`,
        strengthTooltip =
            "How much of the clip the model may re-render: 1.0 pins it " +
            "exactly; 0.9 almost the clip, minor reshaping; 0.5 half clip " +
            "half model; 0.1 a light hint. The zone stays clean at any " +
            "strength.",
        extraInputs = [],
    } = cfg;

    const stateWidget = (node) => node.widgets?.find((w) => w.name === stateWidgetName);

    function readState(node) {
        const raw = stateWidget(node);
        const state = { count: defaultState.count, positions: [...defaultState.positions] };
        if (hasStrength) state.strengths = [...(defaultState.strengths ?? [])];
        try {
            const parsed = JSON.parse(raw?.value || "");
            if (Number.isInteger(parsed?.count)) state.count = Math.min(maxSlots, Math.max(minSlots, parsed.count));
            if (Array.isArray(parsed?.positions)) state.positions = parsed.positions.map((v) => Math.trunc(Number(v)));
            if (hasStrength && Array.isArray(parsed?.strengths)) {
                state.strengths = parsed.strengths.map((v) => {
                    const s = Number(v);
                    return Number.isFinite(s) ? Math.min(1, Math.max(0, s)) : 1;
                });
            }
        } catch (_) {}

        while (state.positions.length < state.count) {
            state.positions.push((state.positions.at(-1) ?? 1) + 17);
        }
        if (hasStrength) {
            while (state.strengths.length < state.count) state.strengths.push(1);
            state.strengths = state.strengths.slice(0, state.count);
        }
        state.positions = state.positions.slice(0, state.count);
        return state;
    }

    function hideStateWidget(node) {
        const widget = stateWidget(node);
        if (!widget || widget._h3CustomHidden) return;
        widget._h3CustomHidden = true;
        widget.computeSize = () => [0, -4];
    }

    const inputName = (i) => `${slotPrefix}${i}`;
    const extraInputName = (extra, i) => `${extra.prefix}${i}`;
    const findInput = (node, name) => node.inputs?.findIndex((input) => input.name === name) ?? -1;

    function ensureInput(node, i) {
        if (findInput(node, inputName(i)) < 0) {
            node.addInput(inputName(i), slotType, { label: slotLabel(i) });
        }
        for (const extra of extraInputs) {
            if (findInput(node, extraInputName(extra, i)) < 0) {
                node.addInput(extraInputName(extra, i), extra.type, { label: extra.label(i) });
            }
        }
    }

    function removeInput(node, i) {
        for (const name of [inputName(i), ...extraInputs.map((e) => extraInputName(e, i))]) {
            const slot = findInput(node, name);
            if (slot < 0) continue;
            if (node.inputs?.[slot]?.link != null) node.disconnectInput(slot);
            node.removeInput(slot);
        }
    }

    function ensurePositionWidget(node, i, initialValue) {
        let widget = node.widgets?.find((w) => w.name === positionLabel(i));
        if (widget) {
            widget.value = initialValue;
            return widget;
        }
        widget = node.addWidget("number", positionLabel(i), initialValue, (value) => {
            widget.value = Math.trunc(Number(value));
            writeState(node);
        }, { min: 0, max: 99999, step: 1, precision: 0 });
        widget.serialize = false;
        widget.options ??= {};
        widget.options.serialize = false;
        return widget;
    }

    function removeWidget(node, name) {
        const idx = node.widgets?.findIndex((w) => w.name === name) ?? -1;
        if (idx >= 0) node.widgets.splice(idx, 1);
    }

    function ensureStrengthWidget(node, i, initialValue) {
        let widget = node.widgets?.find((w) => w.name === strengthLabel(i));
        if (widget) {
            widget.value = initialValue;
            return widget;
        }
        widget = node.addWidget("number", strengthLabel(i), initialValue, (value) => {
            widget.value = Math.min(1, Math.max(0, Number(value)));
            writeState(node);
        }, { min: 0, max: 1, step: 0.01, precision: 2, tooltip: strengthTooltip });
        widget.serialize = false;
        widget.options ??= {};
        widget.options.serialize = false;
        return widget;
    }

    function writeState(node) {
        const raw = stateWidget(node);
        if (!raw) return;
        const positions = [];
        const strengths = [];
        for (let i = 1; i <= node[countProp]; i++) {
            positions.push(Math.trunc(Number(node.widgets?.find((w) => w.name === positionLabel(i))?.value ?? 1)));
            if (hasStrength) {
                strengths.push(Math.min(1, Math.max(0, Number(node.widgets?.find((w) => w.name === strengthLabel(i))?.value ?? 1))));
            }
        }
        const state = { count: node[countProp], positions };
        if (hasStrength) state.strengths = strengths;
        raw.value = JSON.stringify(state);
    }

    function ensureButtons(node) {
        if (node.widgets?.some((w) => w.name === addLabel)) return;

        const add = node.addWidget("button", addLabel, null, () => {
            if (node[countProp] >= maxSlots) return;
            const current = readState(node);
            const i = ++node[countProp];
            const previous = current.positions.at(-1) ?? 1;
            ensureInput(node, i);
            ensurePositionWidget(node, i, previous + 17);
            if (hasStrength) ensureStrengthWidget(node, i, 1);
            writeState(node);
            refreshNode(node);
        });
        add.serialize = false;

        const remove = node.addWidget("button", removeLabel, null, () => {
            if (node[countProp] <= minSlots) return;
            const i = node[countProp]--;
            removeInput(node, i);
            removeWidget(node, positionLabel(i));
            if (hasStrength) removeWidget(node, strengthLabel(i));
            writeState(node);
            refreshNode(node);
        });
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
            if (positionRegex.test(widget.name)) slots.push(widget);
            else if (widget.name === addLabel || widget.name === removeLabel) buttons.push(widget);
            else normal.push(widget);
        }
        slots.sort((a, b) => {
            const ai = Number(a.name.match(/\d+/)?.[0] ?? 0);
            const bi = Number(b.name.match(/\d+/)?.[0] ?? 0);
            return ai - bi;
        });
        node.widgets = [...(raw ? [raw] : []), ...normal, ...slots, ...buttons];
    }

    function refreshNode(node) {
        reorderWidgets(node);
        const size = node.computeSize?.();
        if (size) node.setSize(size);
        app.graph?.setDirtyCanvas?.(true, true);
    }

    function buildUI(node) {
        hideStateWidget(node);
        const state = readState(node);
        node[countProp] = state.count;
        for (let i = maxSlots; i > state.count; i--) {
            removeInput(node, i);
            removeWidget(node, positionLabel(i));
            if (hasStrength) removeWidget(node, strengthLabel(i));
        }
        for (let i = 1; i <= state.count; i++) {
            ensureInput(node, i);
            ensurePositionWidget(node, i, state.positions[i - 1]);
            if (hasStrength) ensureStrengthWidget(node, i, state.strengths[i - 1]);
        }
        ensureButtons(node);
        writeState(node);
        refreshNode(node);
    }

    return {
        name: extName,
        async beforeRegisterNodeDef(nodeType, nodeData) {
            if (nodeData.name !== nodeName) return;
            for (const hook of ["onNodeCreated", "onConfigure"]) {
                const orig = nodeType.prototype[hook];
                nodeType.prototype[hook] = function () {
                    const result = orig?.apply(this, arguments);
                    setTimeout(() => buildUI(this), 0);
                    return result;
                };
            }
        },
    };
}