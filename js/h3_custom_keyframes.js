import { app } from "../../scripts/app.js";

import { makeDynamicNodeExtension } from "./h3_dynamic_ui.js";

app.registerExtension(
    makeDynamicNodeExtension("MiniMaxH3CustomKeyframes", {
        stateWidgetName: "keyframe_state",
        defaultState: { count: 3, positions: [1, 22, 79], strengths: [1, 1, 1] },
        slotPrefix: "keyframe_image_",
        slotType: "IMAGE",
        slotLabel: (i) => `keyframe ${i} image`,
        positionLabel: (i) => `keyframe ${i} position`,
        strengthLabel: (i) => `keyframe ${i} strength`,
        strengthTooltip:
            "How much of the run the model may re-render: 1.0 pins the " +
            "image exactly; 0.9 almost the image, minor reshaping; 0.5 half " +
            "pinned half model; 0.1 a light hint. The zone stays clean at " +
            "any strength.",
        minSlots: 1,
        maxSlots: 32,
        countProp: "_h3CustomKeyframeCount",
        addLabel: "+ Add keyframe",
        removeLabel: "- Remove keyframe",
        positionRegex: /^keyframe \d+ (position|strength)$/,
        extName: "seitanism.H3CustomKeyframes",
        hasStrength: true,
    }),
);