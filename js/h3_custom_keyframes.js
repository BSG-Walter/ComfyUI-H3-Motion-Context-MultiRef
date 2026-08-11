import { app } from "../../scripts/app.js";

import { makeDynamicNodeExtension } from "./h3_dynamic_ui.js";

app.registerExtension(
    makeDynamicNodeExtension("MiniMaxH3CustomKeyframes", {
        stateWidgetName: "keyframe_state",
        defaultState: { count: 3, positions: [1, 22, 79] },
        slotPrefix: "keyframe_image_",
        slotType: "IMAGE",
        slotLabel: (i) => `keyframe ${i} image`,
        positionLabel: (i) => `keyframe ${i} position`,
        minSlots: 1,
        maxSlots: 32,
        countProp: "_h3CustomKeyframeCount",
        addLabel: "+ Add keyframe",
        removeLabel: "- Remove keyframe",
        positionRegex: /^keyframe \d+ position$/,
        extName: "seitanism.H3CustomKeyframes",
    }),
);