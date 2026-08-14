import { app } from "../../scripts/app.js";

import { makeDynamicNodeExtension } from "./h3_dynamic_ui.js";

app.registerExtension(
    makeDynamicNodeExtension("MiniMaxH3CustomVideo", {
        stateWidgetName: "video_state",
        defaultState: { count: 1, positions: [1], strengths: [1] },
        slotPrefix: "video_",
        slotType: "IMAGE",
        slotLabel: (i) => `video ${i}`,
        positionLabel: (i) => `video ${i} position`,
        strengthLabel: (i) => `video ${i} strength`,
        extraInputs: [
            {
                prefix: "video_audio_",
                type: "AUDIO",
                label: (i) => `video ${i} audio`,
            },
        ],
        minSlots: 1,
        maxSlots: 8,
        countProp: "_h3CustomVideoCount",
        addLabel: "+ Add video",
        removeLabel: "- Remove video",
        positionRegex: /^video \d+ (position|strength)$/,
        extName: "seitanism.H3CustomVideo",
        hasStrength: true,
    }),
);