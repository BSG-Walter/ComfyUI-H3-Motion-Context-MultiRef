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
        strengthTooltip:
            "How much of the clip the model may re-render: 1.0 pins it " +
            "exactly; 0.9 almost the clip, minor reshaping; 0.5 half clip " +
            "half model; 0.1 a light hint. The zone stays clean at any " +
            "strength.",
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