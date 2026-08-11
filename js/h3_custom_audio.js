import { app } from "../../scripts/app.js";

import { makeDynamicNodeExtension } from "./h3_dynamic_ui.js";

app.registerExtension(
    makeDynamicNodeExtension("MiniMaxH3CustomAudio", {
        stateWidgetName: "audio_state",
        defaultState: { count: 1, positions: [1], strengths: [1] },
        slotPrefix: "audio_",
        slotType: "AUDIO",
        slotLabel: (i) => `audio ${i}`,
        positionLabel: (i) => `audio ${i} position`,
        minSlots: 1,
        maxSlots: 16,
        countProp: "_h3CustomAudioCount",
        addLabel: "+ Add audio",
        removeLabel: "- Remove audio",
        positionRegex: /^audio \d+ (position|strength)$/,
        extName: "seitanism.H3CustomAudio",
        hasStrength: true,
    }),
);