import type { Tier } from "./api";

// What each tier is called on screen. The data keeps Medium and High (ml.predictions, the scoreboard). Since final_1
// (25 Oct) the tiers are fixed chances of a PUB alert within the hour: Elevated from 5%, and Moderate from 1% (final_1)
// or 3% (final_2). The track record reads the live model's own cut-offs.
export const TIER_LABEL: Record<Tier, string> = {
  Low: "Low",
  Medium: "Moderate",
  High: "Elevated",
};

export const TIER_MEANING: Record<Tier, string> = {
  Low: "A flash-flood alert here in the next hour is unlikely.",
  Medium: "PUB may issue a flash-flood alert here in the next hour.",
  High: "PUB is more likely to issue a flash-flood alert here in the next hour.",
};

// the map legend's short form of TIER_MEANING
export const TIER_SHORT: Record<Tier, string> = {
  Low: "alert unlikely",
  Medium: "PUB alert possible",
  High: "PUB alert more likely",
};

export type Audience = "commute" | "driving" | "home";

// residents and commuters first: most people meet a flash flood on the way somewhere
export const AUDIENCES: { id: Audience; label: string }[] = [
  { id: "commute", label: "Walking, bus or MRT" },
  { id: "driving", label: "Driving" },
  { id: "home", label: "At home" },
];

// Fixed advice per tier and audience, as the plan asks: what a person can do in the hour, nothing the model can't back.
// Moderate is an early heads-up, so its advice is light. Elevated sits where the rule-matched Medium did before final_1,
// so it keeps that advice.
export const ADVICE: Record<Tier, Record<Audience, string>> = {
  Low: {
    commute: "No change needed. Paths and roads here are unlikely to flood in the next hour.",
    driving: "No change needed. Roads here are unlikely to flood in the next hour.",
    home: "Nothing to do now. FloodSense checks again every 5 minutes.",
  },
  Medium: {
    commute: "No need to change plans yet. If the rain gets heavy, keep away from underpasses and paths beside canals.",
    driving: "Drive with care if the rain gets heavy. Low roads and underpasses flood first.",
    home: "Nothing to do yet. If the rain keeps up, check FloodSense again before you go out.",
  },
  High: {
    commute: "Allow extra time. Keep to sheltered walkways, and avoid underpasses and paths beside drains and canals.",
    driving: "Avoid low roads and underpasses here if you can. Check PUB's alerts before you set off.",
    home: "If you live on the ground floor, lift valuables, documents and plugged-in appliances off the floor.",
  },
};

export const PUB_CHANNEL = "https://t.me/pubfloodalerts";

// The Genie space's sample questions (resources/genie.yml), each checked against its answer on 25 Oct 2026. Change
// them together.
export const AGENCY_QUESTIONS = [
  "Which areas had the most FloodSense false alarms after training?",
  "How much warning did PUB and FloodSense give before each flash flood in 2026?",
  "How does FloodSense compare with the rainfall rule?",
  "Which areas had the most PUB flood alerts in 2025?",
  "How has Singapore's flood-prone land changed since 2022?",
];

// Questions to start from in the other views, for the same Genie space, each checked against its answer on 26 Oct 2026
export const ASK_QUESTIONS = {
  now: [
    "Which areas have a FloodSense warning in force right now?",
    "Which areas had the most PUB flood alerts in 2025?",
    "When and where did PUB last report a flash flood?",
  ],
  replay: [
    "Which days in 2026 had the most PUB flood alerts?",
    "How much warning did PUB and FloodSense give before each flash flood in 2026?",
    "In which areas did FloodSense most often warn before PUB after training?",
  ],
  reliability: [
    "How does FloodSense compare with the rainfall rule?",
    "What share of flash floods did FloodSense warn of before PUB?",
    "Which areas had the most FloodSense false alarms after training?",
  ],
};
