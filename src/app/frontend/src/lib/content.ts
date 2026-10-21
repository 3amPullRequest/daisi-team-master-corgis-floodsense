import type { Tier } from "./api";

export const TIER_MEANING: Record<Tier, string> = {
  Low: "A flash-flood alert here in the next hour is unlikely.",
  Medium: "PUB may issue a flash-flood alert here in the next hour.",
  High: "A flash flood is likely here in the next hour.",
};

export type Audience = "driving" | "home" | "shop";

export const AUDIENCES: { id: Audience; label: string }[] = [
  { id: "driving", label: "Driving" },
  { id: "home", label: "At home" },
  { id: "shop", label: "Running a shop" },
];

// Fixed advice per tier and audience, as the plan asks: what a person can do in the hour, nothing the model can't back.
export const ADVICE: Record<Tier, Record<Audience, string>> = {
  Low: {
    driving: "No change to your plans. Roads here are unlikely to flood in the next hour.",
    home: "Nothing to do now. FloodSense checks again every 5 minutes.",
    shop: "Business as usual.",
  },
  Medium: {
    driving: "If you can, avoid low roads and underpasses here for the next hour, and check PUB's alerts before you set off.",
    home: "If you live on the ground floor, lift valuables, documents and plugged-in appliances off the floor.",
    shop: "Get flood barriers or sandbags ready, and move stock away from the entrance and off the floor.",
  },
  High: {
    driving: "Stay out of this area if you can. Never drive into flood water: it is often deeper and faster than it looks.",
    home: "Lift valuables and electrical items off the floor now, and keep away from drains and canals.",
    shop: "Put flood barriers in place now and move stock off the floor.",
  },
};

export const PUB_CHANNEL = "https://t.me/pubfloodalerts";
