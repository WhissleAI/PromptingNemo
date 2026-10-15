"""Natural-language environment phrases with targets, for training the prompt adapter's environment heads.

env_caption(rng) -> (phrase, target) where target = dict(scene, room, phone, snr, ev_start, ev_end) and the phrase is
appended to / prepended to a style caption ("whispering, in a busy cafe, then laughs"). About half of the training
captions get no environment phrase and all-none targets, so plain prompts keep mapping to clean speech.
"""
import random

from .acoustic import ROOMS, SCENES

SCENE_PHRASES = {
    "rain": ["with rain pouring outside", "during a rainstorm", "with rain in the background", "as rain patters on the window",
             "in the rain", "with heavy rain falling"],
    "wind": ["on a windy day", "with wind howling", "outside in strong wind", "with a gusty wind in the background"],
    "water": ["by the ocean", "next to a river", "with waves crashing nearby", "beside a babbling stream", "at the beach"],
    "thunderstorm": ["during a thunderstorm", "with thunder rumbling", "in a storm"],
    "traffic": ["next to a busy road", "with traffic passing by", "on a noisy street corner", "beside a highway"],
    "street": ["out on the street", "in a city square", "in a busy downtown street"],
    "crowd": ["in a crowded room", "at a party with people chatting", "in a noisy crowd", "with lots of people talking around"],
    "cafe": ["in a busy cafe", "at a coffee shop", "in a cafeteria", "at a cafe with chatter around"],
    "restaurant": ["in a restaurant", "at a crowded restaurant", "over dinner in a busy restaurant"],
    "office": ["in an office", "at work in an open-plan office", "in a meeting room", "in an office hallway"],
    "kitchen": ["in the kitchen", "while cooking in the kitchen", "with kitchen sounds in the background"],
    "home": ["at home in the living room", "in a living room", "at home with the TV on"],
    "park": ["in a park", "outdoors in a park", "at the park"],
    "birds": ["with birds chirping", "in a garden full of birdsong", "outside with birds singing"],
    "station": ["at a train station", "on a station platform", "in a busy station"],
    "train": ["on a train", "riding the subway", "on the metro"],
    "bus": ["on a bus", "riding a city bus"],
    "car": ["in a car", "while driving", "from inside a moving car"],
    "music": ["with music playing in the background", "over background music", "with a song playing"],
    "fan": ["with a fan humming", "with the air conditioner running", "with a constant hum in the room"],
    "fire": ["by a crackling fire", "next to a campfire", "beside the fireplace"],
    "static": ["over a static-filled line", "with hiss and static", "through white noise"],
    "machine": ["next to running machinery", "with a washing machine running", "in a workshop with machines"],
    "nature": ["out in a field", "in the countryside", "outdoors in nature"],
}
SNR_WORDS = [  # (phrase template, snr range)
    ("{x}", None), ("{x}", None), ("faint sounds, {x}", (18, 25)), ("{x}, very quietly in the background", (18, 25)),
    ("{x}, quite noisy", (5, 10)), ("{x}, very loud background", (0, 5)), ("barely audible over the noise, {x}", (0, 4)),
    ("{x}, noise clearly audible", (8, 14)),
]
ROOM_PHRASES = {
    "small": ["in a small room", "in a tiny closet", "in a small bathroom", "close to the walls of a small room"],
    "medium": ["in a room with some echo", "in a medium-sized room", "in a classroom", "with a bit of reverb"],
    "large": ["in a huge hall", "in a cathedral", "echoing in a big empty warehouse", "in a large concert hall",
              "in a gymnasium with lots of echo", "with heavy reverb"],
}
# explicit "no environment" phrasing: must map to clean speech (all-none targets), not to a quiet indoor scene
CLEAN_PHRASES = ["in a quiet studio", "clean studio recording", "with no background noise", "in a silent room",
                 "recorded on a professional microphone", "crystal clear audio", "in a soundproof booth",
                 "close to the microphone, studio quality", "noise-free", "like a podcast recorded in a studio"]
PHONE_PHRASES = ["over the phone", "on a phone call", "through a telephone line", "like a phone call", "on a bad phone line",
                 "on speakerphone"]
EVENT_PHRASES = {  # token -> (start phrases, end phrases)
    "laugh": (["starts with a laugh", "laughs before speaking", "after a laugh"], ["then laughs", "ending with a laugh", "followed by laughter", "and laughs afterwards"]),
    "cough": (["coughs first", "after a cough", "clears with a cough first"], ["then coughs", "ending with a cough", "followed by a cough"]),
    "sigh": (["sighs before speaking", "after a sigh"], ["then sighs", "ending with a sigh"]),
    "breath": (["takes a breath first", "after a deep breath"], ["then breathes out", "ending with a breath"]),
    "sneeze": (["sneezes first", "after sneezing"], ["then sneezes", "followed by a sneeze"]),
    "yawn": (["yawns first", "after a yawn"], ["then yawns", "ending with a yawn"]),
    "throat": (["clears their throat first", "after clearing their throat"], ["then clears their throat"]),
    "cry": (["sobs before speaking", "after crying"], ["then starts crying", "ending in sobs"]),
    "scream": (["screams first", "after a scream"], ["then screams"]),
    "cheer": (["cheers first", "after a cheer"], ["then cheers"]),
    "dog": (["a dog barks, then", "after a dog barks"], ["then a dog barks", "with a dog barking at the end"]),
    "cat": (["a cat meows, then"], ["then a cat meows"]),
    "knock": (["after a knock at the door", "a knock, then"], ["followed by a knock at the door", "then someone knocks"]),
    "door": (["a door slams, then", "after a door closes"], ["then a door slams", "followed by a door closing"]),
    "phone": (["the phone rings, then", "after the phone rings"], ["then the phone rings", "with a phone ringing at the end"]),
    "bell": (["a bell rings, then", "after the doorbell"], ["then a bell rings", "followed by the doorbell"]),
    "horn": (["a car honks, then"], ["then a car horn honks", "followed by honking"]),
    "applause": (["after applause", "applause, then"], ["followed by applause", "then the audience applauds"]),
    "glass": (["a glass shatters, then"], ["then a glass breaks", "followed by glass shattering"]),
    "siren": (["a siren wails, then"], ["then a siren passes", "with a siren at the end"]),
    "footsteps": (["footsteps approach, then"], ["then footsteps walk away"]),
    "typing": (["after some typing"], ["then types on a keyboard"]),
    "thunder": (["thunder cracks, then"], ["then thunder rumbles"]),
    "clock": (["a clock ticks, then"], ["then a clock ticks"]),
    "bird": (["a bird calls, then"], ["then a bird chirps"]),
}


def env_caption(rng=random, p_scene=0.6, p_room=0.3, p_phone=0.12, p_event=0.35, p_clean=0.15):
    parts = []
    t = dict(scene=None, room=None, phone=False, snr=None, ev_start=None, ev_end=None)
    if rng.random() < p_clean:
        return rng.choice(CLEAN_PHRASES), t
    if rng.random() < p_scene:
        sc = rng.choice(SCENES)
        tpl, rngsnr = rng.choice(SNR_WORDS)
        parts.append(tpl.replace("{x}", rng.choice(SCENE_PHRASES[sc])))
        t["scene"] = sc
        if rngsnr:
            t["snr"] = rng.uniform(*rngsnr)
    if rng.random() < p_room:
        t["room"] = rng.choice(ROOMS)
        parts.append(rng.choice(ROOM_PHRASES[t["room"]]))
    if rng.random() < p_phone:
        t["phone"] = True
        parts.append(rng.choice(PHONE_PHRASES))
    if rng.random() < p_event:
        tok = rng.choice(sorted(EVENT_PHRASES))
        start = rng.random() < 0.4
        parts.append(rng.choice(EVENT_PHRASES[tok][0 if start else 1]))
        t["ev_start" if start else "ev_end"] = tok
    rng.shuffle(parts)
    return ", ".join(parts), t


def combine(style_caption, env_phrase, rng=random):
    if not env_phrase:
        return style_caption
    if not style_caption:
        return env_phrase
    return f"{style_caption}, {env_phrase}" if rng.random() < 0.7 else f"{env_phrase}, {style_caption}"
