from __future__ import annotations

import json
from datetime import date
from typing import Any, Dict, Optional

from .config import AppConfig


def _brand_context(config: AppConfig) -> str:
    brand = config.brand
    payload = {
        "brand_name": brand.name,
        "handle": brand.handle,
        "app_summary": brand.app_summary,
        "launch_window_days": brand.launch_window_days,
        "goal": brand.goal,
        "audience": brand.audience,
        "voice_rules": brand.voice_rules,
        "cities": brand.cities,
        "price_points_egp": brand.price_points_egp,
        "content_pillars": brand.content_pillars,
        "pain_points": brand.pain_points,
        "banned_phrases": brand.banned_phrases,
        "viral_edit_text_bank": brand.viral_edit_text_bank
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _series_context(config: AppConfig, continuity_state: Dict[str, Any]) -> str:
    if not config.series.enabled:
        return json.dumps({"series_enabled": False}, ensure_ascii=False, indent=2)

    recent_episodes = continuity_state.get("episodes", [])[-3:]
    payload = {
        "series_enabled": True,
        "title": config.series.title,
        "logline": config.series.logline,
        "visual_mode": config.series.visual_mode,
        "animation_style": config.series.animation_style,
        "food_rendering": config.series.food_rendering,
        "protagonist": config.series.protagonist,
        "supporting_cast": config.series.supporting_cast,
        "location_bible": config.series.location_bible,
        "style_bible": config.series.style_bible,
        "act_structure": config.series.act_structure,
        "next_episode_number": continuity_state.get("next_episode_number", 1),
        "active_visual_locks": continuity_state.get("active_visual_locks", []),
        "recent_episodes": recent_episodes,
        "episode_guide_summary": config.series.get_episode_guide_summary(),
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _episode_beat_context(episode_beat: Optional[Dict[str, Any]]) -> str:
    """Return a JSON block describing the specific episode beat to generate, or empty string."""
    if not episode_beat:
        return ""
    return json.dumps(episode_beat, ensure_ascii=False, indent=2)


def _series_rules(config: AppConfig) -> str:
    if not config.series.enabled:
        return """
- Use a single-video concept, not a season arc.
- Focus on standalone stop-scroll performance.
""".strip()

    return f"""
- This is a recurring episodic series, not a one-off ad.
- The visual world is {config.series.visual_mode}.
- Karim (كريم منصور) must ALWAYS look exactly as described in the protagonist visual_locks: messy dark curly hair, green rubber wristband on right hand, cracked Android phone. These never change unless the episode explicitly says so.
- Every shot where Karim's hands are visible must include his green wristband on the right hand.
- The home kitchen's blue and white speckled pot is a recurring visual anchor — include it when in the kitchen.
- Keep the world aligned with this animation style: {config.series.animation_style}
- Keep food aligned with this rendering rule: {config.series.food_rendering}
- Make the episode work as a self-contained short, but leave a reason to return for the next episode.
- Characters only appear if they are listed in that episode's key_characters — do not add characters who haven't been introduced yet.
- Tone and emotional weight must match the act this episode belongs to.
""".strip()


def _character_consistency_rules() -> str:
    return """
CHARACTER CONSISTENCY RULES (non-negotiable for every shot prompt):
- Karim: messy dark curly hair | warm brown eyes | short trimmed beard | green rubber wristband on RIGHT hand | cracked Android phone | home apron over blue hoodie or white t-shirt (chef whites only in professional kitchen episodes)
- Fares: tall and lanky | Ahly jersey or puffer jacket | cheap gimbal in hand | wide expressive eyes
- Um Karim: round face | silver-streaked hair pulled back | colorful galabeyya | reading glasses sometimes on head
- Chef Hatem: heavyset | bald | thick black mustache | chef whites with rolled-up sleeves | calm deliberate posture
- Nour: short natural hair | round wire-frame glasses | camera or phone always in hand
- Salma: chef whites in kitchen scenes | hair tied back tightly | precise efficient movements
- NEVER draw any character differently from their visual_locks. The green wristband is the #1 continuity anchor for Karim.
- All characters must be CARTOON style — bold outlines, expressive faces. NEVER photoreal.
""".strip()


def research_prompt(config: AppConfig, topic_hint: str, continuity_state: Dict[str, Any], episode_beat: Optional[Dict[str, Any]] = None) -> str:
    episode_section = ""
    if episode_beat:
        ep_num = episode_beat.get("episode_number", "?")
        ep_title = episode_beat.get("title_en", "")
        ep_act = episode_beat.get("act", "")
        ep_tone = episode_beat.get("emotion_tone", "")
        ep_beat_text = episode_beat.get("beat", "")
        episode_section = f"""
PINNED EPISODE TO GENERATE:
Episode {ep_num} (Act {ep_act}) — "{ep_title}"
Emotional tone: {ep_tone}
Story beat: {ep_beat_text}

Research must support this specific episode. Find trend signals, TikTok hooks, and Egyptian cultural references that can be woven into this exact story moment.
""".strip()

    return f"""
You are the lead strategist for an Egyptian TikTok growth factory.
Today's date is {date.today().isoformat()}.

Brand context:
{_brand_context(config)}

Series context:
{_series_context(config, continuity_state)}

{episode_section}

Task:
Use Google Search grounding to research what is working right now for Egyptian TikTok food and discovery content.
Search every time. Prioritize recent Egypt-relevant signals, comments language, Ramadan or seasonal context if relevant, pricing tension in EGP, TikTok-style hook patterns, and stop-scroll visual ideas.

Topic hint:
{topic_hint}

Rules:
- Write findings in English for analysis, but every example of hooks, captions, overlays, and cliffhangers must be in colloquial Egyptian Arabic.
- Focus on native TikTok phrasing, not ad copy.
- Mention what to avoid if it feels generic, formal, or non-Egyptian.
- Prefer hooks that create tension through price, challenge, comparison, surprise, or local relevance.
- Include specific visual edit ideas that fit short-form food videos.
- If the series context is enabled, include episode-worthy conflicts, stereotypes, and struggles that are recognizable in Egypt without punching down.
- If a pinned episode is specified, focus research on supporting that exact story beat.

Return JSON only with this shape:
{{
  "market_angle": "string",
  "trend_signals": [
    {{
      "signal": "string",
      "why_it_matters": "string",
      "recency_note": "string"
    }}
  ],
  "audience_tensions": ["string"],
  "hook_patterns": [
    {{
      "pattern_name": "string",
      "why_it_stops_scroll": "string",
      "example_ar": "string"
    }}
  ],
  "egyptian_language_dos": ["string"],
  "egyptian_language_donts": ["string"],
  "visual_edit_moves": ["string"],
  "creator_notes": ["string"],
  "concept_directions": ["string"],
  "episode_tensions": ["string"],
  "search_queries_to_reuse": ["string"]
}}
""".strip()


def concept_prompt(config: AppConfig, research: Dict[str, Any], topic_hint: str, continuity_state: Dict[str, Any], episode_beat: Optional[Dict[str, Any]] = None) -> str:
    episode_instruction = ""
    pinned_fields = ""
    if episode_beat:
        ep_num = episode_beat.get("episode_number", "?")
        ep_title_ar = episode_beat.get("title_ar", "")
        ep_title_en = episode_beat.get("title_en", "")
        ep_act = episode_beat.get("act", "")
        ep_tone = episode_beat.get("emotion_tone", "")
        ep_beat_text = episode_beat.get("beat", "")
        ep_chars = episode_beat.get("key_characters", [])
        ep_location = episode_beat.get("primary_location", "")
        ep_goal = episode_beat.get("continuity_goal", "")
        ep_cliffhanger = episode_beat.get("cliffhanger", "")

        episode_instruction = f"""
MANDATORY: Every concept you generate MUST be for this specific episode:
Episode {ep_num} (Act {ep_act}) — "{ep_title_en}" / "{ep_title_ar}"
Story beat: {ep_beat_text}
Characters in this episode: {', '.join(ep_chars)}
Primary location: {ep_location}
Emotional tone: {ep_tone}
Continuity goal: {ep_goal}
Cliffhanger: {ep_cliffhanger}

The concepts may vary in HOW they frame, hook, and edit this episode — but they must all be versions of THIS specific story beat. Do not invent a different episode. Do not skip or alter the story events. Karim's visual locks and character details must be embedded in every concept.
""".strip()
        pinned_fields = f"""
      "episode_number_hint": {ep_num},
"""
    else:
        pinned_fields = """
      "episode_number_hint": 0,
"""

    return f"""
You are generating TikTok video concepts for a pre-launch Egyptian app.

Brand context:
{_brand_context(config)}

Series context:
{_series_context(config, continuity_state)}

Trend research:
{json.dumps(research, ensure_ascii=False, indent=2)}

Topic hint:
{topic_hint}

{episode_instruction}

Task:
Generate {config.pipeline.concept_pool_size} video concepts for {config.brand.name}.
Each concept must feel native to Egyptian TikTok and must be strong enough to deserve 10 rewrite rounds.

{_series_rules(config)}

Rules:
- Hooks must be in colloquial Egyptian Arabic.
- Concepts should feel post-worthy today.
- Mention the product idea clearly enough that viewers understand the value.
- The strongest concepts use money, food craving, neighborhood relevance, challenge energy, or emotional struggle.
- Each concept should feel like one episode inside the recurring "الشيف كريم" social-comedy world.
- Only include characters who are present in this episode's key_characters list.
- Karim's visual_locks must be embedded in every concept — the green wristband and cracked phone are always present.

Return JSON only:
{{
  "concepts": [
    {{
      "slug": "short-latin-slug",
{pinned_fields}      "name": "string",
      "persona": "string",
      "core_emotion": "string",
      "hook_ar": "string",
      "angle_summary": "string",
      "edit_pattern": "string",
      "episode_conflict": "string",
      "continuity_goal": "string",
      "cliffhanger": "string",
      "why_it_fits_now": "string",
      "model_score": 0
    }}
  ]
}}
""".strip()


def initial_script_prompt(config: AppConfig, research: Dict[str, Any], concept: Dict[str, Any], continuity_state: Dict[str, Any], episode_beat: Optional[Dict[str, Any]] = None) -> str:
    episode_section = ""
    if episode_beat:
        episode_section = f"""
PINNED EPISODE BEAT (you must follow this exactly):
{_episode_beat_context(episode_beat)}
""".strip()

    return f"""
You are writing a short-form TikTok script in colloquial Egyptian Arabic for the series "الشيف كريم".

Brand context:
{_brand_context(config)}

Series context:
{_series_context(config, continuity_state)}

{_character_consistency_rules()}

Research:
{json.dumps(research, ensure_ascii=False, indent=2)}

Chosen concept:
{json.dumps(concept, ensure_ascii=False, indent=2)}

{episode_section}

Task:
Write the first draft for one vertical TikTok episode of "الشيف كريم" for {config.brand.name}.
Make it native, fast, and scroll-stopping.

{_series_rules(config)}

Rules:
- Entire spoken output must be colloquial Egyptian Arabic.
- No Modern Standard Arabic ad language.
- The first 2 seconds must hit with price, challenge, comparison, or surprise.
- Keep the voiceover tight enough for roughly 18-28 seconds.
- Make the product value obvious without sounding like a sales script.
- Include edit text that feels like viral Egyptian food edits.
- Show one clear emotional turn, one social obstacle, and one return-worthy cliffhanger.
- If a pinned episode beat is provided, the script MUST follow those events. Do not invent different story events.
- Karim must speak in his voice: passionate, proud, stubborn, funny under pressure.
- Only reference characters who appear in this episode's key_characters.

Return JSON only:
{{
  "episode_summary": "string",
  "episode_number": 0,
  "act": 0,
  "hook": "string",
  "voiceover": "string",
  "beat_sheet": [
    {{
      "time": "0-2",
      "purpose": "string",
      "line": "string"
    }}
  ],
  "onscreen_text": ["string"],
  "editor_notes": ["string"],
  "primary_caption": "string",
  "caption_variants": ["string"],
  "hashtags": ["string"],
  "pinned_comment": "string",
  "cliffhanger": "string",
  "continuity_updates": ["string"],
  "cta": "string",
  "dialogue": [
    {{
      "speaker": "narrator | كريم | فارس | أم كريم | الشيف حاتم | نور | سلمى",
      "text": "the spoken line in colloquial Egyptian Arabic"
    }}
  ]
}}
""".strip()


def script_iteration_prompt(
    config: AppConfig,
    research: Dict[str, Any],
    concept: Dict[str, Any],
    current_script: Dict[str, Any],
    iteration_index: int,
    continuity_state: Dict[str, Any],
    episode_beat: Optional[Dict[str, Any]] = None
) -> str:
    episode_section = ""
    if episode_beat:
        episode_section = f"""
PINNED EPISODE BEAT (must be preserved in every revision):
{_episode_beat_context(episode_beat)}
""".strip()

    return f"""
You are revising a TikTok script for virality.
This is script iteration {iteration_index} out of {config.pipeline.script_iterations}.

Brand context:
{_brand_context(config)}

Series context:
{_series_context(config, continuity_state)}

{_character_consistency_rules()}

Research:
{json.dumps(research, ensure_ascii=False, indent=2)}

Concept:
{json.dumps(concept, ensure_ascii=False, indent=2)}

{episode_section}

Current script:
{json.dumps(current_script, ensure_ascii=False, indent=2)}

Task:
Critique the current script and rewrite it to increase:
1. scroll-stop power
2. Egyptian dialect authenticity
3. visual momentum
4. comment bait
5. clarity of the app idea
6. rewatchability
7. character voice consistency (Karim must sound like Karim)
8. episode story fidelity (the pinned beat events must be preserved)

{_series_rules(config)}

Rules:
- Stay in colloquial Egyptian Arabic for all script output.
- Avoid the banned phrases in brand context.
- Use short aggressive edit text, not paragraphs.
- If the current script is too generic, make it more local and more debatable.
- Preserve character voice and story continuity. Karim's visual_locks must not be altered.
- If a pinned episode beat is provided, NEVER change the core story events — only improve how they are presented.
- Only reference characters from this episode's key_characters.

Return JSON only:
{{
  "critique": {{
    "strengths": ["string"],
    "issues": ["string"],
    "next_fix": "string"
  }},
  "score_breakdown": {{
    "hook_strength": 0,
    "egyptian_authenticity": 0,
    "visual_potential": 0,
    "clarity": 0,
    "comment_bait": 0,
    "conversion_tease": 0,
    "continuity_strength": 0,
    "episode_fidelity": 0,
    "total": 0
  }},
  "script": {{
    "episode_summary": "string",
    "episode_number": 0,
    "act": 0,
    "hook": "string",
    "voiceover": "string",
    "beat_sheet": [
      {{
        "time": "0-2",
        "purpose": "string",
        "line": "string"
      }}
    ],
    "onscreen_text": ["string"],
    "editor_notes": ["string"],
    "primary_caption": "string",
    "caption_variants": ["string"],
    "hashtags": ["string"],
    "pinned_comment": "string",
    "cliffhanger": "string",
    "continuity_updates": ["string"],
    "cta": "string",
    "dialogue": [
      {{
        "speaker": "narrator | كريم | فارس | أم كريم | الشيف حاتم | نور | سلمى",
        "text": "the spoken line in colloquial Egyptian Arabic"
      }}
    ]
  }}
}}
""".strip()


def initial_video_plan_prompt(
    config: AppConfig,
    research: Dict[str, Any],
    concept: Dict[str, Any],
    final_script: Dict[str, Any],
    continuity_state: Dict[str, Any],
    episode_beat: Optional[Dict[str, Any]] = None
) -> str:
    episode_section = ""
    if episode_beat:
        episode_section = f"""
PINNED EPISODE BEAT (visual plan must match these events and this emotional tone):
{_episode_beat_context(episode_beat)}
""".strip()

    return f"""
You are turning a final TikTok script into Veo-ready shot prompts and edit instructions for the series "الشيف كريم".

Brand context:
{_brand_context(config)}

Series context:
{_series_context(config, continuity_state)}

{_character_consistency_rules()}

Research:
{json.dumps(research, ensure_ascii=False, indent=2)}

Concept:
{json.dumps(concept, ensure_ascii=False, indent=2)}

{episode_section}

Final script:
{json.dumps(final_script, ensure_ascii=False, indent=2)}

Task:
Create a vertical video plan for {config.assets.video_count_per_script} clips.
The Veo prompts must be written in English because the video model responds more reliably to English prompts, but all overlay text and captions must stay in colloquial Egyptian Arabic.

{_series_rules(config)}

CHARACTER VISUAL LOCKS FOR EVERY SHOT PROMPT:
Every shot that includes Karim must explicitly describe: messy dark curly hair, green rubber wristband on right hand, cracked Android phone if visible, home apron over blue hoodie or white t-shirt (or chef whites if this is a professional kitchen episode). These must appear word-for-word in every shot that features Karim.

Rules:
- Each clip should be visually simple and easy to stitch.
- Include negative prompts to reduce generic output: no photoreal humans, no text baked into the scene, no random characters.
- Use overlay text cards that feel native to Egyptian TikTok edits (these go in post, NOT in the generated video).
- Keep each scene compatible with {config.assets.video_duration_seconds} seconds and {config.assets.video_aspect_ratio}.
- Respect the protagonist, supporting cast, kitchen world, props, and visual palette as described in the series context.
- Visual mode is cartoon_hybrid — explicitly avoid photoreal humans. Use stylized expressive animation language.
- The character_sheet_prompt_en must be a complete, standalone description of Karim that could be used independently to regenerate him consistently: include every visual lock in full.
- The location_sheet_prompt_en must describe the kitchen with the blue and white speckled pot, cracked window tile, warm yellow light, and laundry lines visible outside.

Return JSON only:
{{
  "edit_pattern_name": "string",
  "cover_text_ar": "string",
  "subtitle_style": "string",
  "style_guardrails": ["string"],
  "character_sheet_prompt_en": "string",
  "location_sheet_prompt_en": "string",
  "sound_design_notes": ["string"],
  "shots": [
    {{
      "shot_index": 1,
      "duration_seconds": {config.assets.video_duration_seconds},
      "purpose": "string",
      "continuity_lock": "string",
      "shot_image_prompt_en": "string — a still-frame Imagen prompt describing the FIRST FRAME of this shot in full cartoon detail, including all character visual locks. This image will be used as the source frame for image-to-video generation.",
      "veo_prompt_en": "string",
      "negative_prompt_en": "string",
      "overlay_text_ar": "string",
      "transition_note": "string",
      "camera_note": "string"
    }}
  ]
}}
""".strip()


def video_iteration_prompt(
    config: AppConfig,
    research: Dict[str, Any],
    concept: Dict[str, Any],
    final_script: Dict[str, Any],
    current_plan: Dict[str, Any],
    iteration_index: int,
    continuity_state: Dict[str, Any],
    episode_beat: Optional[Dict[str, Any]] = None
) -> str:
    episode_section = ""
    if episode_beat:
        episode_section = f"""
PINNED EPISODE BEAT (visual consistency to this tone and these events must be maintained):
{_episode_beat_context(episode_beat)}
""".strip()

    return f"""
You are revising Veo shot prompts for virality and editability.
This is video-plan iteration {iteration_index} out of {config.pipeline.video_prompt_iterations}.

Brand context:
{_brand_context(config)}

Series context:
{_series_context(config, continuity_state)}

{_character_consistency_rules()}

Research:
{json.dumps(research, ensure_ascii=False, indent=2)}

Concept:
{json.dumps(concept, ensure_ascii=False, indent=2)}

{episode_section}

Final script:
{json.dumps(final_script, ensure_ascii=False, indent=2)}

Current video plan:
{json.dumps(current_plan, ensure_ascii=False, indent=2)}

Task:
Critique the current Veo prompts and improve them.
Increase visual clarity, local relevance, hook density, usefulness for short-form editing, and continuity stability.

{_series_rules(config)}

Rules:
- Veo prompts stay in English.
- Overlay text stays in colloquial Egyptian Arabic.
- Avoid text-heavy visuals inside the generated video.
- Visual mode is cartoon_hybrid — avoid photoreal language. Describe stylized motion, simplified geometry, expressive faces, repeatable backgrounds.
- Prioritize food close-up appeal, clear acting beats, and continuity locks.
- Every shot featuring Karim MUST include his full visual description in the veo_prompt_en: messy dark curly hair, green rubber wristband on right hand. If the previous iteration missed any of these, add them now.
- The character_sheet_prompt_en must include ALL of Karim's visual locks written out in full.
- Check: does every shot prompt explicitly lock Karim's appearance? If not, fix it.

Return JSON only:
{{
  "critique": {{
    "strengths": ["string"],
    "issues": ["string"],
    "next_fix": "string"
  }},
  "score_breakdown": {{
    "visual_hook": 0,
    "editability": 0,
    "local_relevance": 0,
    "clarity": 0,
    "asset_quality": 0,
    "continuity_strength": 0,
    "character_consistency": 0,
    "total": 0
  }},
  "video_plan": {{
    "edit_pattern_name": "string",
    "cover_text_ar": "string",
    "subtitle_style": "string",
    "style_guardrails": ["string"],
    "character_sheet_prompt_en": "string",
    "location_sheet_prompt_en": "string",
    "sound_design_notes": ["string"],
    "shots": [
      {{
        "shot_index": 1,
        "duration_seconds": {config.assets.video_duration_seconds},
        "purpose": "string",
        "continuity_lock": "string",
        "shot_image_prompt_en": "string — still-frame Imagen prompt for the FIRST FRAME of this shot, full cartoon detail including all character visual locks",
        "veo_prompt_en": "string",
        "negative_prompt_en": "string",
        "overlay_text_ar": "string",
        "transition_note": "string",
        "camera_note": "string"
      }}
    ]
  }}
}}
""".strip()


def caption_prompt(config: AppConfig, final_script: Dict[str, Any], video_plan: Dict[str, Any], continuity_state: Dict[str, Any], episode_beat: Optional[Dict[str, Any]] = None) -> str:
    episode_section = ""
    if episode_beat:
        ep_num = episode_beat.get("episode_number", "?")
        ep_cliffhanger = episode_beat.get("cliffhanger", "")
        episode_section = f"""
This is Episode {ep_num}. The episode cliffhanger from the story bible is:
"{ep_cliffhanger}"
Use this to inform the episode_teaser and pinned_comment without making them sound like TV promo copy.
""".strip()

    return f"""
You are writing caption packs for an Egyptian TikTok post for the series "الشيف كريم".

Brand context:
{_brand_context(config)}

Series context:
{_series_context(config, continuity_state)}

{episode_section}

Final script:
{json.dumps(final_script, ensure_ascii=False, indent=2)}

Video plan:
{json.dumps(video_plan, ensure_ascii=False, indent=2)}

Task:
Write one primary caption, three alternatives, one pinned comment, and six reply-bait comments.
All outputs must be colloquial Egyptian Arabic and feel native to TikTok comments.

Rules:
- Make viewers feel this episode belongs to a bigger ongoing story.
- Hint at the next episode without sounding like TV promo copy.
- The pinned comment should feel like Karim wrote it himself — in his voice.
- Reply baits should invite the kind of comments that feed the algorithm: arguments, reactions, recipe requests, location tags.

Return JSON only:
{{
  "primary_caption": "string",
  "caption_variants": ["string"],
  "hashtags": ["string"],
  "pinned_comment": "string",
  "reply_baits": ["string"],
  "episode_teaser": "string"
}}
""".strip()
