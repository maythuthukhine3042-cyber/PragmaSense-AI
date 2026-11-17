"""All prompts in one place so they can be copied into the paper appendix."""


def _situation(context):
    return f"Situation: {context}\n" if context else ""


def zero_shot(thai, context=""):
    # also used as the System-1 draft of PragmaSense-AI
    return (
        "Translate the following spoken Thai utterance into natural spoken Myanmar (Burmese), "
        "written in Unicode Myanmar script.\n"
        f"{_situation(context)}"
        f"Thai: {thai}\n"
        "Output only the Myanmar translation, nothing else."
    )


def chain_of_thought(thai, context=""):
    return (
        "Translate the following spoken Thai utterance into natural spoken Myanmar (Burmese), "
        "written in Unicode Myanmar script.\n"
        f"{_situation(context)}"
        f"Thai: {thai}\n"
        "First think step by step about the speaker's politeness level, tone, and the meaning of "
        "any sentence-final particles, and how to express them in Myanmar. Then give the final answer.\n"
        "Use exactly this format:\n"
        "Reasoning: <your reasoning>\n"
        "Translation: <Myanmar translation only>"
    )


def kb_guided(thai, context, profile, kb):
    """Ablation: the knowledge base is given up front, but there is no critic loop."""
    hints = []
    for fn in sorted(profile.functions):
        hints.append(f"- express '{fn}' (e.g. {', '.join(kb.suggest(fn, profile.gender))})")
    hint_text = "\n".join(hints) if hints else "- plain, neutral register"
    return (
        "Translate the following spoken Thai utterance into natural spoken Myanmar (Burmese), "
        "written in Unicode Myanmar script.\n"
        f"{_situation(context)}"
        f"Thai: {thai}\n"
        f"Pragmatic analysis of the source: register level {profile.level:.1f}/4 "
        f"(1 = rough, 2 = plain, 3 = soft, 4 = polite), speaker gender: {profile.gender}.\n"
        f"The translation should:\n{hint_text}\n"
        "Output only the Myanmar translation, nothing else."
    )


def revise(thai, context, previous, critique):
    return (
        "You translated a spoken Thai utterance into Myanmar (Burmese). A pragmatic checker found "
        "a problem with the tone of your translation. Revise the translation to fix ONLY the problem "
        "described, keeping the meaning the same and the Myanmar natural.\n"
        f"{_situation(context)}"
        f"Thai: {thai}\n"
        f"PREVIOUS TRANSLATION:\n{previous}\n"
        f"CRITIQUE:\n{critique}\n"
        "Output only the revised Myanmar translation, nothing else."
    )


def self_critique(thai, context, translation):
    """Ablation (Self-Refine style): the LLM critiques itself without the knowledge base."""
    return (
        "Check whether this Myanmar translation keeps the politeness level, tone and the function of "
        "the sentence-final particles of the spoken Thai source.\n"
        f"{_situation(context)}"
        f"Thai: {thai}\n"
        f"Myanmar: {translation}\n"
        "If it is fine, answer exactly: OK\n"
        "Otherwise, answer with one or two sentences of feedback describing what to fix."
    )


def clean(text):
    """Keep only the translation line from a model answer."""
    if "Translation:" in text:
        text = text.split("Translation:")[-1]
    lines = [l.strip() for l in text.strip().splitlines() if l.strip()]
    if not lines:
        return ""
    out = lines[0]
    for prefix in ("Myanmar:", "Burmese:", "မြန်မာ:"):
        if out.startswith(prefix):
            out = out[len(prefix):].strip()
    return out.strip(" \"'`“”")
