"""PragmaSense-AI pipeline and the comparison systems.

Tier 1  Perception     : extract Thai particles -> target register profile
Tier 2  Generator (S1) : fast draft translation by the LLM (working memory)
Tier 3  Critic (S2)    : audit against the knowledge base, compute delta,
                         send critique back; gated revision loop (k_max)
"""
from dataclasses import dataclass, field

from . import prompts
from .critic import PragmaticCritic
from .kb import KnowledgeBase


@dataclass
class Result:
    system: str
    translation: str
    llm_calls: int = 0
    latency: float = 0.0
    trace: list = field(default_factory=list)
    iterations: int = 0          # number of revision rounds actually used
    status: str = ""             # accepted_draft / fixed / unresolved / stagnated


class PragmaSense:
    def __init__(self, llm, kb=None, k_max=2, threshold=None):
        self.llm = llm
        self.kb = kb or KnowledgeBase()
        self.critic = PragmaticCritic(self.kb, threshold)
        self.k_max = k_max

    # --------------------------------------------------------- perception
    def perceive(self, thai, speaker_gender=""):
        profile, prop = self.kb.analyse_thai(thai)
        if profile.gender == "unknown" and speaker_gender in ("male", "female"):
            profile.gender = speaker_gender
        return profile, prop

    def _ask(self, prompt, res):
        text, lat = self.llm.generate(prompt)
        res.llm_calls += 1
        res.latency += lat
        return text

    # ----------------------------------------------------- full pipeline
    def translate(self, thai, context="", speaker_gender=""):
        res = Result(system="pragmasense", translation="")
        profile, prop = self.perceive(thai, speaker_gender)
        res.trace.append({"stage": "perception", "particles": profile.marker_texts(),
                          "propositional": prop, "functions": sorted(profile.functions),
                          "target_level": profile.level, "gender": profile.gender})

        # System 1: intuitive draft (same prompt as the zero-shot baseline)
        hyp = prompts.clean(self._ask(prompts.zero_shot(thai, context), res))
        verdict = self.critic.audit(profile, hyp)
        res.trace.append(_trace_step("draft", hyp, verdict))
        best_hyp, best_v = hyp, verdict

        k = 0
        while not best_v.accepted and k < self.k_max:
            k += 1
            new = prompts.clean(self._ask(prompts.revise(thai, context, best_hyp, best_v.critique), res))
            v = self.critic.audit(profile, new)
            improved = v.score() < best_v.score()
            res.trace.append(_trace_step(f"revision {k}", new, v, gate="accepted" if improved else "rejected (no gain)"))
            if improved:
                best_hyp, best_v = new, v
            else:
                break  # convergence gate: stop when a revision does not help

        res.translation = best_hyp
        res.iterations = k
        if verdict.accepted:
            res.status = "accepted_draft"
        elif best_v.accepted:
            res.status = "fixed"
        elif k < self.k_max:
            res.status = "stagnated"
        else:
            res.status = "unresolved"
        return res

    # ------------------------------------------------- comparison systems
    def zero_shot(self, thai, context=""):
        res = Result(system="zero_shot", translation="")
        res.translation = prompts.clean(self._ask(prompts.zero_shot(thai, context), res))
        return res

    def chain_of_thought(self, thai, context=""):
        res = Result(system="cot", translation="")
        raw = self._ask(prompts.chain_of_thought(thai, context), res)
        res.translation = prompts.clean(raw)
        res.trace.append({"stage": "raw", "text": raw})
        return res

    def kb_guided(self, thai, context="", speaker_gender=""):
        """Ablation A: knowledge base in the prompt, no critic loop."""
        res = Result(system="kb_no_critic", translation="")
        profile, _ = self.perceive(thai, speaker_gender)
        res.translation = prompts.clean(self._ask(prompts.kb_guided(thai, context, profile, self.kb), res))
        return res

    def self_refine(self, thai, context=""):
        """Ablation B: LLM critiques itself (no knowledge base), one round."""
        res = Result(system="self_refine", translation="")
        hyp = prompts.clean(self._ask(prompts.zero_shot(thai, context), res))
        fb = self._ask(prompts.self_critique(thai, context, hyp), res).strip()
        res.trace.append({"stage": "self-critique", "text": fb})
        if fb.upper().startswith("OK"):
            res.translation = hyp
            res.status = "accepted_draft"
        else:
            res.translation = prompts.clean(self._ask(prompts.revise(thai, context, hyp, fb), res))
            res.iterations = 1
            res.status = "revised"
        return res


def _trace_step(name, hyp, v, gate=None):
    d = {"stage": name, "hypothesis": hyp, "markers": v.found_markers,
         "generated_level": v.generated_level, "target_level": v.target_level,
         "delta": round(v.delta, 2), "missing": v.missing_functions,
         "gender_error": v.gender_error, "accepted": v.accepted, "critique": v.critique}
    if gate:
        d["gate"] = gate
    return d
