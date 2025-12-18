from typing import Dict, List, Type

from pydantic import BaseModel

ALERTS_EXTRACTION_PROMPT = """You are an expert workplace safety and compliance analyzer. Analyze this employee comment for alerts requiring HR, management, or compliance attention.

COMMENT TO ANALYZE:
{text}

---

ALERT CATEGORIES WITH DEFINITIONS:

**HARASSMENT & DISCRIMINATION:**
- discrimination: Unfair treatment based on race, gender, age, religion, disability, ethnicity, nationality. Includes slurs like "chink", "nigger", "paki", "retard", "towelhead", comments about protected characteristics.
- sexual_harassment: Unwanted sexual advances, inappropriate touching, sexual comments about body/appearance, requests for sexual favors.
- severe_harassment: Sustained pattern of hostile, intimidating behavior creating toxic environment.
- bullying: Repeated verbal abuse, public humiliation, deliberate undermining, mocking.

**VIOLENCE & THREATS:**
- workplace_violence: PHYSICAL acts - hitting, punching, pushing, grabbing that causes harm, assault, physical altercations.
- threat_of_violence: Verbal/written threats of physical harm - "I'll hurt you", "watch your back", "waiting in parking lot".
- coercive_threat: Using power to force compliance - "do X or I'll fire you", "do X or you'll fail", conditional threats.

**FINANCIAL & ETHICAL MISCONDUCT:**
- fraud: Fake expense reports, embezzlement, stealing money, falsified financial records, fake invoices.
- corruption: Kickbacks, steering contracts to family/friends, bribery, conflicts of interest in procurement/hiring.
- ethics_violation: Told to lie to customers, falsify reports, cover up problems, misrepresent products, deceptive practices.

**SAFETY & SECURITY:**
- safety_hazard: Blocked exits, faulty equipment, fire risks, dangerous conditions, OSHA violations.
- physical_safety_concern: Personal injury at work, hurt back, fell from ladder, unsafe working conditions causing harm.
- data_breach: Customer data exposed, passwords leaked, hacking incidents, unauthorized data access.
- security_incident: Suspicious USB drives, malware, unauthorized system access, potential cyber threats.

**QUID PRO QUO & RETALIATION:**
- quid_pro_quo: Exchanging favors for advancement - "do X for promotion", "meet after hours for grade", implying benefits for personal favors.
- retaliation: Punishment for reporting concerns - excluded after complaint, demoted after HR report, "you'll regret going to HR".

**SUBSTANCE ABUSE:**
- substance_abuse_at_work: Drunk, high, intoxicated at work, using drugs on premises, impaired while working.

**MENTAL HEALTH:**
- mental_health_crisis: CRITICAL - suicidal thoughts, wanting to die, "end it all", self-harm ideation. ALWAYS severity: critical.
- mental_wellbeing_concern: Depression, anxiety, overwhelming stress, can't cope, burnout symptoms.

**WORKPLACE ISSUES:**
- pattern_of_unfair_treatment: Being singled out for different rules - "only I have to do X", "everyone else gets Y but not me". NOT about third parties or contracts.
- workload_burnout_risk: Extreme hours, constant overtime, unsustainable workload, denied help.
- management_concern: Poor leadership, arbitrary decisions, lack of transparency.
- interpersonal_conflict: Arguments with colleagues affecting work.
- professional_misconduct: Misusing company resources, running personal business at work.

**LANGUAGE:**
- profanity: Explicit swear words - fuck, shit, damn, bitch, ass, bastard, crap. Must contain ACTUAL profanity.
- inappropriate_language: Crude jokes, offensive non-sexual comments.
- suggestive_language: Sexual innuendo - "that's what she said", winking, double entendres.

---

CRITICAL CLASSIFICATION RULES:

1. **fraud** = MONEY/FINANCIAL deception (fake expenses, embezzlement, stealing)
2. **corruption** = CONFLICTS OF INTEREST (kickbacks, contracts to family, bribery)
3. **ethics_violation** = TOLD TO BE DISHONEST (lie to customers, falsify reports, cover-ups)
4. **workplace_violence** = PHYSICAL ACTS ONLY (hitting, assault). NOT lying or unethical behavior.
5. **physical_safety_concern** = BODILY INJURY/UNSAFE CONDITIONS. NOT financial issues.
6. **pattern_of_unfair_treatment** = PERSONAL treatment ("I am singled out"). NOT about third parties.
7. **profanity** = Must contain ACTUAL swear words. "buggy software" is NOT profanity.
8. **mental_health_crisis** = ALWAYS severity: critical

---

SEVERITY GUIDE:
- critical: Immediate danger - violence, suicide risk, ongoing assault
- high: Serious violations - discrimination, harassment, fraud, threats, data breach, corruption
- moderate: Concerning - unfair treatment, substance abuse, mental wellbeing, coercion
- low: Minor - profanity, suggestive language, interpersonal conflicts

Analyze the comment and return ONLY valid JSON with this structure:
{{"has_alerts": bool, "alerts": [{{"excerpt": str, "reasoning": str, "alert_type": str, "severity": str}}], "non_alert_classification": str|null, "non_alert_reasoning": str|null}}"""


FRANKENSTEIN_GENERATION_PROMPT = """You are a synthetic data generator. Your task is to create a novel, coherent comment by combining excerpts from real workplace comments.

TARGET SPECIFICATION:
- Target labels: {target_labels}
- Comment type: {comment_type}

CANDIDATE EXCERPTS (from real comments with these labels):
{candidate_excerpts}

---

INSTRUCTIONS:

1. Select 2-4 excerpts from the candidates that can be naturally combined
2. Stitch them into a single coherent comment that reads as if written by one person
3. Preserve the idiosyncratic features of the source excerpts (typos, punctuation style, tone, abbreviations, etc.)
4. Add minimal connective tissue (transitions, pronouns) to make it flow naturally
5. Do NOT add new alertable content beyond what's in the excerpts
6. Do NOT sanitize, formalize, or "clean up" the language
7. The final comment should trigger the target labels when analyzed

OUTPUT FORMAT:
Return ONLY valid JSON:
{{"comment": "<your synthesized comment>", "source_excerpts": ["<excerpt1>", "<excerpt2>", ...], "target_labels": {target_labels_json}}}"""


STYLE_TRANSFER_PROMPT = """You are given two texts:
1. A reference text that defines the target **stylistic attributes** (tone, structure, phrasing, vocabulary, elaboration, grammatical tendencies, punctuation tendencies, whitespaces, indentation, extraneous punctuation, abbreviations, emojis, shoutouts, capitalization, misspelling, redundancies, etc.)
2. An input comment that defines the **content** to preserve.

Your task is to rewrite the input comment to mimic the **style** of the reference text **without adding, inferring, or expanding** any meaning beyond what is explicitly in the input.

If the input is very short, keep it short — only modify surface features (e.g., capitalization, formality, punctuation) to align with the reference style.

Never insert new ideas, explanations, actors, or any other superfluous information.

Reference Text:
{reference_comment}

Input Comment:
{input_comment}

Respond only in JSON format:
{{"rewritten_comment": "<your rewritten version>"}}"""


NON_ALERT_GENERATION_PROMPT = """You are a synthetic data generator. Your task is to create a novel, coherent workplace comment that should NOT trigger any safety/compliance alerts.

TARGET SPECIFICATION:
- Classification: {target_classification}
- This comment should be classified as: {classification_description}

REFERENCE COMMENTS (real examples with this classification):
{reference_comments}

---

INSTRUCTIONS:

1. Study the reference comments to understand the tone, style, and content patterns
2. Create a NEW comment that fits the target classification
3. Preserve idiosyncratic features from the references (tone, punctuation style, abbreviations, etc.)
4. The comment should express similar themes but with different specific details
5. Do NOT include any alertable content (harassment, threats, discrimination, etc.)
6. Keep it realistic and natural-sounding

OUTPUT FORMAT:
Return ONLY valid JSON:
{{"comment": "<your synthesized comment>", "target_classification": "{target_classification}"}}"""


NON_ALERT_CLASSIFICATION_DESCRIPTIONS = {
    "performance_complaint": "Complaint about work quality, deadlines, or performance issues without harassment",
    "quality_complaint": "Feedback about product/service quality issues",
    "workload_feedback": "Comments about workload without indicating burnout or crisis",
    "process_improvement": "Suggestions for improving workflows or processes",
    "resource_request": "Requests for tools, equipment, training, or resources",
    "general_dissatisfaction": "General unhappiness without specific alertable concerns",
    "constructive_feedback": "Helpful suggestions for improvement",
    "positive_feedback": "Praise, appreciation, or positive observations",
    "neutral_comment": "Factual observations without positive or negative sentiment",
    "unclear": "Ambiguous content that doesn't fit other categories",
}


def get_extraction_prompt(text: str, prompt_template: str) -> str:
    return prompt_template.format(text=text)


def get_generation_prompt(
    target_labels: List[str],
    candidate_excerpts: Dict[str, List[str]],
    comment_type: str = "alert",
) -> str:
    excerpts_formatted = []
    for label, excerpts in candidate_excerpts.items():
        excerpts_formatted.append(f"\n[{label}]:")
        for i, excerpt in enumerate(excerpts, 1):
            excerpts_formatted.append(f'  {i}. "{excerpt}"')

    return FRANKENSTEIN_GENERATION_PROMPT.format(
        target_labels=target_labels,
        comment_type=comment_type,
        candidate_excerpts="\n".join(excerpts_formatted),
        target_labels_json=str(target_labels),
    )


def get_style_transfer_prompt(reference_comment: str, input_comment: str) -> str:
    return STYLE_TRANSFER_PROMPT.format(
        reference_comment=reference_comment,
        input_comment=input_comment,
    )


def get_non_alert_generation_prompt(
    target_classification: str,
    reference_comments: List[str],
) -> str:
    description = NON_ALERT_CLASSIFICATION_DESCRIPTIONS.get(
        target_classification, "General non-alert comment"
    )

    references_formatted = "\n".join(
        f'  {i}. "{comment}"' for i, comment in enumerate(reference_comments, 1)
    )

    return NON_ALERT_GENERATION_PROMPT.format(
        target_classification=target_classification,
        classification_description=description,
        reference_comments=references_formatted,
    )
