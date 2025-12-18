"""Alerts extraction prompts."""

from .base import PromptTemplate


EXTRACTION_PROMPT = PromptTemplate(
    name="alerts_extraction",
    description="Extract workplace safety and compliance alerts from text",
    template="""You are an expert workplace safety and compliance analyzer. Analyze this employee comment for alerts requiring HR, management, or compliance attention.

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
)


# Convenience function
def format_extraction_prompt(text: str) -> str:
    return EXTRACTION_PROMPT.format(text=text)
