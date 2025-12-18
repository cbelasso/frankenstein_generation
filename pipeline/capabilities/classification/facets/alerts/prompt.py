def alert_detection_prompt(text: str) -> str:
    """Generate optimized prompt for 8B model with clear category definitions."""
    return f"""You are an expert workplace safety and compliance analyzer. Analyze this employee comment for alerts requiring HR, management, or compliance attention.

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

EXAMPLES:

Comment: "My manager has been submitting fake expense reports for trips he never took"
{{"has_alerts": true, "alerts": [{{"excerpt": "submitting fake expense reports for trips he never took", "reasoning": "Financial fraud through falsified expense claims", "alert_type": "fraud", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "The procurement guy is steering all contracts to his brother's company"
{{"has_alerts": true, "alerts": [{{"excerpt": "steering all contracts to his brother's company", "reasoning": "Corruption through conflict of interest in procurement", "alert_type": "corruption", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "We're being told to lie to customers about the product capabilities to close sales"
{{"has_alerts": true, "alerts": [{{"excerpt": "told to lie to customers about the product capabilities", "reasoning": "Ethics violation - instructed to deceive customers", "alert_type": "ethics_violation", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "The CFO is taking kickbacks from vendors"
{{"has_alerts": true, "alerts": [{{"excerpt": "taking kickbacks from vendors", "reasoning": "Corruption through accepting bribes from vendors", "alert_type": "corruption", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "They're falsifying the safety inspection reports"
{{"has_alerts": true, "alerts": [{{"excerpt": "falsifying the safety inspection reports", "reasoning": "Ethics violation - falsifying official safety documents", "alert_type": "ethics_violation", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "He punched me in the face during the meeting"
{{"has_alerts": true, "alerts": [{{"excerpt": "punched me in the face", "reasoning": "Physical violence - assault in workplace", "alert_type": "workplace_violence", "severity": "critical"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "I hurt my back lifting heavy boxes without proper equipment"
{{"has_alerts": true, "alerts": [{{"excerpt": "hurt my back lifting heavy boxes without proper equipment", "reasoning": "Physical injury from unsafe working conditions", "alert_type": "physical_safety_concern", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "I'm the only one who has to get approval for expenses under $50"
{{"has_alerts": true, "alerts": [{{"excerpt": "I'm the only one who has to get approval for expenses under $50", "reasoning": "Employee singled out for different rules than colleagues", "alert_type": "pattern_of_unfair_treatment", "severity": "moderate"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "He called me a fucking chink in front of everyone"
{{"has_alerts": true, "alerts": [{{"excerpt": "called me a fucking chink", "reasoning": "Racial slur and ethnic discrimination", "alert_type": "discrimination", "severity": "high"}}, {{"excerpt": "fucking", "reasoning": "Contains profanity", "alert_type": "profanity", "severity": "low"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "That's what she said! You know what I mean ;) wink wink"
{{"has_alerts": true, "alerts": [{{"excerpt": "That's what she said! You know what I mean ;) wink wink", "reasoning": "Sexual innuendo and suggestive language", "alert_type": "suggestive_language", "severity": "low"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "This fucking project is a nightmare"
{{"has_alerts": true, "alerts": [{{"excerpt": "fucking project", "reasoning": "Contains explicit profanity", "alert_type": "profanity", "severity": "low"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "I don't see the point anymore, I want to end it all"
{{"has_alerts": true, "alerts": [{{"excerpt": "don't see the point anymore, I want to end it all", "reasoning": "Suicidal ideation requiring immediate intervention", "alert_type": "mental_health_crisis", "severity": "critical"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "She implied a promotion might happen faster if we spent private time together"
{{"has_alerts": true, "alerts": [{{"excerpt": "promotion might happen faster if we spent private time together", "reasoning": "Quid pro quo - implying advancement in exchange for personal relationship", "alert_type": "quid_pro_quo", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "You'll regret going to HR about this. Watch your back."
{{"has_alerts": true, "alerts": [{{"excerpt": "You'll regret going to HR about this. Watch your back", "reasoning": "Threatening retaliation for reporting to HR", "alert_type": "retaliation", "severity": "high"}}], "non_alert_classification": null, "non_alert_reasoning": null}}

Comment: "The new software update is buggy and crashes frequently"
{{"has_alerts": false, "alerts": [], "non_alert_classification": "quality_complaint", "non_alert_reasoning": "Technical feedback about software quality, no profanity or serious concerns"}}

Comment: "The project timeline seems aggressive given our current resources"
{{"has_alerts": false, "alerts": [], "non_alert_classification": "workload_feedback", "non_alert_reasoning": "Feedback about project timeline and resourcing"}}

Comment: "It would be more efficient if we had a shared calendar for meeting room bookings"
{{"has_alerts": false, "alerts": [], "non_alert_classification": "process_improvement", "non_alert_reasoning": "Constructive suggestion for process improvement"}}

Comment: "We really need a second printer on this floor"
{{"has_alerts": false, "alerts": [], "non_alert_classification": "resource_request", "non_alert_reasoning": "Request for additional office equipment"}}

Comment: "The training was fantastic! The instructor really knew their stuff"
{{"has_alerts": false, "alerts": [], "non_alert_classification": "positive_feedback", "non_alert_reasoning": "Positive feedback about training quality"}}

Comment: "My team lead never meets deadlines and it delays everyone else's work"
{{"has_alerts": false, "alerts": [], "non_alert_classification": "performance_complaint", "non_alert_reasoning": "Complaint about colleague's performance, not a serious violation"}}

---

SEVERITY GUIDE:
- critical: Immediate danger - violence, suicide risk, ongoing assault
- high: Serious violations - discrimination, harassment, fraud, threats, data breach, corruption
- moderate: Concerning - unfair treatment, substance abuse, mental wellbeing, coercion
- low: Minor - profanity, suggestive language, interpersonal conflicts

Analyze the comment and return ONLY valid JSON."""
