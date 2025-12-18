"""Generation prompts (Frankenstein, style transfer, non-alert)."""

from typing import Dict, List

from .base import PromptTemplate


FRANKENSTEIN_PROMPT = PromptTemplate(
    name="frankenstein_generation",
    description="Generate novel comments by combining excerpts",
    template="""You are a synthetic data generator. Your task is to create a novel, coherent comment by combining excerpts from real workplace comments.

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
)


STYLE_TRANSFER_PROMPT = PromptTemplate(
    name="style_transfer",
    description="Rewrite content to match reference style",
    template="""You are given two texts:
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
)


NON_ALERT_PROMPT = PromptTemplate(
    name="non_alert_generation",
    description="Generate non-alert comments for a classification",
    template="""You are a synthetic data generator. Your task is to create a novel, coherent workplace comment that should NOT trigger any safety/compliance alerts.

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
)


# Classification descriptions for non-alert generation
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


def format_frankenstein_prompt(
    target_labels: List[str],
    candidate_excerpts: Dict[str, List[str]],
    comment_type: str = "alert",
) -> str:
    """Format the Frankenstein generation prompt."""
    excerpts_formatted = []
    for label, excerpts in candidate_excerpts.items():
        excerpts_formatted.append(f"\n[{label}]:")
        for i, excerpt in enumerate(excerpts, 1):
            excerpts_formatted.append(f'  {i}. "{excerpt}"')
    
    return FRANKENSTEIN_PROMPT.format(
        target_labels=target_labels,
        comment_type=comment_type,
        candidate_excerpts="\n".join(excerpts_formatted),
        target_labels_json=str(target_labels),
    )


def format_style_transfer_prompt(reference_comment: str, input_comment: str) -> str:
    """Format the style transfer prompt."""
    return STYLE_TRANSFER_PROMPT.format(
        reference_comment=reference_comment,
        input_comment=input_comment,
    )


def format_non_alert_prompt(
    target_classification: str,
    reference_comments: List[str],
) -> str:
    """Format the non-alert generation prompt."""
    description = NON_ALERT_CLASSIFICATION_DESCRIPTIONS.get(
        target_classification,
        "General non-alert comment"
    )
    
    references_formatted = "\n".join(
        f'  {i}. "{comment}"'
        for i, comment in enumerate(reference_comments, 1)
    )
    
    return NON_ALERT_PROMPT.format(
        target_classification=target_classification,
        classification_description=description,
        reference_comments=references_formatted,
    )
