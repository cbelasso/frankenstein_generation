"""
Composition prompt for weaving excerpts into coherent text.
"""

from typing import List

from pipeline.tasks.generation.models import ExcerptSet


def composition_prompt(excerpt_set: ExcerptSet) -> str:
    """
    Generate a prompt for composing excerpts into a coherent text.

    Args:
        excerpt_set: ExcerptSet containing excerpts and their labels

    Returns:
        Prompt string for the composition task
    """
    # Format excerpts with their labels
    excerpt_list = "\n".join(
        f'{i + 1}. "{excerpt}" (exhibits: {label})'
        for i, (excerpt, label) in enumerate(
            zip(excerpt_set.excerpts, excerpt_set.source_labels)
        )
    )

    return f"""You are an expert at writing realistic employee feedback comments.

Your task is to compose a single, coherent employee comment that naturally incorporates the following excerpts. The final comment should read as if written by one person in a natural voice.

**EXCERPTS TO INCORPORATE:**
{excerpt_list}

---

**REQUIREMENTS:**

1. **Preserve the core meaning** of each excerpt — do not water down, remove, or significantly alter the key content
2. **Create natural flow** — use transitions, combine related ideas, vary sentence structure
3. **Maintain consistent voice** — the comment should sound like one person wrote it
4. **Keep it realistic** — this should read like genuine employee feedback (survey response, HR comment, etc.)
5. **Reasonable length** — aim for 2-5 sentences total, depending on the number of excerpts

**DO NOT:**
- Add new complaints or issues not present in the excerpts
- Remove or significantly soften the key elements from each excerpt
- Make it sound artificial or overly formal
- Add excessive context or backstory

---

**OUTPUT FORMAT:**
Return a JSON object with:
- **composed_text**: The final composed employee comment
- **coherence_notes**: Brief note on how you connected the excerpts (1 sentence)

---

**EXAMPLE:**

Excerpts:
1. "called me a slur" (exhibits: discrimination)
2. "need more training sessions" (exhibits: add_or_increase)

Good output:
{{
  "composed_text": "My supervisor called me a slur during the team meeting last week, and honestly the whole department could use more training sessions on workplace respect.",
  "coherence_notes": "Connected discrimination incident to training recommendation as a natural response."
}}

Bad output (loses key content):
{{
  "composed_text": "I think we need better training.",
  "coherence_notes": "Simplified the feedback."
}}

---

Now compose a coherent comment from the given excerpts. Return ONLY valid JSON."""
