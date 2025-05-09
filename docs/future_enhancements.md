## Detailed TODO / Future Enhancements:

1.  **Refine Different-Speaker Score Distribution Calculation:**
    *   **Goal:** Generate a more case-specific score distribution for the "different speakers" hypothesis.
    *   **Current Method:** Compares scores between different speakers *within* the reference population (e.g., VoxCeleb).
    *   **Proposed Method:** Calculate similarity scores by comparing the embeddings from the *actual probe audio chunks* against the embeddings of *all speakers* in the reference population dataset.
    *   **Benefit:** Provides a distribution tailored to the probe voice, showing how it compares against a known, diverse set of speakers. This can strengthen the interpretation when comparing the probe-reference score against this distribution. The same-speaker distribution calculation (comparing different utterances of the same speaker *within* the reference population) will remain unchanged initially.

2.  **Refine Same-Speaker Score Distribution Calculation (Conditional):**
    *   **Goal:** Generate a more case-specific score distribution for the "same speaker" hypothesis, *if sufficient data is available*.
    *   **Condition:** Requires having *multiple reference audio recordings* confirmed to be from the *same individual* as the primary reference sample in the case.
    *   **Proposed Method (if condition met):** Calculate similarity scores by comparing embeddings *between the different available reference recordings* of the claimed speaker.
    *   **Benefit:** Models the score variability specifically for that individual based on their own recordings, potentially improving accuracy over using the general population variability.
    *   **Fallback:** If only one reference recording exists, continue using the standard method (scores from same-speaker comparisons within the general reference population).

3.  **Improve the Reference Population:**
    *   **Goal:** Enhance the quality, diversity, size, and relevance of the background reference population dataset used for generating score distributions.
    *   **Potential Methods:**
        *   *Data Augmentation:* Apply noise, reverberation, or other transformations to existing reference audio to better match case audio conditions.
        *   *Cohort Selection:* Implement strategies to select a subset of the reference population that is acoustically closer (e.g., similar channel, noise characteristics) to the case audio.
        *   *Acquire/Integrate Diverse Data:* Incorporate additional speaker datasets reflecting a wider range of languages, accents, recording qualities, etc., relevant to forensic scenarios.
        *   *Quality Control:* Implement stricter filtering of the reference population to remove low-quality or potentially mislabeled samples.
    *   **Benefit:** A more robust and relevant reference population leads to more reliable score distributions and more confident verification conclusions. 