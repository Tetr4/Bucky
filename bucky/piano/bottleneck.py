from typing import Optional

from bucky.piano.perception import PerceptionInput, PerceptionType


class InformationBottleneck:
    def __init__(self, output_limit: int):
        self._output_limit = output_limit
        self._module_weights: dict[PerceptionType, float] = {
            PerceptionType.USER_INPUT: 2.0,
            PerceptionType.VISION: 0.5
        }

    def filter_and_compress(self, shared_state_inputs: list[PerceptionInput], current_goal: str = "") -> list[PerceptionInput]:
        scored_inputs: list[tuple[float, PerceptionInput]] = []
        for item in shared_state_inputs:
            # A. Heuristic scoring (rule-based weighting)
            weight = self._module_weights.get(item.type, 1.0)
            final_score = item.base_priority * weight

            # B. Contextual relevance (in a real system via vector embeddings)
            # Simplified: if the event matches the current goal, boost priority
            if current_goal and any(word in item.content.lower() for word in current_goal.lower().split()):
                final_score += 3.0  # relevance bonus

            scored_inputs.append((final_score, item))

        # C. The actual bottleneck
        # Sort by highest score (most important events first)
        scored_inputs.sort(key=lambda x: x[0], reverse=True)

        # Let only the most important element through (bottleneck limitation)
        return [item for score, item in scored_inputs[:self._output_limit]]
