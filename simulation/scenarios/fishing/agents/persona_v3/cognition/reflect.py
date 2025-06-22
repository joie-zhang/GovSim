from simulation.persona.cognition import ReflectComponent
from simulation.persona.common import ChatObservation, PersonaIdentity
from simulation.utils import ModelWandbWrapper

from .reflect_prompts import (
    prompt_insight_and_evidence,
    prompt_memorize_from_conversation,
    prompt_planning_thought_on_conversation,
)


class FishingReflectComponent(ReflectComponent):

    def __init__(
        self,
        model: ModelWandbWrapper,
        model_framework: ModelWandbWrapper,
    ):
        super().__init__(model, model_framework)
        self.prompt_insight_and_evidence = prompt_insight_and_evidence
        self.prompt_planning_thought_on_conversation = (
            prompt_planning_thought_on_conversation
        )
        self.prompt_memorize_from_conversation = prompt_memorize_from_conversation

    def run(self, focal_points: list[str]):
        acc = []
        # Get all persona names for the prompts
        all_persona_names = [p.identity.name for p in self.persona.other_personas.values()] + [self.persona.identity.name]
        
        for focal_point in focal_points:
            retireved_memory = self.persona.retrieve.retrieve([focal_point], 10)

            insights = self.prompt_insight_and_evidence(
                self.model, self.persona.identity, retireved_memory, all_persona_names
            )
            for insight in insights:
                self.persona.store.store_thought(insight, self.persona.current_time)
                acc.append(insight)

    def reflect_on_convesation(self, conversation: list[tuple[str, str]]):
        # Get all persona names for the prompts
        all_persona_names = [p.identity.name for p in self.persona.other_personas.values()] + [self.persona.identity.name]
        
        planning = self.prompt_planning_thought_on_conversation(
            self.model, self.persona.identity, conversation, all_persona_names
        )  # TODO should be this be store in scratch for planning?
        self.persona.store.store_thought(planning, self.persona.current_time)
        memo = self.prompt_memorize_from_conversation(
            self.model, self.persona.identity, conversation, all_persona_names
        )
        self.persona.store.store_thought(memo, self.persona.current_time)
