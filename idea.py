from __future__ import annotations

import random
from textwrap import dedent

from pydantic import BaseModel
from termcolor import cprint
from tqdm import trange

from utils import sample


class Idea:
    def generate_seed(self):
        raise NotImplementedError("Subclasses must implement this method")

    def generate_idea(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")

    def evaluate_idea(self, *args, **kwargs):
        raise NotImplementedError("Subclasses must implement this method")



class DALYSeed(BaseModel):
    approaches: list[str] # each approach should be fairly high level (max a few words).

class DALYApproach(BaseModel):
    title: str
    short_description: str
    full_description: str
    short_comings: list[str]
    daly_per_dolar: float # Estimate the amount of daly saved per dolar spend with this idea.


class IsCorrect(BaseModel):
    is_valid: bool # return true if the approach is feasible and doesn't have any fundamental issues, false if the approach has some fundamental flaws.
    reason: str # explain why the approach is valid or not.

class DALYIdea(Idea):

    def generate_seed(self) -> DALYSeed:
        prompt = "Please come up with a variety of general approaches (at least 50) that are considered towards improving people's DALY such as vaccinations, treatments, etc."
        return sample(prompt, DALYSeed)

    def generate_idea(self, approaches: list[str]) -> DALYApproach:
        prompt = dedent(f"""Come up with a fully fleshed out idea for improving 
                            people's DALY. The idea should be a single approach 
                            that is considered towards improving people's DALY:

                            Consider the following high level approaches, be creative:
                            {", ".join(approaches)}
                            """)
        return sample(prompt, DALYApproach)

    def evaluate_approach(self, approach: DALYApproach) -> IsCorrect:
        prompt = "Determine if the following approach is feasible, and doesn't have any fundamental issues. "
        prompt += f"Approach {approach.title}\n"
        prompt += f"Short description: {approach.short_description}\n"
        prompt += f"Full description: {approach.full_description}\n"
        prompt += f"Shortcomings: {approach.short_comings}\n"
        return sample(prompt, IsCorrect)


def main() -> None:
    """
    Small demo of Pydantic-validated structured sampling for idea generation.

    Note: this is *not* used by the math scaling experiments in `main.py`.
    """
    idea = DALYIdea()
    seeds = idea.generate_seed().approaches
    cprint(f"Generated {len(seeds)} seed approaches.", "green")

    for i in trange(3, desc="Generating ideas"):
        random_approaches = random.sample(seeds, 3)
        approach = idea.generate_idea(random_approaches)
        verdict = idea.evaluate_approach(approach)
        cprint(f"\n=== Idea {i + 1} ===", "cyan")
        if hasattr(approach, "model_dump_json"):
            print(approach.model_dump_json(indent=2))  # pydantic v2
        else:
            print(approach.json(indent=2))  # pydantic v1
        cprint("\n--- Feasibility check ---", "cyan")
        if hasattr(verdict, "model_dump_json"):
            print(verdict.model_dump_json(indent=2))  # pydantic v2
        else:
            print(verdict.json(indent=2))  # pydantic v1

if __name__ == "__main__":
    main()