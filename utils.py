from pydantic import BaseModel
import inspect
import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential


def sample_str(prompt: str) -> str:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content

from tenacity import RetryError

def sample(prompt: str, obj: BaseModel) -> BaseModel:
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=15))
    def _run_sample():
        nonlocal prompt, obj
        prompt += "\n\nPlease provide answer as json string between ```json and ```:\n" + inspect.getsource(obj)
        prompt += "\n\nnothing else. just the json string."
        response = sample_str(prompt)
        response = response.replace("```json", "").replace("```", "")
        return obj.model_validate_json(response)

    try:
        return _run_sample()
    except RetryError as e:
        import ipdb; ipdb.set_trace()
        raise e


class City(BaseModel):
    name: str # The name of the city
    country: str # The country of the city
    population: int # The population of the city

def main():
    obj = sample("What is the capital of France?", City)
    import ipdb
    ipdb.set_trace()
    pass

if __name__ == "__main__":
    main()