def format_math_prompt(problem):
    # The "Final Answer:" line is what parse.py looks for.
    return (
        "Solve the following problem. Show your reasoning.\n"
        "Return your final answer on a single line exactly as:\n"
        "Final Answer: <answer>\n\n"
        "Problem:\n"
        f"{problem}"
    )
