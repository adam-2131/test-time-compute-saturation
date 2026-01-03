import os
import re


TARGET_HEADER = (
    "run_id,date,model,dataset,method,compute_multiplier,accuracy,"
    "pass_at_k_accuracy,majority_vote_accuracy,"
    "total_decoded_tokens,total_completion_tokens_used,total_tokens_used,"
    "num_problems,notes"
)

OLD_HEADER_NO_RUN_ID = (
    "date,model,dataset,method,compute_multiplier,accuracy,"
    "pass_at_k_accuracy,majority_vote_accuracy,"
    "total_decoded_tokens,total_completion_tokens_used,total_tokens_used,"
    "num_problems,notes"
)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _normalize_fields(fields: list[str]) -> tuple[str, str, str, str, str, str, str, str, str, str, str, str, str, str] | None:
    # Some lines may have an extra leading comma or missing run_id.
    # If a line has 11 fields with an empty first field, drop the empty field.
    if len(fields) in {11, 13} and fields[0] == "":
        fields = fields[1:]

    # Legacy rows have 10 fields (no extra accuracies, no total_* columns).
    # Older rows have 12 fields (no extra accuracies, but include total_* columns).
    # New rows have 14 fields.
    if len(fields) == 10:
        # run_id,date,model,dataset,method,compute_multiplier,accuracy,total_decoded_tokens,num_problems,notes
        # -> insert blanks for pass@k/majority after accuracy and for total_* after total_decoded_tokens
        fields = fields[:7] + ["", ""] + [fields[7]] + ["", ""] + fields[8:]
    elif len(fields) == 12:
        # run_id,date,model,dataset,method,compute_multiplier,accuracy,total_decoded_tokens,total_completion_tokens_used,total_tokens_used,num_problems,notes
        # -> insert blanks for pass@k/majority after accuracy
        fields = fields[:7] + ["", ""] + fields[7:]
    if len(fields) != 14:
        return None

    run_id = fields[0]  # may be empty => backfilled later
    rest = fields[1:]
    (
        date,
        model,
        dataset,
        method,
        compute_multiplier,
        accuracy,
        pass_at_k_accuracy,
        majority_vote_accuracy,
        total_decoded_tokens,
        total_completion_tokens_used,
        total_tokens_used,
        num_problems,
        notes,
    ) = rest
    return (
        run_id,
        date,
        model,
        dataset,
        method,
        compute_multiplier,
        accuracy,
        pass_at_k_accuracy,
        majority_vote_accuracy,
        total_decoded_tokens,
        total_completion_tokens_used,
        total_tokens_used,
        num_problems,
        notes,
    )


def main() -> None:
    in_path = os.path.join("results", "scaling_results_raw.csv")
    out_path = os.path.join("results", "scaling_results_clean.csv")

    os.makedirs("results", exist_ok=True)

    cleaned_rows: list[list[str]] = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line_number, line in enumerate(f, start=1):
            line = line.rstrip("\n")
            if not line:
                continue

            # Parse each line by splitting on commas.
            fields = [x.strip() for x in line.split(",")]

            # Skip header-ish lines.
            if line == TARGET_HEADER or line == OLD_HEADER_NO_RUN_ID:
                continue
            if fields and fields[0] in {"run_id", "date"}:
                continue

            normalized = _normalize_fields(fields)
            if normalized is None:
                continue

            (
                run_id,
                date,
                model,
                dataset,
                method,
                compute_multiplier,
                accuracy,
                pass_at_k_accuracy,
                majority_vote_accuracy,
                total_decoded_tokens,
                total_completion_tokens_used,
                total_tokens_used,
                num_problems,
                notes,
            ) = normalized

            # If first field is empty, treat run_id as missing.
            if not run_id:
                run_id = f"legacy_{line_number}"

            # Drop junk rows.
            if model in {"m", "m2"}:
                continue
            if dataset in {"d", "d2"}:
                continue
            if not DATE_RE.match(date):
                continue

            try:
                int(compute_multiplier)
                int(total_decoded_tokens)
                if total_completion_tokens_used:
                    int(total_completion_tokens_used)
                if total_tokens_used:
                    int(total_tokens_used)
                int(num_problems)
            except ValueError:
                continue

            try:
                float(accuracy)
                if pass_at_k_accuracy:
                    float(pass_at_k_accuracy)
                if majority_vote_accuracy:
                    float(majority_vote_accuracy)
            except ValueError:
                continue

            cleaned_rows.append(
                [
                    run_id,
                    date,
                    model,
                    dataset,
                    method,
                    compute_multiplier,
                    accuracy,
                    pass_at_k_accuracy,
                    majority_vote_accuracy,
                    total_decoded_tokens,
                    total_completion_tokens_used,
                    total_tokens_used,
                    num_problems,
                    notes,
                ]
            )

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        f.write(TARGET_HEADER + "\n")
        for row in cleaned_rows:
            f.write(",".join(row) + "\n")

    print(f"wrote {len(cleaned_rows)} cleaned rows")


if __name__ == "__main__":
    main()


