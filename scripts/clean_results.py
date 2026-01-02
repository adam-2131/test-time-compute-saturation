import os
import re


TARGET_HEADER = (
    "run_id,date,model,dataset,method,compute_multiplier,accuracy,"
    "total_decoded_tokens,num_problems,notes"
)

OLD_HEADER_NO_RUN_ID = (
    "date,model,dataset,method,compute_multiplier,accuracy,"
    "total_decoded_tokens,num_problems,notes"
)

DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _normalize_fields(fields: list[str]) -> tuple[str, str, str, str, str, str, str, str, str, str] | None:
    # Some lines may have an extra leading comma or missing run_id.
    # If a line has 11 fields with an empty first field, drop the empty field.
    if len(fields) == 11 and fields[0] == "":
        fields = fields[1:]

    # After normalization, we ONLY accept exactly 10 fields.
    if len(fields) != 10:
        return None

    run_id = fields[0]  # may be empty => backfilled later
    rest = fields[1:]
    date, model, dataset, method, compute_multiplier, accuracy, total_decoded_tokens, num_problems, notes = rest
    return (
        run_id,
        date,
        model,
        dataset,
        method,
        compute_multiplier,
        accuracy,
        total_decoded_tokens,
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
                total_decoded_tokens,
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
                int(num_problems)
            except ValueError:
                continue

            try:
                float(accuracy)
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
                    total_decoded_tokens,
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


