import json
import matplotlib.pyplot as plt
from collections import Counter


# Load evaluation questions
with open("eval_qs_with_indices.txt", "r", encoding="utf-8") as file:
    evaluation_questions = json.load(file)


# Fields to visualize
fields = [
    "question_type",
    "answer_type",
    "difficulty"
]


for field in fields:

    # Count each category
    values = [question[field] for question in evaluation_questions]
    counts = Counter(values)

    # Create bar chart
    plt.figure(figsize=(8, 5))
    plt.bar(counts.keys(), counts.values())

    plt.title(f"Distribution of {field.replace('_', ' ').title()}")
    plt.xlabel(field.replace("_", " ").title())
    plt.ylabel("Number of Questions")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()

    # Save chart
    plt.savefig(f"{field}_distribution.png", dpi=300)

    # Display chart
    plt.show()