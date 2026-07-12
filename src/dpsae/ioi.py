"""Canonical indirect-object-identification prompt generation.

The prompt families and entity lists are adapted from Wang et al. (2023),
"Interpretability in the Wild", and its public Easy-Transformer reference
implementation:
https://github.com/redwoodresearch/Easy-Transformer/blob/main/easy_transformer/ioi_dataset.py

This module keeps only deterministic data generation and token-position
bookkeeping. It deliberately does not depend on Easy-Transformer.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import torch
from torch import Tensor


CANONICAL_NAMES = (
    "Michael", "Christopher", "Jessica", "Matthew", "Ashley", "Jennifer",
    "Joshua", "Amanda", "Daniel", "David", "James", "Robert", "John",
    "Joseph", "Andrew", "Ryan", "Brandon", "Jason", "Justin", "Sarah",
    "William", "Jonathan", "Stephanie", "Brian", "Nicole", "Nicholas",
    "Anthony", "Heather", "Eric", "Elizabeth", "Adam", "Megan", "Melissa",
    "Kevin", "Steven", "Thomas", "Timothy", "Christina", "Kyle", "Rachel",
    "Laura", "Lauren", "Amber", "Brittany", "Danielle", "Richard",
    "Kimberly", "Jeffrey", "Amy", "Crystal", "Michelle", "Tiffany",
    "Jeremy", "Benjamin", "Mark", "Emily", "Aaron", "Charles", "Rebecca",
    "Jacob", "Stephen", "Patrick", "Sean", "Erin", "Jamie", "Kelly",
    "Samantha", "Nathan", "Sara", "Dustin", "Paul", "Angela", "Tyler",
    "Scott", "Katherine", "Andrea", "Gregory", "Erica", "Mary", "Travis",
    "Lisa", "Kenneth", "Bryan", "Lindsey", "Kristen", "Jose", "Alexander",
    "Jesse", "Katie", "Lindsay", "Shannon", "Vanessa", "Courtney",
    "Christine", "Alicia", "Cody", "Allison", "Bradley", "Samuel",
)

PLACES = ("store", "garden", "restaurant", "school", "hospital", "office", "house", "station")
OBJECTS = ("ring", "kiss", "bone", "basketball", "computer", "necklace", "drink", "snack")

# BABA form: S1 occurs before IO in the first clause. S2 is the repeated
# subject in the main clause. ABBA swaps only S1 and IO in the first clause.
CANONICAL_BABA_TEMPLATES = (
    "Then, {S1} and {IO} went to the {PLACE}. {S2} gave a {OBJECT} to {IO}",
    "Then, {S1} and {IO} had a lot of fun at the {PLACE}. {S2} gave a {OBJECT} to {IO}",
    "Then, {S1} and {IO} were working at the {PLACE}. {S2} decided to give a {OBJECT} to {IO}",
    "Then, {S1} and {IO} were thinking about going to the {PLACE}. "
    "{S2} wanted to give a {OBJECT} to {IO}",
    "Then, {S1} and {IO} had a long argument, and afterwards {S2} said to {IO}",
    "After {S1} and {IO} went to the {PLACE}, {S2} gave a {OBJECT} to {IO}",
    "When {S1} and {IO} got a {OBJECT} at the {PLACE}, {S2} decided to give it to {IO}",
    "When {S1} and {IO} got a {OBJECT} at the {PLACE}, {S2} decided to give the {OBJECT} to {IO}",
    "While {S1} and {IO} were working at the {PLACE}, {S2} gave a {OBJECT} to {IO}",
    "While {S1} and {IO} were commuting to the {PLACE}, {S2} gave a {OBJECT} to {IO}",
    "After lunch, {S1} and {IO} went to the {PLACE}. {S2} gave a {OBJECT} to {IO}",
    "Afterwards, {S1} and {IO} went to the {PLACE}. {S2} gave a {OBJECT} to {IO}",
    "Then, {S1} and {IO} had a long argument. Afterwards {S2} said to {IO}",
    "The {PLACE} {S1} and {IO} went to had a {OBJECT}. {S2} gave it to {IO}",
    "Friends {S1} and {IO} found a {OBJECT} at the {PLACE}. {S2} gave it to {IO}",
)

TEMPLATE_SPLITS = {
    "discovery": tuple(range(0, 8)),
    "validation": tuple(range(8, 11)),
    "test": tuple(range(11, 15)),
}


@dataclass(frozen=True)
class IOIExample:
    prompt: str
    abc_prompt: str
    swapped_prompt: str
    io_name: str
    subject_name: str
    third_name: str
    place: str
    object_name: str
    template_family: int
    order: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _abba_template(template: str) -> str:
    sentinel = "{FIRST_SUBJECT}"
    return template.replace("{S1}", sentinel, 1).replace(
        "{IO}", "{S1}", 1
    ).replace(sentinel, "{IO}", 1)


def _render_prefix(
    template: str,
    *,
    io_name: str,
    subject_name: str,
    main_subject: str,
    place: str,
    object_name: str,
) -> str:
    full = template.format(
        IO=io_name,
        S1=subject_name,
        S2=main_subject,
        PLACE=place,
        OBJECT=object_name,
    )
    answer_suffix = f" {io_name}"
    if not full.endswith(answer_suffix):
        raise ValueError("IOI template must end in the indirect-object answer")
    return full[: -len(answer_suffix)]


def generate_ioi_examples(
    *,
    count: int,
    names: Sequence[str],
    template_families: Sequence[int],
    seed: int,
) -> list[IOIExample]:
    """Generate balanced ABBA/BABA examples and paired controls."""

    if len(names) < 3:
        raise ValueError("at least three names are required")
    if not template_families:
        raise ValueError("at least one template family is required")
    rng = random.Random(seed)
    examples = []
    for index in range(count):
        family = template_families[(index // 2) % len(template_families)]
        order = "BABA" if index % 2 == 0 else "ABBA"
        template = CANONICAL_BABA_TEMPLATES[family]
        if order == "ABBA":
            template = _abba_template(template)
        io_name, subject_name, third_name = rng.sample(list(names), 3)
        place = rng.choice(PLACES)
        object_name = rng.choice(OBJECTS)
        prompt = _render_prefix(
            template,
            io_name=io_name,
            subject_name=subject_name,
            main_subject=subject_name,
            place=place,
            object_name=object_name,
        )
        abc_prompt = _render_prefix(
            template,
            io_name=io_name,
            subject_name=subject_name,
            main_subject=third_name,
            place=place,
            object_name=object_name,
        )
        swapped_prompt = _render_prefix(
            template,
            io_name=subject_name,
            subject_name=io_name,
            main_subject=io_name,
            place=place,
            object_name=object_name,
        )
        examples.append(
            IOIExample(
                prompt=prompt,
                abc_prompt=abc_prompt,
                swapped_prompt=swapped_prompt,
                io_name=io_name,
                subject_name=subject_name,
                third_name=third_name,
                place=place,
                object_name=object_name,
                template_family=family,
                order=order,
            )
        )
    return examples


def canonical_name_splits(tokenizer, *, seed: int = 0) -> dict[str, tuple[str, ...]]:
    """Return disjoint single-token GPT-2 name splits."""

    eligible = [
        name
        for name in CANONICAL_NAMES
        if len(tokenizer.encode(" " + name, add_special_tokens=False)) == 1
    ]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    first = len(eligible) // 2
    second = first + (len(eligible) - first) // 2
    return {
        "discovery": tuple(eligible[:first]),
        "validation": tuple(eligible[first:second]),
        "test": tuple(eligible[second:]),
    }


def tokenize_ioi_examples(
    examples: Sequence[IOIExample], tokenizer, *, variant: str = "prompt"
) -> dict[str, Tensor]:
    """Tokenize a prompt variant and locate IO, S1, S2, and END positions."""

    if variant not in {"prompt", "abc_prompt", "swapped_prompt"}:
        raise ValueError(f"unknown IOI prompt variant: {variant}")
    tokenizer.pad_token = tokenizer.eos_token
    prompts = [getattr(example, variant) for example in examples]
    encoded = tokenizer(prompts, padding=True, return_tensors="pt")
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    io_positions, s1_positions, s2_positions = [], [], []
    io_ids, subject_ids, third_ids = [], [], []

    for row, example in zip(input_ids, examples):
        if variant == "swapped_prompt":
            io_name, subject_name = example.subject_name, example.io_name
        else:
            io_name, subject_name = example.io_name, example.subject_name
        main_name = example.third_name if variant == "abc_prompt" else subject_name
        io_id = tokenizer.encode(" " + io_name, add_special_tokens=False)[0]
        subject_id = tokenizer.encode(" " + subject_name, add_special_tokens=False)[0]
        main_id = tokenizer.encode(" " + main_name, add_special_tokens=False)[0]
        io_matches = (row == io_id).nonzero(as_tuple=False).flatten().tolist()
        subject_matches = (row == subject_id).nonzero(as_tuple=False).flatten().tolist()
        main_matches = (row == main_id).nonzero(as_tuple=False).flatten().tolist()
        if len(io_matches) != 1:
            raise ValueError(f"expected one IO occurrence, found {io_matches}")
        if variant == "abc_prompt":
            if len(subject_matches) != 1 or len(main_matches) != 1:
                raise ValueError("ABC prompt must contain S1 and the third name once")
            s1_position, s2_position = subject_matches[0], main_matches[0]
        else:
            if len(subject_matches) != 2:
                raise ValueError(f"expected two subject occurrences, found {subject_matches}")
            s1_position, s2_position = subject_matches
        io_positions.append(io_matches[0])
        s1_positions.append(s1_position)
        s2_positions.append(s2_position)
        io_ids.append(io_id)
        subject_ids.append(subject_id)
        third_ids.append(main_id)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "io_position": torch.tensor(io_positions),
        "s1_position": torch.tensor(s1_positions),
        "s2_position": torch.tensor(s2_positions),
        "end_position": attention_mask.sum(dim=1) - 1,
        "io_token_id": torch.tensor(io_ids),
        "subject_token_id": torch.tensor(subject_ids),
        "main_subject_token_id": torch.tensor(third_ids),
    }
