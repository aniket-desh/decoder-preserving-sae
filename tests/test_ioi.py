import pytest
import torch

from dpsae.ioi import IOIExample, generate_ioi_examples, tokenize_ioi_examples


class ToyTokenizer:
    pad_token = "<pad>"
    eos_token = "<pad>"
    eos_token_id = 0
    pad_token_id = 0

    def __init__(self) -> None:
        self.vocabulary = {"<pad>": 0}

    def encode(self, text, add_special_tokens=False):
        return [self._id(token) for token in text.strip().replace(".", " .").split()]

    def _id(self, token):
        if token not in self.vocabulary:
            self.vocabulary[token] = len(self.vocabulary)
        return self.vocabulary[token]

    def __call__(self, texts, padding=True, return_tensors="pt"):
        rows = [self.encode(text) for text in texts]
        width = max(map(len, rows))
        ids = torch.zeros(len(rows), width, dtype=torch.long)
        mask = torch.zeros_like(ids)
        for index, row in enumerate(rows):
            ids[index, : len(row)] = torch.tensor(row)
            mask[index, : len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}


def test_generator_balances_order_and_builds_counterfactuals() -> None:
    examples = generate_ioi_examples(
        count=12,
        names=("Alice", "Bob", "Carol", "David"),
        template_families=(0, 1, 2),
        seed=3,
    )
    assert [example.order for example in examples].count("ABBA") == 6
    assert [example.order for example in examples].count("BABA") == 6
    for example in examples:
        assert not example.prompt.endswith(example.io_name)
        assert example.subject_name in example.prompt
        assert example.third_name in example.abc_prompt
        assert example.io_name in example.swapped_prompt


def test_token_positions_distinguish_duplicate_and_abc_controls() -> None:
    tokenizer = ToyTokenizer()
    examples = generate_ioi_examples(
        count=4,
        names=("Alice", "Bob", "Carol", "David"),
        template_families=(0,),
        seed=4,
    )
    standard = tokenize_ioi_examples(examples, tokenizer)
    abc = tokenize_ioi_examples(examples, tokenizer, variant="abc_prompt")
    for index, example in enumerate(examples):
        row = standard["input_ids"][index]
        subject_id = tokenizer.encode(" " + example.subject_name)[0]
        assert row[standard["s1_position"][index]] == subject_id
        assert row[standard["s2_position"][index]] == subject_id
        third_id = tokenizer.encode(" " + example.third_name)[0]
        assert abc["input_ids"][index, abc["s2_position"][index]] == third_id
        assert standard["end_position"][index] == standard["attention_mask"][index].sum() - 1


def test_invalid_prompt_variant_is_rejected() -> None:
    tokenizer = ToyTokenizer()
    example = IOIExample("x", "x", "x", "A", "B", "C", "p", "o", 0, "ABBA")
    with pytest.raises(ValueError, match="variant"):
        tokenize_ioi_examples([example], tokenizer, variant="bad")
