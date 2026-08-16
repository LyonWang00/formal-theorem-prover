import pytest

from lean_prover.lean_training.sft_pipeline.config import SFTTrainConfig
from lean_prover.lean_training.sft_pipeline.trainer import (
    assert_supervised_eos_contract,
    inspect_supervised_eos_contract,
)


class FakeTokenizer:
    eos_token = "<eos>"
    eos_token_id = 2
    pad_token = "<pad>"
    pad_token_id = 0

    def __call__(self, text, add_special_tokens=True):
        del add_special_tokens
        ids = []
        cursor = 0
        while cursor < len(text):
            if text.startswith(self.eos_token, cursor):
                ids.append(self.eos_token_id)
                cursor += len(self.eos_token)
            else:
                ids.append(10 + ord(text[cursor]))
                cursor += 1
        return {"input_ids": ids}


def row(completion="proof<eos>"):
    return {"prompt": "statement:", "completion": completion}


def effective(tokenizer, record, labels):
    input_ids = tokenizer(
        record["prompt"] + record["completion"]
    )["input_ids"]
    return lambda _index, _row: (input_ids, labels)


def expected_labels(tokenizer, record):
    prompt_ids = tokenizer(record["prompt"])["input_ids"]
    full_ids = tokenizer(
        record["prompt"] + record["completion"]
    )["input_ids"]
    return [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]


def test_supervised_eos_happy_path_and_default_required():
    tokenizer = FakeTokenizer()
    report = assert_supervised_eos_contract([row()], tokenizer)
    assert report["passed"] is True
    assert report["records_with_supervised_eos"] == 1
    assert report["records_whose_last_valid_label_is_eos"] == 1
    assert SFTTrainConfig.__dataclass_fields__["require_supervised_eos"].default is True


def test_missing_eos_fails():
    with pytest.raises(RuntimeError, match="EOS_GATE_FAILED"):
        assert_supervised_eos_contract([row("proof")], FakeTokenizer())


def test_masked_eos_fails():
    tokenizer = FakeTokenizer()
    record = row()
    labels = expected_labels(tokenizer, record)
    labels[-1] = -100
    report = inspect_supervised_eos_contract(
        [record], tokenizer, labels_provider=effective(tokenizer, record, labels)
    )
    assert report["records_with_eos_masked_as_minus_100"] == 1
    assert report["passed"] is False


def test_zero_labels_fail():
    tokenizer = FakeTokenizer()
    record = row()
    labels = [-100] * len(
        tokenizer(record["prompt"] + record["completion"])["input_ids"]
    )
    report = inspect_supervised_eos_contract(
        [record], tokenizer, labels_provider=effective(tokenizer, record, labels)
    )
    assert report["zero_label_records"] == 1
    assert report["passed"] is False


def test_empty_completion_fails():
    with pytest.raises(RuntimeError, match="EOS_GATE_FAILED"):
        assert_supervised_eos_contract([row("<eos>")], FakeTokenizer())


def test_semantic_truncation_fails_without_replacing_proof_tail():
    with pytest.raises(RuntimeError, match="semantic_truncation"):
        assert_supervised_eos_contract(
            [row("long-proof<eos>")], FakeTokenizer(), max_seq_length=4
        )


def test_padding_preserves_distinct_eos_and_last_valid_label():
    tokenizer = FakeTokenizer()
    record = row()
    labels = expected_labels(tokenizer, record)
    input_ids = tokenizer(
        record["prompt"] + record["completion"]
    )["input_ids"] + [tokenizer.pad_token_id, tokenizer.pad_token_id]
    labels = labels + [-100, -100]
    report = assert_supervised_eos_contract(
        [record],
        tokenizer,
        labels_provider=lambda _index, _row: (input_ids, labels),
    )
    assert report["records_with_eos_masked_as_minus_100"] == 0
    assert report["records_whose_last_valid_label_is_eos"] == 1


def test_last_valid_label_must_equal_eos():
    tokenizer = FakeTokenizer()
    record = row()
    labels = expected_labels(tokenizer, record)
    labels[-1] = labels[-2]
    with pytest.raises(RuntimeError, match="last_valid_label_is_not_eos"):
        assert_supervised_eos_contract(
            [record], tokenizer, labels_provider=effective(tokenizer, record, labels)
        )
