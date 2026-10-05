"""Exercise real, offline CPU training and complete-state resume, not a Trainer mock."""

from __future__ import annotations

import copy
import json
import shutil
import string
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

MODEL_KEYS = ("llama_3_2_3b", "llama_3_1_8b", "qwen3_4b")
PROJECT_ROOT = Path(__file__).resolve().parents[1]


class InterruptedTraining(RuntimeError):
    """Simulated process failure after the recovery receipt is durable."""


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def snapshot(path):
    return {
        str(file.relative_to(path)): file.read_bytes() for file in path.rglob("*") if file.is_file()
    }


@pytest.fixture
def controlled_training(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    pytest.importorskip("peft")
    pytest.importorskip("accelerate")
    tokenizers = pytest.importorskip("tokenizers")
    from simulated_students import training
    from simulated_students.dataset import DatasetSplits

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("WORLD_SIZE", "1")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)

    config = yaml.safe_load((PROJECT_ROOT / "configs/train.yaml").read_text())
    config["training"].update(epochs=3, effective_batch_size=4, checkpoint_steps=1)
    config["lora"].update(rank=2, alpha=4, dropout=0.05)
    for model in config["models"].values():
        model.update(train_batch_size=2, validation_batch_size=2, gradient_accumulation_steps=2)
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    specials = ["<unk>", "<eos>", "<|finetune_right_pad_id|>"]
    vocab = {char: index for index, char in enumerate(specials + list(string.printable))}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<unk>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Split("", behavior="isolated")
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<|finetune_right_pad_id|>",
        model_max_length=1024,
        chat_template=(
            "{% for message in messages %}{{ message['role'] + ':' + message['content'] "
            "+ eos_token }}{% endfor %}"
            "{% if add_generation_prompt %}{{ 'assistant:' }}{% endif %}"
        ),
    )
    local_model = tmp_path / "tiny-llama"
    transformers.set_seed(1234)
    model = transformers.LlamaForCausalLM(
        transformers.LlamaConfig(
            vocab_size=len(tokenizer),
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=1024,
            bos_token_id=None,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            attention_dropout=0.0,
        )
    )
    model.save_pretrained(local_model)
    tokenizer.save_pretrained(local_model)
    real_model_loader = transformers.AutoModelForCausalLM.from_pretrained
    real_tokenizer_loader = transformers.AutoTokenizer.from_pretrained
    model_loads = []

    def load_model(model_id, **kwargs):
        assert model_id == "controlled-tiny"
        assert kwargs.pop("revision") == "fixture"
        assert kwargs["device_map"] is None
        assert kwargs["dtype"] == torch.float32
        model_loads.append(model_id)
        return real_model_loader(local_model, local_files_only=True, **kwargs)

    def load_tokenizer(model_id, **kwargs):
        assert model_id in {"controlled-tiny", config["data"]["filter_tokenizer_id"]}
        kwargs.pop("revision")
        return real_tokenizer_loader(local_model, local_files_only=True, **kwargs)

    def dialogues(count, offset):
        return [
            {
                "key": (offset + index, index),
                "question": f"{index}+1?",
                "subjects": [],
                "turns": [
                    {"role": "tutor", "content": "Try."},
                    {"role": "student", "content": str(index + 1)},
                ],
            }
            for index in range(count)
        ]

    splits = DatasetSplits(train=dialogues(8, 0), validation=dialogues(2, 100), test=[])
    monkeypatch.setattr(training.AutoModelForCausalLM, "from_pretrained", load_model)
    monkeypatch.setattr(training.AutoTokenizer, "from_pretrained", load_tokenizer)
    monkeypatch.setattr(training, "load_eedi_splits", lambda _: splits)
    interrupt = {"enabled": False, "step": 2}
    rebased_best = []

    class StopAfterSavedEpoch(transformers.TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            # Stop only after a sealed checkpoint with a selected best adapter.
            if (
                interrupt["enabled"]
                and state.global_step == interrupt["step"]
                and state.best_model_checkpoint
            ):
                checkpoint = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                assert (checkpoint / "resume.json").is_file()
                assert read_json(checkpoint / "trainer_state.json")["best_model_checkpoint"]
                raise InterruptedTraining(f"checkpoint-{state.global_step} sealed after evaluation")

        def on_train_begin(self, args, state, control, **kwargs):
            if state.best_model_checkpoint:
                rebased_best.append(state.best_model_checkpoint)

    class ControlledTrainer(transformers.Trainer):
        def __init__(self, *args, **kwargs):
            assert isinstance(kwargs["callbacks"][-1], training.RecoveryCallback)
            kwargs["callbacks"] = [*kwargs["callbacks"], StopAfterSavedEpoch()]
            super().__init__(*args, **kwargs)

    monkeypatch.setattr(training, "Trainer", ControlledTrainer)

    def run(output_dir, model_key="llama_3_2_3b", **kwargs):
        return training.train_student(
            tmp_path / "offline-dataset",
            output_dir,
            config_path=config_path,
            model_key=model_key,
            model_id_override="controlled-tiny",
            model_revision="fixture",
            **kwargs,
        )

    yield SimpleNamespace(
        torch=torch,
        training=training,
        config=config,
        config_path=config_path,
        splits=splits,
        interrupt=interrupt,
        run=run,
        model_loads=model_loads,
        rebased_best=rebased_best,
    )
    torch.set_num_threads(previous_threads)


@pytest.mark.parametrize("model_key", MODEL_KEYS)
def test_real_training_resume_is_bitwise_equal(controlled_training, tmp_path, model_key):
    fixture = controlled_training
    uninterrupted = tmp_path / "uninterrupted"
    fixture.run(uninterrupted, model_key)
    interrupted = tmp_path / "interrupted"
    fixture.interrupt["enabled"] = True
    with pytest.raises(InterruptedTraining, match="checkpoint-2 sealed"):
        fixture.run(interrupted, model_key)
    checkpoint = interrupted / "checkpoint-2"
    original = snapshot(interrupted)
    saved_state = read_json(checkpoint / "trainer_state.json")
    assert saved_state["global_step"] == 2
    assert saved_state["epoch"] == 1.0
    assert saved_state["max_steps"] == 6
    assert Path(saved_state["best_model_checkpoint"]).parent == interrupted
    receipt = read_json(checkpoint / "resume.json")
    assert receipt["identity"]["limits"] == [None, None, None]
    assert receipt["identity"]["runtime"]["device"] == "cpu"

    # Move a backup, leaving the source intact: receipts/native state must not
    # need edits to turn the old absolute best-model path into a usable new one.
    backup = tmp_path / "backup"
    shutil.copytree(interrupted, backup)
    relocated = tmp_path / "relocated"
    shutil.move(str(backup), relocated)
    fixture.interrupt["enabled"] = False
    fixture.run(relocated, model_key, resume_from_checkpoint=relocated / "checkpoint-2")
    assert fixture.rebased_best == [str(relocated / "checkpoint-2")]
    assert snapshot(interrupted) == original

    final_state = read_json(uninterrupted / "trainer_state.json")
    resumed_state = read_json(relocated / "trainer_state.json")
    for state in (final_state, resumed_state):
        assert state["epoch"] == 3.0
        assert state["global_step"] == state["max_steps"] == 6
        assert state["num_train_epochs"] == 3
    assert final_state["best_metric"] == resumed_state["best_metric"]
    assert (
        Path(final_state["best_model_checkpoint"]).name
        == Path(resumed_state["best_model_checkpoint"]).name
    )
    assert Path(resumed_state["best_model_checkpoint"]).parent == relocated
    full_evals = [
        entry["eval_loss"] for entry in final_state["log_history"] if "eval_loss" in entry
    ]
    resumed_evals = [
        entry["eval_loss"] for entry in resumed_state["log_history"] if "eval_loss" in entry
    ]
    assert len(full_evals) == 3
    assert full_evals == resumed_evals

    from safetensors.torch import load_file

    # train_student exports the selected final adapter directly in output_dir.
    adapter_final = load_file(str(uninterrupted / "adapter_model.safetensors"))
    adapter_resumed = load_file(str(relocated / "adapter_model.safetensors"))
    assert adapter_final.keys() == adapter_resumed.keys()
    assert adapter_final
    assert any(
        fixture.torch.count_nonzero(tensor).item() > 0
        for name, tensor in adapter_final.items()
        if "lora_B" in name
    ), "LoRA B starts at zero; a nonzero tensor proves the adapter actually trained"
    for name in adapter_final:
        assert fixture.torch.equal(adapter_final[name], adapter_resumed[name]), name
    selected = load_file(
        str(Path(resumed_state["best_model_checkpoint"]) / "adapter_model.safetensors")
    )
    assert all(fixture.torch.equal(adapter_resumed[name], selected[name]) for name in selected)


@pytest.mark.parametrize("changed", ["config", "train", "validation", "model", "code", "runtime"])
def test_resume_rejects_changed_identity_before_model_loading(
    controlled_training,
    tmp_path,
    monkeypatch,
    changed,
):
    fixture = controlled_training
    output = tmp_path / "interrupted"
    fixture.interrupt["enabled"] = True
    with pytest.raises(InterruptedTraining):
        fixture.run(output)
    checkpoint = output / "checkpoint-2"
    original = snapshot(output)
    fixture.interrupt["enabled"] = False
    config = copy.deepcopy(fixture.config)
    model_key = "llama_3_2_3b"
    if changed == "config":
        config["training"]["learning_rate"] *= 2
        fixture.config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    elif changed in {"train", "validation"}:
        getattr(fixture.splits, changed)[0]["turns"][-1]["content"] = "Changed answer"
    elif changed == "model":
        model_key = "qwen3_4b"
    elif changed == "code":
        original_hash = fixture.training.file_sha256
        monkeypatch.setattr(
            fixture.training,
            "file_sha256",
            lambda path: "0" * 64 if path.name == "training.py" else original_hash(path),
        )
    else:
        monkeypatch.setattr(fixture.training.platform, "python_version", lambda: "changed-runtime")

    def forbidden_model_load(*args, **kwargs):
        pytest.fail("An incompatible checkpoint must fail before loading model weights")

    def forbidden_gpu_probe(*args, **kwargs):
        pytest.fail("Controlled resume must not initialize a GPU")

    monkeypatch.setattr(
        fixture.training.AutoModelForCausalLM, "from_pretrained", forbidden_model_load
    )
    monkeypatch.setattr(fixture.torch.cuda, "get_device_name", forbidden_gpu_probe)
    monkeypatch.setattr(fixture.torch.cuda, "init", forbidden_gpu_probe)
    with pytest.raises(ValueError, match="Resume identity differs"):
        fixture.run(output, model_key, resume_from_checkpoint=checkpoint)
    assert snapshot(output) == original


def test_legacy_8b_model_only_checkpoint_is_rejected_before_cuda_or_loaders(
    controlled_training,
    tmp_path,
    monkeypatch,
):
    fixture = controlled_training
    output = tmp_path / "legacy-8b"
    checkpoint = output / "checkpoint-34"
    checkpoint.mkdir(parents=True)
    # An old LoRA-only export, with no optimizer/scheduler/RNG or receipt.
    (checkpoint / "adapter_config.json").write_text(
        json.dumps(
            {
                "base_model_name_or_path": fixture.config["models"]["llama_3_1_8b"]["id"],
            }
        )
    )
    (checkpoint / "trainer_state.json").write_text(json.dumps({"global_step": 34, "epoch": 2}))

    def forbidden(*args, **kwargs):
        pytest.fail("Legacy checkpoints must fail before CUDA checks and pretrained loading")

    for name in ("is_available", "device_count", "is_bf16_supported", "init"):
        monkeypatch.setattr(fixture.torch.cuda, name, forbidden)
    monkeypatch.setattr(fixture.training.AutoModelForCausalLM, "from_pretrained", forbidden)
    monkeypatch.setattr(fixture.training.AutoTokenizer, "from_pretrained", forbidden)
    with pytest.raises(ValueError, match="no complete-state receipt"):
        fixture.training.train_student(
            tmp_path / "offline-dataset",
            output,
            config_path=fixture.config_path,
            model_key="llama_3_1_8b",
            resume_from_checkpoint=checkpoint,
        )


@pytest.mark.parametrize("epoch,expected", [(0.5, True), (1.0, False)])
def test_intermediate_save_never_preempts_epoch_evaluation(controlled_training, epoch, expected):
    callback = controlled_training.training.RecoveryCallback({}, 1)
    state = SimpleNamespace(global_step=1, epoch=epoch)
    control = SimpleNamespace(should_save=False)
    callback.on_step_end(None, state, control)
    assert control.should_save is expected


@pytest.mark.parametrize("step,max_steps", [(3, None), (2, 6)])
def test_mid_epoch_and_explicit_step_horizon_resume(controlled_training, tmp_path, step, max_steps):
    fixture = controlled_training
    if max_steps is not None:
        fixture.config["training"]["epochs"] = 0.5
        fixture.config_path.write_text(yaml.safe_dump(fixture.config))
    full = tmp_path / "full"
    fixture.run(full, max_steps=max_steps)
    interrupted = tmp_path / "interrupted"
    fixture.interrupt.update(enabled=True, step=step)
    with pytest.raises(InterruptedTraining):
        fixture.run(interrupted, max_steps=max_steps)
    checkpoint = interrupted / f"checkpoint-{step}"
    if step == 3:
        assert read_json(checkpoint / "trainer_state.json")["epoch"] == 1.5
    fixture.interrupt["enabled"] = False
    resumed = tmp_path / "resumed"
    shutil.copytree(interrupted, resumed)
    fixture.run(resumed, max_steps=max_steps, resume_from_checkpoint=resumed / checkpoint.name)
    from safetensors.torch import load_file

    original = load_file(str(full / "adapter_model.safetensors"))
    restored = load_file(str(resumed / "adapter_model.safetensors"))
    assert all(fixture.torch.equal(original[name], restored[name]) for name in original)
    original_state = read_json(full / "trainer_state.json")
    resumed_state = read_json(resumed / "trainer_state.json")
    assert resumed_state["global_step"] == original_state["global_step"] == 6
    assert [e["eval_loss"] for e in original_state["log_history"] if "eval_loss" in e] == [
        e["eval_loss"] for e in resumed_state["log_history"] if "eval_loss" in e
    ]
