"""Fine-tuning LoRA (PEFT) de la synthèse sur CPU, à partir de `train.jsonl`.

- Format : message utilisateur = `input` (prompt exact de l'inférence), réponse
  de l'assistant = `target` (JSON `GroundedAnswer` de l'enseignant), via le
  gabarit de chat du modèle. La perte ne porte que sur les tokens de la réponse.
- Validation : les exemples d'un document entier sont mis de côté (et non des
  lignes au hasard) : la perte de validation mesure la généralisation à un
  document jamais vu, et révèle le surapprentissage d'une époque à l'autre.

    uv run python finetune/train_lora.py --data finetune/data --out finetune/adapter
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_rows(data: Path) -> list[dict[str, str]]:
    return [
        json.loads(line)
        for line in (data / "train.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def split_by_document(
    rows: list[dict[str, str]], data: Path, holdout_doc: int
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Met de côté toutes les questions générées pour le document `holdout_doc`."""
    path = data / "questions" / f"doc{holdout_doc:02d}.json"
    held = {
        q["question"] for q in json.loads(path.read_text(encoding="utf-8"))["questions"]
    }
    return (
        [r for r in rows if r["question"] not in held],
        [r for r in rows if r["question"] in held],
    )


def drop_leakage(
    rows: list[dict[str, str]], data: Path
) -> tuple[list[dict[str, str]], int]:
    """Retire les exemples recoupant le jeu d'évaluation (cf. `leakage_report.json`).

    Sont exclus : les questions quasi identiques à une question d'évaluation, et
    les exemples dont la **réponse cible** énonce un passage attendu par
    l'évaluation. Un contexte qui contient seulement ces formules (« manquement
    grave »...) est conservé : les faits des documents diffèrent, et l'exclure
    retirait plus de la moitié des exemples de contrats.
    """
    report = json.loads((data / "leakage_report.json").read_text(encoding="utf-8"))
    near = {pair[1] for pair in report["near_duplicate_questions_ratio_ge_0.8"]}
    passages = [p.lower() for p in report["eval_relevant_substrings_found"]]
    kept = [
        r
        for r in rows
        if r["question"] not in near
        and not any(p in r["target"].lower() for p in passages)
    ]
    return kept, len(rows) - len(kept)


def encode(tokenizer, row: dict[str, str]) -> tuple[torch.Tensor, torch.Tensor]:  # type: ignore[no-untyped-def]
    """input_ids et labels (-100 sur le prompt : perte sur la réponse seule)."""
    messages = [{"role": "user", "content": row["input"]}]
    prompt_ids = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, return_dict=True
    )["input_ids"]
    full_ids = tokenizer.apply_chat_template(
        [*messages, {"role": "assistant", "content": row["target"]}],
        return_dict=True,
    )["input_ids"]
    assert full_ids[: len(prompt_ids)] == prompt_ids, "gabarit de chat non préfixe"
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    return torch.tensor([full_ids]), torch.tensor([labels])


@torch.no_grad()
def mean_loss(model, examples) -> float:  # type: ignore[no-untyped-def]
    model.eval()
    losses = [
        model(input_ids=ids, labels=labels).loss.item() for ids, labels in examples
    ]
    model.train()
    return sum(losses) / len(losses)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("finetune/data"))
    parser.add_argument("--out", type=Path, default=Path("finetune/adapter"))
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--holdout-doc", type=int, default=9)
    parser.add_argument("--max-train", type=int, help="Limite (mesure de vitesse).")
    parser.add_argument("--max-val", type=int, help="Limite (mesure de vitesse).")
    parser.add_argument(
        "--no-gradient-checkpointing",
        action="store_true",
        help="Plus rapide, plus de mémoire (activations conservées).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    model = AutoModelForCausalLM.from_pretrained(args.base_model, dtype=torch.float32)
    if not args.no_gradient_checkpointing:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)
    trainable, total = model.get_nb_trainable_parameters()

    rows, dropped = drop_leakage(load_rows(args.data), args.data)
    print(f"exemples exclus (recoupement avec l'évaluation) : {dropped}", flush=True)
    train_rows, val_rows = split_by_document(rows, args.data, args.holdout_doc)
    if args.max_train:
        train_rows = train_rows[: args.max_train]
    if args.max_val:
        val_rows = val_rows[: args.max_val]
    train = [encode(tokenizer, r) for r in train_rows]
    val = [encode(tokenizer, r) for r in val_rows]
    lengths = sorted(ids.shape[1] for ids, _ in train)
    print(
        f"train={len(train)} val={len(val)} (document {args.holdout_doc} mis de côté) | "
        f"tokens/exemple médiane={lengths[len(lengths) // 2]} max={lengths[-1]} | "
        f"paramètres entraînables={trainable:,} / {total:,} ({100 * trainable / total:.2f} %)",
        flush=True,
    )

    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    steps = math.ceil(len(train) / args.grad_accum) * args.epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda s: max(0.0, 1 - s / steps),  # décroissance linéaire
    )

    history: list[dict[str, float]] = []
    val_loss = mean_loss(model, val)
    history.append({"epoch": 0, "val_loss": val_loss})
    print(f"époque 0 : val_loss={val_loss:.4f}", flush=True)

    model.train()
    started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        order = list(range(len(train)))
        random.shuffle(order)
        running: list[float] = []
        for i, index in enumerate(order, start=1):
            ids, labels = train[index]
            loss = model(input_ids=ids, labels=labels).loss
            (loss / args.grad_accum).backward()
            running.append(loss.item())
            if i % args.grad_accum == 0 or i == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            if i % 20 == 0 or (args.max_train and i <= args.max_train):
                elapsed = time.perf_counter() - started
                print(
                    f"époque {epoch} exemple {i}/{len(order)} "
                    f"loss(20)={sum(running[-20:]) / 20:.4f} écoulé={elapsed / 60:.1f} min",
                    flush=True,
                )
        train_loss = sum(running) / len(running)
        val_loss = mean_loss(model, val)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"époque {epoch} : train_loss={train_loss:.4f} val_loss={val_loss:.4f}",
            flush=True,
        )
        model.save_pretrained(str(args.out / f"epoch{epoch}"))

    summary = {
        "base_model": args.base_model,
        "lora": {
            "rank": args.rank,
            "alpha": args.alpha,
            "dropout": args.dropout,
            "target_modules": sorted(config.target_modules),
        },
        "learning_rate": args.lr,
        "schedule": "linéaire vers 0, sans warmup",
        "epochs": args.epochs,
        "effective_batch_size": args.grad_accum,
        "gradient_checkpointing": not args.no_gradient_checkpointing,
        "excluded_for_leakage": dropped,
        "train_examples": len(train),
        "val_examples": len(val),
        "holdout_document": args.holdout_doc,
        "trainable_parameters": trainable,
        "total_parameters": total,
        "median_tokens_per_example": lengths[len(lengths) // 2],
        "training_minutes": round((time.perf_counter() - started) / 60, 1),
        "history": history,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
