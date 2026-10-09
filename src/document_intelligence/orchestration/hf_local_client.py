"""Client `LLMClient` pour un petit modèle Hugging Face local (CPU), avec adaptateur LoRA.

Utilisé pour l'expérience de fine-tuning (`finetune/`). Le modèle n'a pas de
function calling : les sorties structurées passent par le repli « JSON par
instruction » de `LLMClient.complete_structured`, dont le texte
(`json_instruction_prompt`) est exactement celui du jeu d'entraînement.

Dépendances optionnelles, hors lock : `uv pip install -r finetune/requirements.txt`.
"""

import time
from typing import Any

from document_intelligence.observability import registry, trace_span
from document_intelligence.orchestration.llm import LLMClient


class HFLocalLLMClient(LLMClient):
    """Génération gloutonne (déterministe) avec un modèle causal local.

    Args:
        base_model: identifiant Hugging Face ou dossier local du modèle de base.
        adapter: dossier d'un adaptateur LoRA (PEFT) à fusionner ; `None` = modèle de base.
        max_new_tokens: borne de la sortie générée.
    """

    def __init__(
        self,
        base_model: str,
        adapter: str | None = None,
        max_new_tokens: int = 384,
        threads: int | None = None,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if threads:
            torch.set_num_threads(threads)
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(base_model)
        model = AutoModelForCausalLM.from_pretrained(base_model, dtype=torch.float32)
        if adapter:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, adapter).merge_and_unload()
        model.eval()
        self._model = model
        self._max_new_tokens = max_new_tokens
        self.model = f"{base_model}+{adapter}" if adapter else base_model

    def encode(self, prompt: str) -> Any:
        """Entrée tokenisée au format chat, identique à celle de l'entraînement."""
        return self._tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

    def complete(self, prompt: str) -> str:
        inputs = self.encode(prompt)
        prompt_tokens = int(inputs["input_ids"].shape[1])
        started = time.perf_counter()
        with trace_span(
            "llm.call",
            **{"gen_ai.system": "local", "gen_ai.request.model": self.model},
        ) as span:
            with self._torch.inference_mode():
                output = self._model.generate(
                    **inputs,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,
                    pad_token_id=self._tokenizer.eos_token_id,
                )
            generated = output[0, prompt_tokens:]
            span["gen_ai.usage.input_tokens"] = prompt_tokens
            span["gen_ai.usage.output_tokens"] = int(generated.shape[0])
        registry.record_llm_call(
            input_tokens=prompt_tokens,
            output_tokens=int(generated.shape[0]),
            latency_ms=(time.perf_counter() - started) * 1000,
        )
        return str(self._tokenizer.decode(generated, skip_special_tokens=True))
