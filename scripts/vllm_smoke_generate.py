from __future__ import annotations

from vllm import LLM, SamplingParams


def main() -> None:
    llm = LLM(
        model="Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True,
        gpu_memory_utilization=0.65,
        max_model_len=1024,
    )
    outputs = llm.generate(
        ["Complete this Lean proof:\nexample : True := by\n"],
        SamplingParams(max_tokens=16, temperature=0.0),
    )
    for output in outputs:
        print(output.outputs[0].text)


if __name__ == "__main__":
    main()

