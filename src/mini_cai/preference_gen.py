"""
preference_gen.py

Sample two responses (A,B) from the SL-CAI model per prompt, ask a judge
(GPT-4o or local model) which is better (harmless / helpful) according to a
randomly-sampled constitutional principle, and save a JSONL file ready for
TRL's DPOTrainer:

{"prompt": ..., "chosen": ..., "rejected": ...}

Usage:
python -m src.mini_cai.preference_gen \
    --prompt_file data/raw/red_team_prompts_preference.jsonl \
    --sl_cai_path models/stage_02_sl_cai \
    --out_file    data/processed/dpo_preferences.jsonl \
    --judge gpt-4o \
    --n_per_prompt 2
"""

import argparse, json, random, os, pathlib
from tqdm.auto import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Judge-prompt utilities
from src.mini_cai.constants import DPO_CAI_PRINCIPLES as CONSTITUTION


def build_judge_prompt(prompt: str, a: str, b: str) -> str:
    principle = random.choice(CONSTITUTION)
    return (
        "Human: Consider the following conversation between a human and an assistant:\n"
        f"{prompt}\n"
        f"{principle}\n"
        f"(A) {a}\n"
        f"(B) {b}\n"
        "The answer is:"
    )


# GPT-4o judge
class GPT4Judge:
    def __init__(self, model="gpt-4o"):
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.model = model

    def choose(self, q: str) -> str:
        chat = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": q}],
            temperature=0.3,
            max_tokens=2,
        )
        first = chat.choices[0].message.content.strip().upper()
        return "A" if first.startswith("A") else "B"


# Local judge (e.g. Mistral-7B-Instruct)
class LocalJudge:
    """
    Small HF model that answers with 'A' or 'B' given the constitutional prompt.
    """

    def __init__(self, model_id="mistralai/Mistral-7B-Instruct-v0.1"):
        tok = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(
            model_id, device_map="auto", torch_dtype=torch.float16
        )
        self.pipe = pipeline("text-generation", model=model, tokenizer=tok, device=0)

    def choose(self, q: str) -> str:
        ans = self.pipe(
            q,
            max_new_tokens=2,
            temperature=0.1,
            do_sample=False,
            pad_token_id=self.pipe.tokenizer.eos_token_id,
        )[0]["generated_text"]
        return "A" if "A" in ans[:5].upper() else "B"


def load_judge(name: str):
    if name.lower() in {"gpt-4", "gpt-4o"}:
        return GPT4Judge()
    return LocalJudge(model_id=name)


# CLI & helpers
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt_file", required=True)
    p.add_argument("--sl_cai_path", required=True)
    p.add_argument("--out_file", required=True)
    p.add_argument(
        "--judge", default="gpt-4o", help="'gpt-4o' (OpenAI) or a local HF model id"
    )
    p.add_argument("--n_per_prompt", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def load_prompts(path: pathlib.Path):
    return [json.loads(l)["prompt"] for l in open(path)]


# Main
def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    prompts = load_prompts(args.prompt_file)
    print("Prompts:", len(prompts))

    # SL-CAI policy
    tok = AutoTokenizer.from_pretrained(args.sl_cai_path)
    policy = AutoModelForCausalLM.from_pretrained(
        args.sl_cai_path, device_map="auto", torch_dtype=torch.float16
    )
    gen = pipeline(
        "text-generation",
        model=policy,
        tokenizer=tok,
        device=0 if torch.cuda.is_available() else -1,
    )

    judge = load_judge(args.judge)

    out_path = pathlib.Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w") as fout:
        for prompt in tqdm(prompts, desc="pref-gen"):
            # (1) sample several responses, pick two distinct ones
            samples = [
                gen(
                    prompt,
                    max_new_tokens=256,
                    temperature=1.0,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=tok.eos_token_id,
                )[0]["generated_text"]
                for _ in range(max(4, args.n_per_prompt * 2))
            ]
            a, b = random.sample(samples, 2)

            # (2) judge
            judge_prompt = build_judge_prompt(prompt, a, b)
            choice = judge.choose(judge_prompt)

            chosen, rejected = (a, b) if choice == "A" else (b, a)
            fout.write(
                json.dumps({"prompt": prompt, "chosen": chosen, "rejected": rejected})
                + "\n"
            )

    print(f"Preference JSONL saved to {out_path}")


if __name__ == "__main__":
    main()
