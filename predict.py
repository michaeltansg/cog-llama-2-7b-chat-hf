from typing import List
from cog import BasePredictor, Input
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

WEIGHTS_DIR = 'weights'
MODEL_NAME = 'Llama-2-7b-chat-hf'
MODEL_PATH = f'{WEIGHTS_DIR}/{MODEL_NAME}'


B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
PROMPT_TEMPLATE = f"{B_INST} {B_SYS}{{system_prompt}}{E_SYS}{{instruction}} {E_INST}"

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information."""

class Predictor(BasePredictor):
    def setup(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, local_files_only=True)
        self.model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, local_files_only=True)

    def predict(
        self,
        prompt: str = Input(description="Text prompt to send to the model."),
        system_prompt: str = Input(
            description="System prompt to send to the model. This is prepended to the prompt and helps guide system behavior.", 
            default=DEFAULT_SYSTEM_PROMPT,
        ),
        max_length: int = Input(
            description="Maximum number of tokens to generate. A word is generally 2-3 tokens",
            ge=1,
            default=500,
        ),
        temperature: float = Input(
            description="Adjusts randomness of outputs, greater than 1 is random and 0 is deterministic, 0.75 is a good starting value.",
            ge=0.01,
            le=5,
            default=0.75,
        ),
        top_p: float = Input(
            description="When decoding text, samples from the top p percentage of most likely tokens; lower to ignore less likely tokens",
            ge=0.01,
            le=1.0,
            default=1.0
        ),
        repetition_penalty: float = Input(
            description="Penalty for repeated words in generated text; 1 is no penalty, values greater than 1 discourage repetition, less than 1 encourage it.",
            ge=0.01,
            le=5,
            default=1
        ),
    ) -> List[str]:

        prompt = prompt.strip('\n').lstrip(B_INST).rstrip(E_INST).strip()
        prompt_templated = PROMPT_TEMPLATE.format(system_prompt=system_prompt.strip(), instruction=prompt.strip())

        input = self.tokenizer(prompt_templated, return_tensors="pt").input_ids.to(self.device)

        outputs = self.model.generate(
            input,
            max_length=max_length,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
        out = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return out