from my_tq_star import setup_models_and_tokenizer
import torch
import torch.nn.functional as F
from tqdm import tqdm
import warnings
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
)


class TQ:
    def __init__(
        self,
        max_new_tokens=128,
        temperature=0.7,
        topk=1,
        helpful_weight=0.0,
        harmless_weight=0.0,
        chunk_size=5,
        prescreen_beam_width=40,
        is_helpful_r_chat_type=True,
        is_harmless_r_chat_type=True,
    ) -> None:

        llm_model_name = "ChenmieNLP/Zephyr-7B-Beta-Helpful"
        helpful_reward_model_name = "LxzGordon/URM-LLaMa-3-8B"
        harmless_reward_model_name = "NCSOFT/Llama-3-OffsetBias-RM-8B"

        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.chunk_size = chunk_size
        self.topk = topk
        self.helpful_weight = helpful_weight
        self.harmless_weight = harmless_weight
        self.is_helpful_r_chat_type = is_helpful_r_chat_type
        self.is_harmless_r_chat_type = is_harmless_r_chat_type

        self.llm = AutoModelForCausalLM.from_pretrained(
            llm_model_name, device_map="cpu", torch_dtype=torch.bfloat16
        )
        self.llm_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        if self.llm_tokenizer.pad_token_id is None:
            self.llm_tokenizer.pad_token_id = self.llm_tokenizer.eos_token_id

        self.llm_tokenizer.padding_side = "left"
        self.llm = self.llm.to("cuda").eval()

        if self.helpful_weight > 0.0:
            self.helpful_reward = AutoModelForSequenceClassification.from_pretrained(
                helpful_reward_model_name,
                # num_labels=1,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )
            self.helpful_reward_tokenizer = AutoTokenizer.from_pretrained(
                helpful_reward_model_name
            )
            if self.helpful_reward_tokenizer.pad_token_id is None:
                self.helpful_reward_tokenizer.pad_token_id = (
                    self.helpful_reward_tokenizer.eos_token_id
                )
                # self.helpful_reward.config.pad_token_id = self.helpful_reward_tokenizer.eos_token_id
            self.helpful_reward_tokenizer.padding_side = "left"
            self.helpful_reward = self.helpful_reward.to("cuda").eval()

        if self.harmless_weight > 0.0:
            self.harmless_reward = AutoModelForSequenceClassification.from_pretrained(
                harmless_reward_model_name,
                num_labels=1,
                torch_dtype=torch.bfloat16,
                device_map="cpu",
            )

            self.harmless_reward_tokenizer = AutoTokenizer.from_pretrained(
                harmless_reward_model_name
            )
            if self.harmless_reward_tokenizer.pad_token_id == None:
                self.harmless_reward_tokenizer.pad_token_id = (
                    self.harmless_reward_tokenizer.eos_token_id
                )
                # self.harmless_reward.config.pad_token_id =  self.harmless_reward_tokenizer.eos_token_id
            self.harmless_reward_tokenizer.padding_side = "left"
            self.harmless_reward = self.harmless_reward.to("cuda").eval()

        self.device = "cuda"
        self.inputs = None
        self.logits_vector = None
        self.prescreen_beam_width = prescreen_beam_width
        self.num_prompt_tokens = 0
        self.prompt_text = None
        self.scores = []

    def get_inputs(self):
        self.inputs = self.inputs.to(self.device)
        attention_mask = torch.ones_like(self.inputs)
        return {"input_ids": self.inputs, "attention_mask": attention_mask}

    def generate_next_token(self):
        prescreen_logits, prescreen_tokens = torch.topk(
            self.logits_vector, dim=-1, k=self.prescreen_beam_width
        )
        expanded_inputs = torch.unsqueeze(self.inputs, 1).repeat(
            1, self.prescreen_beam_width, 1
        )

        to_rm_eval = torch.dstack((expanded_inputs, prescreen_tokens))

        # flat_trme_ext = self.llm.generate(
        #     input_ids=to_rm_eval[0],
        #     attention_mask=torch.ones_like(to_rm_eval[0]),
        #     max_new_tokens=5,
        # )

        possible_sentences = [
            self.llm_tokenizer.decode(tokens[self.num_prompt_tokens :])
            # for tokens in flat_trme_ext
            for tokens in to_rm_eval[0]
        ]

        responses = [
            [
                {"role": "user", "content": self.prompt_text},
                {"role": "assistant", "content": resp},
            ]
            for resp in possible_sentences
        ]

        scores = prescreen_logits.flatten()
        if self.helpful_weight > 0.0:

            if self.is_helpful_r_chat_type:
                helpful_reward_texts = (
                    self.helpful_reward_tokenizer.apply_chat_template(
                        responses, tokenize=False, add_generation_prompt=False
                    )
                )
                helpful_reward_texts = [
                    text.replace(self.helpful_reward_tokenizer.bos_token, "")
                    for text in helpful_reward_texts
                ]
            else:
                helpful_reward_texts = [
                    self.prompt_text + resp for resp in possible_sentences
                ]

            helpful_reward_text_tokens = self.helpful_reward_tokenizer(
                helpful_reward_texts, return_tensors="pt", padding=True
            ).to("cuda")

            with torch.inference_mode():
                helpful_reward_logits = self.helpful_reward(
                    **helpful_reward_text_tokens
                ).logits[:, 0]
            scores += self.helpful_weight * helpful_reward_logits.flatten()

        if self.harmless_weight > 0.0:
            if self.is_harmless_r_chat_type:
                harmless_reward_texts = (
                    self.harmless_reward_tokenizer.apply_chat_template(
                        responses, tokenize=False, add_generation_prompt=False
                    )
                )
                harmless_reward_texts = [
                    text.replace(self.harmless_reward_tokenizer.bos_token, "")
                    for text in harmless_reward_texts
                ]
            else:
                harmless_reward_texts = [
                    self.prompt_text + resp for resp in possible_sentences
                ]

            harmless_reward_text_tokens = self.harmless_reward_tokenizer(
                harmless_reward_texts, return_tensors="pt", padding=True
            ).to("cuda")

            with torch.inference_mode():
                harmless_reward_logits = self.harmless_reward(
                    **harmless_reward_text_tokens
                ).logits

            scores += self.harmless_weight * harmless_reward_logits.flatten()

        new_scores = scores / self.temperature
        probs = F.softmax(new_scores, dim=-1)
        top_k_id = torch.argmax(probs, dim=-1, keepdim=True)

        self.inputs = to_rm_eval[0][top_k_id]
        self.scores.append(probs[top_k_id].item())

    def generate(self, prompt_text):
        self.scores = []
        self.prompt_text = prompt_text
        self.inputs = self.llm_tokenizer(prompt_text, return_tensors="pt")[
            "input_ids"
        ].to("cuda")
        self.num_prompt_tokens = self.inputs.shape[-1]

        # Ensure the LLM and Reward Models are in GPU and clear the inputs and outputs after calculation
        for iter in tqdm(range(self.max_new_tokens)):
            with torch.inference_mode():

                logits = self.llm(
                    input_ids=self.inputs,
                    attention_mask=torch.ones_like(self.inputs),
                    return_dict=True,
                ).logits

                self.logits_vector = logits[:, -1]

            self.generate_next_token()

            if self.inputs[0][-1].item() == self.llm_tokenizer.eos_token_id:
                break

        return (
            self.llm_tokenizer.decode(self.inputs[0][self.num_prompt_tokens :]),
            self.scores,
        )
