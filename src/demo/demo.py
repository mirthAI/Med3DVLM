import os
import random
from dataclasses import dataclass, field

import numpy as np
import SimpleITK as sitk
import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


@dataclass
class AllArguments:
    model_name_or_path: str = field(default="./models/Med3DVLM-Qwen-2.5-7B")

    image_path: str = field(
        default="./data/demo/024421/Axial_C__portal_venous_phase.nii.gz"
    )

    question: str = field(
        default="Describe the findings of the medical image you see.",
        metadata={"help": "Question to ask the model."},
    )

    model_max_length: int = field(
        default=512, metadata={"help": "Maximum length of the input sequence."}
    )


def main():
    seed_everything(42)
    device = torch.device("cuda")  # 'cpu', 'cuda'
    dtype = torch.bfloat16  # or bfloat16, float16, float32

    parser = transformers.HfArgumentParser(AllArguments)
    args = parser.parse_args_into_dataclasses()[0]

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
        trust_remote_code=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=dtype,
        device_map=device,
        trust_remote_code=True,
    )

    proj_out_num = (
        model.get_model().config.proj_out_num
        if hasattr(model.get_model().config, "proj_out_num")
        else 256
    )

    question = args.question

    image_tokens = "<im_patch>" * proj_out_num
    input_txt = image_tokens + question
    input_id = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device=device)

    image_np = np.expand_dims(
        sitk.GetArrayFromImage(sitk.ReadImage(args.image_path)), axis=0
    )
    image_pt = torch.from_numpy(image_np).unsqueeze(0).to(dtype=dtype, device=device)

    generation = model.generate(
        images=image_pt,
        inputs=input_id,
        max_new_tokens=args.model_max_length,
        do_sample=True,
        top_p=0.9,
        temperature=1.0,
    )

    generated_texts = tokenizer.batch_decode(generation, skip_special_tokens=True)

    print("question: ", question)
    print("generated_texts: ", generated_texts[0])


if __name__ == "__main__":
    main()
