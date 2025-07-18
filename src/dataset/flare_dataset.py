import json
import os
import random

import monai.transforms as mtf
import numpy as np
import SimpleITK as sitk
import torch
from monai.data import set_track_meta
from torch.utils.data import ConcatDataset, Dataset

prompt_templates = {
    "simple": """You are an AI assistant trained to act as an abdominal radiologist. Please describe in detail the findings in this abdomen CT scan.
        """
}


def read_image(path):
    ext = path.split(os.sep)[-1].split(".")[-1]
    if ext == "npy":
        image = np.load(path)  # nomalized 0-1, C,D,H,W
    else:
        img_sitk = sitk.ReadImage(path)
        img_data = sitk.GetArrayFromImage(img_sitk)

        if len(img_data.shape) == 4:
            img_data = img_data[1]

        img_data = np.clip(img_data, -160.0, 240.0)
        img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data))
        image = np.expand_dims(img_data, 0)
        del img_data, img_sitk
    return image


class FLARECapDataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        # print("Arguments provided in 'args':")
        # for key, value in vars(self.args).items():
        #     print(f"{key}: {value}")

        self.image_tokens = "<im_patch>" * args.proj_out_num

        assert len(args.json_path) == len(
            args.data_root
        ), "You need to provide the image directory for every dataset's JSON."

        self.data_list = self._make_combined_json(args)

        # print(f"Length dataset: {len(self.data_list)}")

        if args.prompt in prompt_templates.keys():
            # print("Prompt is a dict")
            self.prompt = prompt_templates[args.prompt]
        else:
            self.prompt = args.prompt

        train_transform = mtf.Compose(
            [
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
            ]
        )
        self.resize_transform = mtf.Resize(args.data_img_size)
        set_track_meta(False)

        if mode == "train" or mode == "train_val":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
            self.return_all = True
        elif "test" in mode:
            self.transform = val_transform

    def _make_combined_json(self, args):
        paths = args.json_path
        image_paths = args.data_root

        if isinstance(paths, list):
            combined = []
            for idx, p in enumerate(paths):
                img_path = (
                    image_paths[idx] if isinstance(image_paths, list) else image_paths
                )
                with open(p, "r") as f:
                    data = json.load(f)
                    for item in data:
                        item["volume_path"] = img_path + os.sep + item["case_id"]
                    combined.extend(data)
            json_file = combined
        else:
            img_path = image_paths[0] if isinstance(image_paths, list) else image_paths
            with open(paths, "r") as f:
                data = json.load(f)
                for item in data:
                    item["volume_path"] = img_path + os.sep + item["case_id"]
            json_file = data
        return json_file

    def __len__(self):
        return len(self.data_list)

    def __getitem_validation__(self, idx):
        data = self.data_list[idx]

        finding = data["findings"]
        finding = "\n".join([f"{key}: {value}" for key, value in finding.items()])

        prompt = self.prompt
        msg = prompt
        system = msg.split(".")[0] + "."
        content = ".".join(msg.split(".")[1:])

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": content},
        ]
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)
        question = self.image_tokens + prompt

        image_path = data["volume_path"]
        # if image_path.startswith("/"):
        #     image_abs_path = image_path
        # else:
        #     image_abs_path = os.path.join(self.data_root, image_path)

        image = read_image(image_path)
        image = self.transform(image)
        image = self.resize_transform(image)

        text_tensor = self.tokenizer(question, return_tensors="pt")
        input_id = text_tensor["input_ids"]

        ret = {
            "image": image,
            "input_id": input_id,
            "question": question,
            "answer": finding,
            "image_name": image_path.split(os.sep)[-1].split(".")[0],
            "question_type": "Caption",
        }

        return ret

    def __getitem__(self, idx):
        max_tries = 10
        for _ in range(max_tries):
            try:
                if self.mode == "validation":
                    return self.__getitem_validation__(idx)

                data = self.data_list[idx]

                finding = data["findings"]
                finding = "\n".join(
                    [f"{key}: {value}" for key, value in finding.items()]
                )

                # first sentence in prompt is system prompt.
                msg = self.prompt
                system = msg.split(".")[0] + "."
                content = ".".join(msg.split(".")[1:])
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ]
                prompt = self.tokenizer.apply_chat_template(messages, tokenize=False)

                question = self.image_tokens + prompt

                image_path = data["volume_path"]
                image = read_image(image_path)
                image = self.transform(image)
                image = self.resize_transform(image)

                text_tensor = self.tokenizer(
                    question + " " + finding,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                input_id = text_tensor["input_ids"][0]
                attention_mask = text_tensor["attention_mask"][0]

                valid_len = torch.sum(attention_mask)
                if valid_len < len(input_id):
                    input_id[valid_len] = self.tokenizer.eos_token_id

                question_tensor = self.tokenizer(
                    question,
                    max_length=self.args.max_length,
                    truncation=True,
                    padding="max_length",
                    return_tensors="pt",
                )

                question_len = torch.sum(question_tensor["attention_mask"][0])

                label = input_id.clone()
                label[:question_len] = -100
                if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                    label[label == self.tokenizer.pad_token_id] = -100
                    if valid_len < len(label):
                        label[valid_len] = self.tokenizer.eos_token_id
                else:
                    label[label == self.tokenizer.pad_token_id] = -100

                ret = {
                    "image": image,
                    "input_id": input_id,
                    "label": label,
                    "attention_mask": attention_mask,
                    "question": question,
                    "answer": finding,
                    "question_type": "Caption",
                }

                return ret
            except Exception as e:
                print(
                    f"Error in __getitem__ at index {idx}: {e}, name: {self.data_list[idx]}"
                )
                idx = random.randint(0, len(self.data_list) - 1)


class FLAREVQADataset(Dataset):
    def __init__(self, args, tokenizer, mode="train"):
        self.args = args
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.mode = mode

        # print("Arguments provided in 'args':")
        # for key, value in vars(self.args).items():
        #     print(f"{key}: {value}")

        self.image_tokens = "<im_patch>" * args.proj_out_num

        assert len(args.json_path) == len(
            args.data_root
        ), "You need to provide the image directory for every dataset's JSON."

        self.data_list = self._make_combined_json(args)

        # print(f"Length dataset: {len(self.data_list)}")

        train_transform = mtf.Compose(
            [
                mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
                mtf.Resize(args.data_img_size),
            ]
        )

        val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
                mtf.Resize(args.data_img_size),
            ]
        )
        set_track_meta(False)

        if mode == "train" or mode == "train_val":
            self.transform = train_transform
        elif mode == "validation":
            self.transform = val_transform
        elif "test" in mode:
            self.transform = val_transform

    def _make_combined_json(self, args):
        paths = args.json_path
        image_paths = args.data_root

        if isinstance(paths, list):
            combined = []
            for idx, p in enumerate(paths):
                img_path = (
                    image_paths[idx] if isinstance(image_paths, list) else image_paths
                )
                with open(p, "r") as f:
                    data = json.load(f)
                    for item in data:
                        item["volume_path"] = img_path + os.sep + item["case_id"]
                    combined.extend(data)
            json_file = combined
        else:
            img_path = image_paths[0] if isinstance(image_paths, list) else image_paths
            with open(paths, "r") as f:
                data = json.load(f)
                for item in data:
                    item["volume_path"] = img_path + os.sep + item["case_id"]
            json_file = data
        return json_file

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx: int):
        max_tries = 10
        for _ in range(max_tries):
            # try:
            data = self.data_list[idx]

            img_path = data["volume_path"]
            image = self.transform(read_image(img_path))

            if len(data["local_vqa"]) == 0:
                use_global = True
            else:
                use_global = random.random() < 0.5

            if use_global:
                vqa = data["global_vqa"][0]
                q_txt = vqa["question"].rstrip()
                ans = vqa["answer"]
                ans = ", ".join(ans) if isinstance(ans, list) else ans
                if "choices" in vqa and vqa["choices"]:
                    choices = vqa["choices"]
                    q_txt = f"{q_txt} Choices: {choices}"
            else:
                locals_ = data["local_vqa"]
                root = random.choice([q for q in locals_ if q["follow_up"] == -1])
                chain = [root] + [q for q in locals_ if q["follow_up"] == root["id"]]

                q_lines, a_lines = [], []
                for i, q in enumerate(chain, start=1):
                    q_line = q["question"].rstrip()
                    if "choices" in q and q["choices"]:
                        choices = q["choices"]
                        q_line = f"{q_line} Choices: {choices}"
                    q_lines.append(f"{i}. {q_line}")
                    a_lines.append(f"{i}. {q['answer']}")

                q_txt = "\n".join(q_lines)
                ans = "\n".join(a_lines)

            conversation = [
                {
                    "role": "system",
                    "content": (
                        "You are an AI assistant acting as a radiologist tasked with "
                        "answering a multiple-choice question based on a CT scan."
                    ),
                },
                {"role": "user", "content": self.image_tokens + " " + q_txt},
            ]
            prompt = self.tokenizer.apply_chat_template(conversation, tokenize=False)

            pair = self.tokenizer(
                prompt + " " + ans,
                max_length=self.args.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            input_id, attn = pair["input_ids"][0], pair["attention_mask"][0]

            valid_len = torch.sum(attn)
            if valid_len < len(input_id):
                input_id[valid_len] = self.tokenizer.eos_token_id

            q_only = self.tokenizer(
                prompt,
                max_length=self.args.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            q_len = torch.sum(q_only["attention_mask"][0])

            label = input_id.clone()
            label[:q_len] = -100  # mask question tokens
            if self.tokenizer.pad_token_id == self.tokenizer.eos_token_id:
                label[label == self.tokenizer.pad_token_id] = -100
                if valid_len < len(label):
                    label[valid_len] = self.tokenizer.eos_token_id
            else:
                label[label == self.tokenizer.pad_token_id] = -100

            return {
                "image": image,
                "input_id": input_id,
                "label": label,
                "attention_mask": attn,
                "question": prompt,
                "answer": ans,
                "question_type": "global" if use_global else "local_chain",
            }

            # except Exception as exc:
            #     print(f"[WARN] __getitem__ failed at {idx}: {exc}")
            #     idx = random.randint(0, len(self.data_list) - 1)


class UniDatasets(Dataset):
    def __init__(self, args, tokenizer, mode="train", **kwargs):
        super(UniDatasets, self).__init__()
        cap_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in FLARECapDataset.__init__.__code__.co_varnames
        }
        vqa_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k in FLAREVQADataset.__init__.__code__.co_varnames
        }
        self.ds_list = [
            FLARECapDataset(args, tokenizer, mode=mode, **cap_kwargs),
            FLAREVQADataset(args, tokenizer, mode=mode, **vqa_kwargs),
        ]
        self.dataset = ConcatDataset(self.ds_list)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]
