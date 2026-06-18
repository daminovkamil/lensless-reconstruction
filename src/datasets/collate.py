import torch


def stack(key):
    return torch.stack([elem[key] for elem in dataset_items]).contiguous()


def collate_fn(dataset_items: list[dict]):
    result_batch = {
        "lensless": stack("lensless"),
        "psf": stack("psf"),
        "img_id": [elem["img_id"] for elem in dataset_items],
    }

    if "gt" in dataset_items[0]:
        result_batch["gt"] = stack("gt")

    return result_batch
