import os
import sys


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from ultralytics import YOLO
import detection_2d.compat  # Registers tasks.detection_2d -> detection_2d shims.


def is_lora_conv2d(module):
    return (
        module.__class__.__name__ == "LoRAConv2d"
        and hasattr(module, "lora_A")
        and hasattr(module, "lora_B")
        and hasattr(module, "scaling")
        and hasattr(module, "weight")
        and hasattr(module, "in_channels")
        and hasattr(module, "out_channels")
    )


def count_lora_conv2d(model):
    return sum(1 for module in model.modules() if is_lora_conv2d(module))


def bake_lora_for_inference(yolo):
    import torch

    before = count_lora_conv2d(yolo.model)
    print(f"YOLO-loaded model has {before} LoRAConv2d layers before bake.")

    baked = 0
    for parent_module in list(yolo.model.modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not is_lora_conv2d(child_module):
                continue
            with torch.no_grad():
                lora_weight = (child_module.lora_B @ child_module.lora_A).view(
                    child_module.weight.shape
                ) * child_module.scaling
                new_conv = torch.nn.Conv2d(
                    in_channels=child_module.in_channels,
                    out_channels=child_module.out_channels,
                    kernel_size=child_module.kernel_size,
                    stride=child_module.stride,
                    padding=child_module.padding,
                    dilation=child_module.dilation,
                    groups=child_module.groups,
                    bias=child_module.bias is not None,
                    padding_mode=child_module.padding_mode,
                )
                new_conv.weight.data = child_module.weight.data + lora_weight
                if child_module.bias is not None:
                    new_conv.bias.data = child_module.bias.data.clone()
            setattr(parent_module, child_name, new_conv)
            baked += 1

    after = count_lora_conv2d(yolo.model)
    print(f"Baked {baked} LoRAConv2d layers. Remaining LoRAConv2d: {after}.")
    return baked


def inspect_checkpoint(model_path):
    import torch
    import detection_2d.models.lora as lora_mod

    print("Pre-loading checkpoint directly to inspect:")
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    model_obj = ckpt.get("ema") or ckpt.get("model")
    conv2d_count = sum(1 for _, mod in model_obj.named_modules() if isinstance(mod, torch.nn.Conv2d))
    lora_count = sum(1 for _, mod in model_obj.named_modules() if is_lora_conv2d(mod))
    print(f"Direct loaded model has {conv2d_count} torch.nn.Conv2d instances.")
    print(f"Direct loaded model has {lora_count} LoRAConv2d-like instances.")

    print("Class of LoRAConv2d in checkpoint:")
    for _, mod in model_obj.named_modules():
        if "LoRAConv2d" in str(type(mod)):
            print(f"  Type: {type(mod)}")
            print(f"  Is lora_mod.LoRAConv2d? {isinstance(mod, lora_mod.LoRAConv2d)}")
            print(f"  Is LoRAConv2d-like? {is_lora_conv2d(mod)}")
            break


def main():
    model_path = os.path.join(script_dir, "student_lora_best.pt")
    if not os.path.exists(model_path):
        print(f"Model file {model_path} not found.")
        return

    print(f"Loading model from {model_path}...")
    inspect_checkpoint(model_path)

    model = YOLO(model_path)
    print("Baking LoRA weights directly from the checkpoint-loaded YOLO model...")
    bake_lora_for_inference(model)

    print("Running validation inference...")
    try:
        data_yaml = os.path.join(project_root, "datasets", "URPC2020.yaml")
        if not os.path.exists(data_yaml):
            print(f"Dataset config not found at {data_yaml}. Trying fallback...")
            data_yaml = os.path.join(project_root, "datasets", "URPC2020", "data.yaml")

        metrics = model.val(data=data_yaml, half=False)
        print("Validation metrics:")
        print(f"mAP50: {metrics.box.map50}")
        print(f"mAP50-95: {metrics.box.map}")
    except Exception as exc:
        print(f"Error during validation: {exc}")
        print("Make sure the dataset configuration file is present in datasets/.")


if __name__ == "__main__":
    main()
