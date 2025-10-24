import json
import yaml
import torch
from torchvision.utils import save_image
from forward_operator import get_operator
from data import get_dataset
from pathlib import Path
from omegaconf import OmegaConf
import hydra
import numpy as np
from PIL import Image
import os
from eval import get_eval_fn, Evaluator


def safe_dir(dir):
    """
    get (or create) a directory
    """
    if not Path(dir).exists():
        Path(dir).mkdir(parents=True, exist_ok=True)
    return Path(dir)


def norm(x):
    """
    normalize data to [0, 1] range
    """
    return (x * 0.5 + 0.5).clip(0, 1)


def tensor_to_pils(x):
    """
    [B, C, H, W] tensor -> list of pil images
    """
    pils = []
    for x_ in x:
        np_x = norm(x_).permute(1, 2, 0).cpu().numpy() * 255
        np_x = np_x.astype(np.uint8)
        pil_x = Image.fromarray(np_x)
        pils.append(pil_x)
    return pils


@hydra.main(version_base='1.3', config_path='configs', config_name='default.yaml')
def main(args):
    # fix random seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.cuda.set_device('cuda:{}'.format(args.gpu))

    print(yaml.dump(OmegaConf.to_container(args, resolve=True), indent=4))

    # get data
    dataset = get_dataset(**args.data)
    total_number = len(dataset)
    images = dataset.get_data(total_number, 0)

    # get forward operator & measurement
    task_group = args.task[args.task_group]
    forward_operator = get_operator(**task_group.operator)
    y = forward_operator.measure(images)

    # get baseline operator (matching the forward operator)
    baseline_operator_name = task_group.operator.name + '_baseline'
    baseline_operator_config = dict(task_group.operator)
    baseline_operator_config['name'] = baseline_operator_name
    baseline_operator = get_operator(**baseline_operator_config)
    
    # apply baseline reconstruction
    reconstructed = baseline_operator(y)

    # get evaluator (same as posterior_sample.py)
    eval_fn_list = []
    for eval_fn_name in args.eval_fn_list:
        eval_fn_list.append(get_eval_fn(eval_fn_name))
    evaluator = Evaluator(eval_fn_list)

    # evaluate baseline reconstruction
    # Add batch dimension for evaluation: [1, B, C, H, W]
    baseline_samples = reconstructed.unsqueeze(0)  # Shape: [1, B, C, H, W]
    results = evaluator.report(images, y, baseline_samples)

    # log hyperparameters and configurations
    os.makedirs(args.save_dir, exist_ok=True)
    save_dir = safe_dir(Path(args.save_dir))
    root = safe_dir(save_dir / args.name)
    with open(str(root / 'config.yaml'), 'w') as file:
        yaml.safe_dump(OmegaConf.to_container(args, resolve=True), file, default_flow_style=False, allow_unicode=True)

    # save evaluation results
    markdown_text = evaluator.display(results)
    with open(str(root / 'eval.md'), 'w') as file:
        file.write(markdown_text)
    json.dump(results, open(str(root / 'metrics.json'), 'w'), indent=4)
    print(markdown_text)

    # save individual reconstructed images
    pil_image_list = tensor_to_pils(reconstructed)
    image_dir = safe_dir(root / 'reconstructed')
    for idx in range(len(pil_image_list)):
        image_path = image_dir / '{:05d}.png'.format(idx)
        pil_image_list[idx].save(str(image_path))

    # save original images for comparison
    pil_original_list = tensor_to_pils(images)
    original_dir = safe_dir(root / 'original')
    for idx in range(len(pil_original_list)):
        image_path = original_dir / '{:05d}.png'.format(idx)
        pil_original_list[idx].save(str(image_path))

    # save measurements
    pil_measurement_list = tensor_to_pils(y)
    measurement_dir = safe_dir(root / 'measurements')
    for idx in range(len(pil_measurement_list)):
        image_path = measurement_dir / '{:05d}.png'.format(idx)
        pil_measurement_list[idx].save(str(image_path))

    # log grid results (original, measurement, reconstructed)
    stack = torch.cat([images, y, reconstructed])
    save_image(stack * 0.5 + 0.5, fp=str(root / 'grid_results.png'), nrow=total_number)

    print(f'Baseline reconstruction complete for {args.name}!')
    print(f'Results saved to: {root}')


if __name__ == '__main__':
    main()