import gc
import urllib.request

import torch
from torchvision.io import read_image
import torch.nn.functional as F
import torchvision.transforms.v2.functional as TF
import torchvision.transforms.v2 as v2
import pytest

from seg import all_models, dice_loss, f1_score
ALL_MODELS = list(all_models())


augment = v2.Compose([
    v2.ColorJitter(
        brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2
    ),
    v2.GaussianNoise()
])
augment_spatial = v2.Compose([
    v2.RandomRotation(10),
    v2.RandomPerspective(distortion_scale=0.3, p=0.3),
    v2.RandomHorizontalFlip(p=0.5)
])


@pytest.fixture(autouse=True)
def base_fixture():
    free, avail = (x/(1024*1024*1024) for x in torch.cuda.mem_get_info())
    alloc = torch.cuda.memory_allocated(0)/(1024*1024*1024)
    cur_rsrv = torch.cuda.memory_reserved(0)/(1024*1024*1024)
    max_rsrv = torch.cuda.max_memory_reserved(0)/(1024*1024*1024)
    # print(
    #     f"\nbefore: free: {free:.1f} avail: {avail:.1f} "
    #     f"alloc: {alloc:.1f} cur_rsrv: {cur_rsrv:.1f} "
    #     f"max_rsrv: {max_rsrv:.1f}"
    # )


    yield
    free, avail = (x/(1024*1024*1024) for x in torch.cuda.mem_get_info())
    alloc = torch.cuda.memory_allocated(0)/(1024*1024*1024)
    cur_rsrv = torch.cuda.memory_reserved(0)/(1024*1024*1024)
    max_rsrv = torch.cuda.max_memory_reserved(0)/(1024*1024*1024)
    # print(
    #     f"\nafter: free: {free:.1f} avail: {avail:.1f} "
    #     f"alloc: {alloc:.1f} cur_rsrv: {cur_rsrv:.1f} "
    #     f"max_rsrv: {max_rsrv:.1f}"
    # )
    torch.cuda.empty_cache()
    free, avail = (x/(1024*1024*1024) for x in torch.cuda.mem_get_info())
    alloc = torch.cuda.memory_allocated(0)/(1024*1024*1024)
    cur_rsrv = torch.cuda.memory_reserved(0)/(1024*1024*1024)
    max_rsrv = torch.cuda.max_memory_reserved(0)/(1024*1024*1024)
    # print(
    #     f"\tafter: free: {free:.1f} avail: {avail:.1f} "
    #     f"alloc: {alloc:.1f} cur_rsrv: {cur_rsrv:.1f} "
    #     f"max_rsrv: {max_rsrv:.1f}"
    # )



@pytest.mark.parametrize("module,model", ALL_MODELS)
def test_url_is_valid(module, model):
    if module.models[model] is None:
        pytest.skip("No pretrained weights available")
    if not hasattr(module, "get_url"):
        pytest.skip("Manual download not needed")
    url = module.get_url(*module.models[model])
    with urllib.request.urlopen(url) as res:
        assert res.status == 200
        assert res.length > 1e6


def load_image_and_annotation(i):
    free, avail = (x/(1024*1024*1024) for x in torch.cuda.mem_get_info())
    alloc = torch.cuda.memory_allocated(0)/(1024*1024*1024)
    cur_rsrv = torch.cuda.memory_reserved(0)/(1024*1024*1024)
    max_rsrv = torch.cuda.max_memory_reserved(0)/(1024*1024*1024)
    print(
        f"\nload_img: free: {free:.1f} avail: {avail:.1f} "
        f"alloc: {alloc:.1f} cur_rsrv: {cur_rsrv:.1f} "
        f"max_rsrv: {max_rsrv:.1f}"
    )

    image = TF.convert_image_dtype(
        read_image(f"image-{i}.png").unsqueeze(0),
        torch.float32
    ).cuda()
    target = read_image(f"annotation-{i}.png").unsqueeze(0)
    target = (target.to(torch.uint8) > 0).float().cuda()
    return image, target


@pytest.fixture(scope="session")
def sample1():
    return load_image_and_annotation("1")


@pytest.fixture(scope="session")
def sample2():
    return load_image_and_annotation("2")


@pytest.fixture(scope="session")
def sample3():
    return load_image_and_annotation("3")


@pytest.mark.parametrize("pretrained", [False, True])
@pytest.mark.parametrize("val", [False, True])
@pytest.mark.parametrize("module,model_name", ALL_MODELS)
def test_convergence(
    module, model_name, pretrained, val, sample1, sample2, sample3
):
    lr = 1e-5 if module.__name__ == "sam" else 1e-4
    warmup_steps = 100
    if module.models[model_name] is None and pretrained:
        pytest.skip("No pretrained weights available")
    model = module.new(model_name, pretrained).train().cuda()
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    step = 1
    last_improved = 1
    best_f1 = 0.
    lr_incr = ((9 * lr) / warmup_steps)
    train_imgs, train_masks = map(torch.cat, zip(sample1, sample2))
    while True:
        img = augment(train_imgs)
        aug = augment_spatial(torch.cat([img, train_masks], dim=1))
        img, targ = aug[:, :3].contiguous(), aug[:, -1:].contiguous()
        img = TF.normalize(
            img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        out = model(img)
        loss = dice_loss(out, targ) \
            + F.binary_cross_entropy_with_logits(out, targ)
        optim.zero_grad()
        loss.backward()
        optim.step()

        img, targ = sample3 if val else (train_imgs, train_masks)
        img = TF.normalize(
            img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        model.eval()
        with torch.no_grad():
            out = model(img)
        f1 = f1_score(out, targ).item()

        if val and f1 > 0.4 or f1 > 0.6:
            print(" at step", step, end=" ")
            break
        elif f1 > best_f1:
            last_improved = step
            best_f1 = f1
        elif (step - last_improved) > 200 or step >= 2000:
            pytest.fail(f"at step {step} f1: {best_f1:.5f}")

        if val and step <= warmup_steps:
            for param_group in optim.param_groups:
                param_group["lr"] += lr_incr

        step += 1
    del model, optim, train_imgs, train_masks, img, targ
