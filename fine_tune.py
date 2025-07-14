"""
Filename : fine_tune_ghn.py
Author : Archit
Date Created : 7/14/2025
Description : Fine-tune GHN-3 to emit 10-class heads for all torchvision models.
Language : python3
"""
import argparse
import torch, torchvision
from torch.utils.data import DataLoader
from ppuda.config import init_config
from ghn3 import from_pretrained, Graph, Logger, custom_loader

# --------------- 1. Args -------------------
parser = argparse.ArgumentParser(description='GHN-3 fine-tuning for CIFAR-10')
parser.add_argument('--ckpt', type=str, required=True, help='Path to GHN-3 checkpoint')
parser.add_argument('--epochs', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training')
args_raw = parser.parse_args()
args = init_config(mode='train_ghn', ckpt=args_raw.ckpt, debug=0)

# --------------- 2. CIFAR-10 loader -------------------
tf = torchvision.transforms.Compose([
    torchvision.transforms.Resize(224),  # 👈 Enables all big models
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize((0.491, 0.482, 0.447),
                                     (0.247, 0.243, 0.262)),
])

trainset = torchvision.datasets.CIFAR10(root="~/data", train=True, download=True, transform=tf)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=4)

# --------------- 3. Setup GHN -------------------
device = args.device
ghn = from_pretrained(args.ckpt, debug_level=args.debug).to(device)
ghn.train(); ghn.debug_level = 0
optim = torch.optim.AdamW(ghn.parameters(), lr=2e-4, weight_decay=1e-2)
crit = torch.nn.CrossEntropyLoss()
logger = Logger(args.epochs * len(trainloader))

# --------------- 4. Valid torchvision models -------------------
ARCH_POOL = [
    lambda: torchvision.models.resnet18(num_classes=10),
    lambda: torchvision.models.resnet50(num_classes=10),
    lambda: torchvision.models.mobilenet_v2(num_classes=10),
    lambda: torchvision.models.mobilenet_v3_large(num_classes=10),
    lambda: torchvision.models.densenet121(num_classes=10),
    lambda: torchvision.models.vgg11_bn(num_classes=10),
    lambda: torchvision.models.convnext_tiny(num_classes=10),
    lambda: torchvision.models.squeezenet1_0(num_classes=10),
    lambda: torchvision.models.shufflenet_v2_x1_0(num_classes=10),
    lambda: torchvision.models.efficientnet_b0(num_classes=10),
    lambda: torchvision.models.inception_v3(num_classes=10, aux_logits=False)
]



def random_templates(k, device):
    models, graphs = [], []
    for _ in range(k):
        model = ARCH_POOL[torch.randint(len(ARCH_POOL), (1,)).item()]().to(device)
        graph = Graph(model)
        models.append(model)
        graphs.append(graph)
    return models, graphs

# --------------- 5. Fine-tune Loop -------------------
step = 0
for epoch in range(args.epochs):
    for imgs, labels in trainloader:
        imgs, labels = imgs.to(device), labels.to(device)
        optim.zero_grad()

        # meta-batch of K architectures
        templates, graphs = random_templates(8, device)
        loss_sum = 0.

        for model, graph in zip(templates, graphs):
            pred_model = ghn(model, graph, keep_grads=True)
            logits = pred_model(imgs)
            loss = crit(logits, labels)
            loss_sum += loss

        loss_sum = loss_sum / args.batch_size  # average loss over the batch
        loss_sum.backward()
        torch.nn.utils.clip_grad_norm_(ghn.parameters(), 5)
        optim.step()

        logger(step, {'loss': loss_sum.item()})
        step += 1

# --------------- 6. Save the new GHN -------------------
torch.save(ghn.state_dict(), "ghn3_cifar10.pt")
print("✅ Saved fine-tuned GHN for CIFAR-10: ghn3_cifar10.pt")
