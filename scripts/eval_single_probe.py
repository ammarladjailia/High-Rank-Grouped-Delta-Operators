import os
import sys
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression

sys.path.append(os.path.abspath("."))
from scripts.compute_linear_probe import get_model, extract_features
from metrics.representation import compute_representation_metrics

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    device = "cpu"
    
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root="data", train=True, download=False, transform=transform_val)
    testset = torchvision.datasets.CIFAR10(root="data", train=False, download=False, transform=transform_val)
    
    trainloader = DataLoader(trainset, batch_size=256, shuffle=False, num_workers=4)
    testloader = DataLoader(testset, batch_size=256, shuffle=False, num_workers=4)
    
    conf = "cifar10_residual_adapter_parameter_matched"
    seed = 123
    
    model_path = f"outputs/{conf}/seed_{seed}/checkpoint_best.pt"
    print(f"Loading {model_path}...")
    
    model = get_model(f"configs/{conf}.yaml")
    state_dict = torch.load(model_path, map_location=device)["model_state"]
    model.load_state_dict(state_dict)
    
    print("Extracting train features...")
    X_train, y_train = extract_features(model, trainloader, device)
    print("Extracting test features...")
    X_test, y_test = extract_features(model, testloader, device)
    
    print("Training Logistic Regression...")
    clf = LogisticRegression(max_iter=1000, n_jobs=-1)
    clf.fit(X_train, y_train)
    acc = clf.score(X_test, y_test)
    
    print("Computing Representation Metrics...")
    rep_metrics = compute_representation_metrics(torch.tensor(X_test), torch.tensor(y_test))
    
    print("\n=== FINAL RESULTS (SEED 123) ===")
    print(f"Linear Probe Acc : {acc*100:.2f}%")
    print(f"Fisher Ratio     : {rep_metrics['fisher_ratio']:.4f}")
    print(f"Marge Intra (Sw) : {rep_metrics['s_w']:.4f}")
    print(f"Marge Inter (Sb) : {rep_metrics['s_b']:.4f}")

if __name__ == "__main__":
    main()
