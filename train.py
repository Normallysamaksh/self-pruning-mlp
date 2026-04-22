import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from model import PrunableMLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --------------------
# DATA
# --------------------
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False)


# --------------------
# TRAIN FUNCTION
# --------------------
def train(lambda_val=1e-4, epochs=5):
    model = PrunableMLP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            classification_loss = criterion(outputs, labels)

            # Sparsity loss (L1 on gates)
            gates = model.get_all_gates()
            sparsity_loss = torch.sum(gates)

            loss = classification_loss + lambda_val * sparsity_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss:.2f}")

    return model


# --------------------
# EVALUATION
# --------------------
def evaluate(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)

            correct += (predicted == labels).sum().item()
            total += labels.size(0)

    accuracy = 100 * correct / total

    # compute sparsity
    gates = model.get_all_gates()
    sparsity = (gates < 1e-2).float().mean().item() * 100

    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Sparsity: {sparsity:.2f}%")

    return accuracy, sparsity


# --------------------
# MAIN
# --------------------
if __name__ == "__main__":
    model = train(lambda_val=1e-4, epochs=5)
    evaluate(model)

    torch.save(model.state_dict(), "model.pth")
