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
    from model import NormalMLP

    print("\n===== TRAINING NORMAL MODEL =====")
    normal_model = NormalMLP().to(device)
    optimizer = optim.Adam(normal_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # train normal model
    for epoch in range(10):
        normal_model.train()
        total_loss = 0

        for inputs, labels in trainloader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = normal_model(inputs)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"[Normal] Epoch {epoch+1}, Loss: {total_loss:.2f}")

    normal_acc, _ = evaluate(normal_model)

    print("\n===== TRAINING PRUNING MODEL =====")
    pruned_model = train(lambda_val=1e-4, epochs=10)
    pruned_acc, pruned_sparsity = evaluate(pruned_model)

    print("\n===== FINAL BENCHMARK =====")
    print(f"Normal Model Accuracy: {normal_acc:.2f}%")
    print(f"Pruned Model Accuracy: {pruned_acc:.2f}%")
    print(f"Pruned Model Sparsity: {pruned_sparsity:.2f}%")
