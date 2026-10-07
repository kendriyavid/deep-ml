import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def train_model(model, X_train, y_train, X_val, y_val, epochs, batch_size, lr):

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    lossobj = nn.CrossEntropyLoss()

    history = []

    for epoch in range(epochs):

        model.train()

        permuted = torch.randperm(len(y_train))
        X_permuted = X_train[permuted]
        y_permuted = y_train[permuted]

        train_loss = 0
        minibatches = 0

        for index in range(0, len(y_train), batch_size):

            X_batch = X_permuted[index:index+batch_size]
            y_batch = y_permuted[index:index+batch_size]

            optimizer.zero_grad()

            y_pred = model(X_batch)

            loss = lossobj(y_pred, y_batch)

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            minibatches += 1

        train_loss = train_loss / minibatches


        # ================= VALIDATION =================
        with torch.no_grad():

            model.eval()

            global_loss = []
            accuracy_global = []

            epochloss = 0
            minibatches = 0
            correct = 0
            total = 0

            for val_index in range(0, len(y_val), batch_size):

                X_val_batch = X_val[val_index:val_index+batch_size]
                y_val_batch = y_val[val_index:val_index+batch_size]

                y_val_pred = model(X_val_batch)

                loss = lossobj(y_val_pred, y_val_batch)

                epochloss += loss.item()
                minibatches += 1

                y_val_pred = y_val_pred.argmax(dim=1)

                correct += torch.sum(y_val_pred == y_val_batch).item()
                total += len(y_val_batch)

            val_loss = epochloss / minibatches
            val_accuracy = correct / total

            global_loss.append(val_loss)
            accuracy_global.append(val_accuracy)


        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        })

    return history