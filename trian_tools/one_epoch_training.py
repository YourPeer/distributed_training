import torch
def train_one_epoch(model,optimizer,criterion,train_dataloader,test_dataloader,epoch):
    model.train()
    for batch_idx,(data,target) in enumerate(train_dataloader):
        data=data.cuda(non_blocking=True)
        target=target.cuda(non_blocking=True)

        output=model(data)
        loss=criterion(output,target)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        optimizer.average()
        if batch_idx%5==0:
            accuracy=eval_one_epoch(model,test_dataloader)
            print(accuracy)
    return accuracy

def eval_one_epoch(model,test_dataloader):
    model.eval()
    correct = 0
    total = 0
    for data in test_dataloader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        with torch.no_grad():
            outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    model.train()
    return 100 * correct // total