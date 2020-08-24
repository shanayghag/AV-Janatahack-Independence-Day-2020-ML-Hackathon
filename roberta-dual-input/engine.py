import torch
import tqdm

def loss_fn(outputs, targets):
    return torch.nn.BCEWithLogitsLoss()(outputs, targets)

def train(epoch, model, training_loader, device, optimizer, scheduler):
    model.train()
    for _,data in tqdm(enumerate(training_loader, 0), total=len(training_loader)):
        ids_1 = data['ids_abstract'].to(device, dtype = torch.long)
        mask_1 = data['mask_abstract'].to(device, dtype = torch.long)

        ids_2 = data['ids_title'].to(device, dtype = torch.long)
        mask_2 = data['mask_title'].to(device, dtype = torch.long)

        targets = data['targets'].to(device, dtype = torch.float)

        outputs = model(ids_1, mask_1, ids_2, mask_2)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _%1000==0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

def validation(epoch, model,testing_loader,device):
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in tqdm(enumerate(testing_loader, 0), total=len(testing_loader)):
            ids_1 = data['ids_abstract'].to(device, dtype = torch.long)
            mask_1 = data['mask_abstract'].to(device, dtype = torch.long)
            
            ids_2 = data['ids_title'].to(device, dtype = torch.long)
            mask_2 = data['mask_title'].to(device, dtype = torch.long)

            targets = data['targets'].to(device, dtype = torch.float)
            outputs = model(ids_1, mask_1, ids_2, mask_2)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets