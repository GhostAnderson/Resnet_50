from torchvision import transforms

BATCH_SIZE = 128

import dataset as ds
transform_train = transforms.Compose(
    [
        transforms.Resize(400),
        transforms.RandomCrop(400,40),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ]
)
transform_test = transforms.Compose(
    [
        transforms.Resize(400),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-t",'--train',help="Train a Resnet-50 Model with specified Dataset given by -d",action='store_true',default=False)
    group.add_argument('-u','--test',help='Test a Resnet-50 model with specified Network given by -n and specified test set given by --testset',action='store_true',default=False)
    group.add_argument('-c','--continued',help='Continue Training with specified semi-trained Network given by --pretrained_net and specified pretrain epoch num given by --pretrain_epoches',default=False)

    parser.add_argument('--cuda',default=False,help="Decide whether to use NVIDIA Gpu to accelerate",action='store_true')
    parser.add_argument('-e','--epoch',type=int,help='Epoch nums',default=50)
    parser.add_argument('-o','--output',help="A absolute path of where you want the model and files to be saved in")
    parser.add_argument('--pretrained_net',help="A absolute path of a pretrained network")
    parser.add_argument('--pretrained_epoches',type=int,help='Number of the already trained epoches of the given pretrained network')
    parser.add_argument('-d','--dataset',help='A absolute path of a dataset, with sub-folder: /train, /test')
    parser.add_argument('-n','--network',help='A absolute path of a pretrained Resnet-50 network')
    parser.add_argument('-s','--testset',help='A absolute path of a test set')

    args = parser.parse_args()
    if args.train:
        import torch
        import torchvision.models.resnet as resnet
        trainset = ds.Dataset(path=args.dataset,transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

        testset = ds.Dataset(path=args.dataset,transform=transform_test,train=False)
        testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False)

        device = torch.device("cuda" if args.cuda else "cpu")
        net = resnet.resnet50(pretrained=False)

        fc_input = net.fc.in_features
        net.fc = torch.nn.Linear(fc_input,2)
        net.to(device)

        crit = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(net.parameters(),lr=0.001,weight_decay=5e-4)

        import os

        if not os.path.exists(args.output):
            os.makedirs(args.output)

        for epoch in range(args.epoch):
            print("Now Training Epoch:",epoch)
            loss_val = 0.0
            total = 0
            correct = 0
            for i,data in enumerate(trainloader):
                net.train()
                input_,tag = data
                input_,tag = input_.to(device),tag.to(device)

                optim.zero_grad()

                outputs = net(input_)
                outputs = torch.tensor(outputs)
                loss = crit(outputs,tag)
                loss.backward()
                optim.step()

                loss_val += loss.item()
                _,predicted = torch.max(outputs.data,1)
                total+=tag.size(0)
                correct += predicted.eq(tag.data).cpu().sum()
                if(epoch%5 == 0):
                    with open(args.output+'/output.txt','a') as f:
                        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' % (epoch + 1, (i + 1 + epoch * BATCH_SIZE), loss_val / (i + 1), 100. * correct / total))
                        f.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% \n'
                                % (epoch + 1, (i + 1 + epoch * BATCH_SIZE), loss_val / (i + 1), 100. * correct / total))
                        f.flush()
            
            print("Testing Network")
            with torch.no_grad():
                correct = 0
                total = 0
                for data in testloader:
                    net.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                acc = 100. * correct / total
                print('ACC:', acc)
                print('Saving model......')
                torch.save(net.state_dict(), args.output+'/net_'+(epoch+1)+'.pth')
                with open(args.output+'/acc.txt','a') as f:
                    f.write('EPOCH=',epoch+1,'ACC=',acc)
                    f.flush()


    elif args.continued:
        import torch
        import torchvision.models.resnet as resnet
        trainset = ds.Dataset(transform=transform_train)
        trainloader = torch.utils.data.DataLoader(trainset,batch_size=BATCH_SIZE,shuffle=True,num_workers=2)

        testset = ds.Dataset(transform=transform_test,train=False)
        testloader = torch.utils.data.DataLoader(testset,batch_size=100,shuffle=False)
        device = torch.device("cuda" if args.cuda else "cpu")

        net = resnet.resnet50(pretrained=False)

        fc_input = net.fc.in_features
        net.fc = torch.nn.Linear(fc_input,2)

        net.load_state_dict(args.pretrained_net)
        crit = torch.nn.CrossEntropyLoss()
        optim = torch.optim.Adam(lr=0.001,weight_decay=5e-4)

        for epoch in range(args.pretrain_epoches,args.epoch):
            print("Now Training Epoch:",epoch)
            loss_val = 0.0
            total = 0
            correct = 0
            for i,data in enumerate(trainloader):
                net.train()
                input_,tag = data
                input_,tag = input_.to(device),tag.to(device)

                optim.zero_grad()

                outputs = net(input_)
                outputs = torch.tensor(outputs)
                loss = crit(outputs,tag)
                loss.backward()
                optim.step()

                loss_val += loss.item()
                _,predicted = torch.max(outputs.data,1)
                total+=tag.size(0)
                correct += predicted.eq(tag.data).cpu().sum()
                if(epoch%5 == 0):
                    with open(args.output+'/output.txt','a') as f:
                        print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% ' % (epoch + 1, (i + 1 + epoch * BATCH_SIZE), loss_val / (i + 1), 100. * correct / total))
                        f.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% \n'
                                % (epoch + 1, (i + 1 + epoch * BATCH_SIZE), loss_val / (i + 1), 100. * correct / total))
                        f.flush()
            
            print("Testing Network")
            with torch.no_grad():
                correct = 0
                total = 0
                for data in testloader:
                    net.eval()
                    images, labels = data
                    images, labels = images.to(device), labels.to(device)
                    outputs = net(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum()
                acc = 100. * correct / total
                print('ACC:', acc)
                print('Saving model......')
                torch.save(net.state_dict(), args.output+'/net_'+(epoch+1)+'.pth')
                with open(args.output+'/acc.txt','a') as f:
                    f.write('EPOCH=',epoch+1,'ACC=',acc)
                    f.flush()
    elif args.test:
        import torch
        import torchvision.models.resnet as resnet
    else:
        parser.print_help()