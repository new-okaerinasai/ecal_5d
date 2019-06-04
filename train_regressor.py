import numpy as np

print("haha")
print("lmao")
import os
gpu_id = 0
os.environ['CUDA_VISIBLE_DEVICES'] = '%d' % gpu_id

from torchvision.models import vgg19_bn, vgg19
from torch import nn
import torch
from torch import optim
from torch.autograd import Variable
import nirvana_dl

input_path = nirvana_dl.input_data_path() 
data_train = np.load(os.path.join(input_path, 'data_train.npy'))
data_test = np.load(os.path.join(input_path, 'data_test.npy'))
target_train = np.load(os.path.join(input_path, 'target_train.npy'))
target_test = np.load(os.path.join(input_path, 'target_test.npy'))

N_train = data_train.shape[0]
N_test = data_test.shape[0]

net = vgg19(num_classes=5, init_weights=True)
net.features[0] = nn.Conv2d(1, 64, 3, 1, 1)
for i, f in enumerate(net.features):
    if isinstance(net.features[i], nn.ReLU):
        net.features[i] = nn.LeakyReLU(0.2, True)
    if isinstance(net.features[i], nn.MaxPool2d):
        net.features[i] = nn.AvgPool2d(2)
net.classifier = nn.Linear(512, 5)

criterion = nn.SmoothL1Loss(reduce=False)
optimizer = optim.SGD(net.parameters(), lr=1e-2, momentum=0.9)
scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

t_data_train = torch.from_numpy(data_train.astype('float32'))
t_data_test = torch.from_numpy(data_test.astype('float32'))
t_target_train = torch.from_numpy(target_train.astype('float32'))
t_target_test = torch.from_numpy(target_test.astype('float32'))

n_epoch = 500
batch_size = 256
epoch_len = 500

net = net.cuda()

min_loss = +np.inf

for e in range(1, n_epoch+1):
    net.train()
    for i in range(epoch_len):
        idx = torch.randint(0, N_train, (batch_size,)).type(torch.LongTensor)
        input = Variable(t_data_train[idx].cuda())
        target = Variable(t_target_train[idx].cuda())
        loss = criterion(net(input), target).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    net.eval()
    mean_loss = []
    for i in range(t_data_test.shape[0] // 100):
        input = Variable(t_data_test[i*100:(i+1)*100].cuda())
        print(input.shape)
        target = Variable(t_target_test[i*100:(i+1)*100].cuda())
#        y_out = deprocess(net(input).detach().cpu().numpy())
#        y_gt = deprocess(target.detach().cpu().numpy())
        y_out = net(input).detach().cpu().numpy()
        y_gt = target.detach().cpu().numpy()
        mean_loss += [np.abs(y_out - y_gt).mean(0, keepdims=True)]
    mean_loss = np.concatenate(mean_loss).mean(0)
    print(e, mean_loss)
    loss = mean_loss.mean()
    scheduler.step(loss)
    if loss < min_loss:
        print('saved', 'lr: %f' % (1e-2 * (0.99)**e))
        torch.save(net.cpu(), os.path.join(nirvana_dl.snapshot.get_snapshot_path(), 'regressor_vgg_plain.pkl'))
        net.cuda()
        min_loss = loss
