---
title: "Training on Large Batches"
date: 2019-06-17T12:21:44-04:00
tags: ['deep learning', 'training tricks']
categories: ['deep learning']
---

According to Sebastian Ruder's blog [post](http://ruder.io/nlp-imagenet/), the ImageNet moment of NLP has arrived. Especially, models like e.g <span style="color:blue">BERT</span>, <span style="color:blue">ELMO</span>, <span style="color:blue">UlMFIT</span>, <span style="color:blue">Open-GPT</span>, <span style="color:blue">Transformer-XL</span> have become the main stream choice of most downstream NLP tasks. However, it is still quite difficult to conduct a transfer learning task from a pre-training model such as 345 millions parameter <span style="color:blue">Open-GPT2</span> with a large batch say 256. Certainly, if your NLP tasks is with small datasets, and you are able to use batch_size = 8, and wait for 2-4 hours to do it, that is none of the cases I am talking about here. In ***Leafy*** projects, I was mainly dealing with more than 300GB text data, so feeding them into a sequence model takes a lot of work. Here, I am going to share you a post that how to train a neural network with large batches with different tricks. 


## In particular, this post includes:

1. Train a model on a single or multi GPU server with batches larger than the GPUs memory or when even a single training sample won’t fit 
2. Most efficient use of a multi-GPU machine
3. The simplest way to train a model using several machines in a distributed setup.


## Simplest trick: gradient accumulation.


Here, I mainly use ***Pytorch*** as the backend framework due to its simplicity and its advantage, dynamic language programming. (Of course, you can consider eager mode on Tensorflow as dynamic, but Pytorch is natural). Comparing to standard optimization which looks like below,

````python                           
for i, (inputs, label) in enumerate(training_set):
	outputs = model(inputs)               
	loss = criterion(outputs labels) 

	# backward
	optimizer.zero_grad() # then reset gradients tensor
	loss.backward()                           
	optimizer.step()         				  
````    

the <span style="color:blue">gradient accumulation</span> looks like below:

````python
for i, (inputs, labels) in enumerate(training_set):
	outputs = model(inputs)               
	loss = criterion(outputs labels) 

	# backward
	# loss normalization 
	loss = loss / accumulation_steps # taking average loss over accumulated steps
	# back propagation
	loss.backward()                                 
	# update parameters
	if (i+1) % accumulation_steps == 0:             
		optimizer.step()  # update all parameters
		optimizer.zero_grad() # then reset gradients tensor
                       
````

In summary, gradient accumulation is essentially accumulating gradients in K batches, and then update and reset. It works for large batch size since it intuitively increases 1 batch to K batches. This is a good trick for limited GPU memory usage. Note that learning rate needs to be larger too related to the choice of K accumulation steps.


## Torch.utils.checkpoint

When a model is too large for a single GPU, is there a way to implement the training? <span style="color:red">Yes, but it has a cost. It costs computing over GPU usage</span>. The trick is to use the ***checkpoint*** function in Pytorch. If you would like to know what exactly it does in details, check [here](https://pytorch.org/docs/stable/checkpoint.html), the official documentation, for explanation. Here is an example, say u have a 1000 layers model. It is too large to fit in GPU, so what `torch.utils.checkpoint` can do is to split the model into N segments. Intuitively, backward propagation only needs to perform over each segment at time, so it doest not need to run over all the parameters of a model once per time. The code to achieve it is like below

````python
# pseudo 1000 layers model
layers = [nn.Linear(10, 10) for _ in range(1000)]
model = nn.Sequential(*layers)

from torch.utils.checkpoint import checkpoint_sequential

# split into two segments
num_segments = 2
x = checkpoint_sequential(model, num_segments, input)
x.sum().backward() 
````
This concept has interactions with meta-learning, and it has been experimentally proved to be useful for sequence model.


## Multi-GPU Utility Maximization

Well, if you are lucky to have at least 4 GPUs in the same time, parallel computing is a great choice to yield faster training. Here, The top option is always the data parallelism. 

````python
parallel_model = torch.nn.DataParallel(model) 
predictions = parallel_model(inputs)          
loss = criterion(predictions, labels)     
loss.backward()                               
optimizer.step()                              
predictions = parallel_model(inputs)          
````

However, there is only one issue, the imbalanced GPU usage in the forward passing. A better illustration I found from *zhihu* is like this 

![](https://cdn-images-1.medium.com/max/2400/1*FpDHkWJhkLL7KxU01Lf9Lw.png)


As step 4 of the Forward pass (top-right) shows the results of *ALL* the parallel computations are gathered on GPU-1. When training a language model, it is very painful though. A quick example can be BERT base-chinese model, which has max_len = 512, vocab_len = 21128. If we do batch_size = 32 (4 bytes to store each element) in memory, so the model takes about 1,44 GB. We need to double that to store the associated gradient tensors, our model output thus requires 2,88 GB of memory! It is a quite big portion of a typical 8 GB GTX 1080 memory and means that GPU-1 will be over-used so it limits the effect of parallelism. 

What can we do then? 

## Balance load on multi-GPU machine

There are two main solution to the imbalanced GPU usage issue:

1. computing the loss in the forward pass of your model
2. computing the loss in a parallel fashion


Thanks to [张航](https://hangzhang.org/), he solved the problems simply by creating his own version of data parallelism. If you are interested, download [here](https://gist.github.com/thomwolf/7e2407fbd5945f07821adae3d9fd1312), and then import them like normal torch.nn.utils.DataParallel. 

````python
from parallel import DataParallelModel, DataParallelCriterion

parallel_model = DataParallelModel(model)             # solution 1
parallel_loss  = DataParallelCriterion(loss_function) # solution 2
predictions = parallel_model(inputs)      
loss = parallel_loss(predictions, labels) 
loss.backward()                           
optimizer.step()                          
predictions = parallel_model(inputs)      
````

The difference between <span style="color:red">DataParallelModel</span> and <span style="color:red">torch.nn.DataParallel</span> is just that the output of the forward pass (predictions) is not gathered on *GPU-1* and is thus a tuple of multiple gpu, *n_gpu*, tensors, each tensor being located on a respective GPU. 

The DataParallelCriterion takes input the tuple of n_gpu tensors and the target labels tensor. It computes the loss function in parallel on each GPU, splitting the target label tensor the same way the model input was chunked by DataParallel. A related illustration become like below 

![](https://cdn-images-1.medium.com/max/1600/1*F6SXjBp6BCoFTZ26RKnz9A.png)


## Distributed Computing

Well, if you are really lucky, you can even try distributed computing over severs and each server is a mulit-GPU device. In this case, you can try a even larger batch size. In case, readers do not know what distributed computing is, so I am here to explain a little bit. A simple way to understand the distributed computing is that you are training a model in a synchronized way by calling independent python training script on each node (sever), and each training script has its own optimizer and python interpreter. Simply put, the workflow is changed. In command line, training a CNN model on MNIST dataset looks like 

````bash
python mnsit.py --init-method tcp://192.168.54.179:22225 --rank 0 --world-size 2
python mnsit.py --init-method tcp://192.168.54.179:22225 --rank 1 --world-size 2
python mnsit.py --init-method tcp://192.168.54.179:22225 --rank 2 --world-size 2
python mnsit.py --init-method tcp://192.168.54.179:22225 --rank 3 --world-size 2
````

A good code to implement is like below:

````python
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import time

import torch.nn.parallel
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data 
import torch.utils.data.distributed
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=1024, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--init-method', type=str, default='tcp://127.0.0.1:23456')
parser.add_argument('--rank', type=int)
parser.add_argument('--world-size',type=int)

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

# initialization
dist.init_process_group(init_method=args.init_method,backend="gloo",world_size=args.world_size,rank=args.rank,group_name="pytorch_test")

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

train_dataset=datasets.MNIST('data', train=True, download=True,
               transform=transforms.Compose([
                   transforms.ToTensor(),
                   transforms.Normalize((0.1307,), (0.3081,))
               ]))

# distirbuted sampling
train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

train_loader = torch.utils.data.DataLoader(train_dataset,
    batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()
if args.cuda:
    # to different cuda devices
    model.cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)
    # model = torch.nn.DataParallel(model,device_ids=[0,1,2,3]).cuda()
    # model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

tot_time=0;

for epoch in range(1, args.epochs + 1):
    # set epoch for gathering same epoch info over synchronized jobs
    train_sampler.set_epoch(epoch)
    start_cpu_secs = time.time()
    #long running
    train(epoch)
    end_cpu_secs = time.time()
    print("Epoch {} of {} took {:.3f}s".format(
        epoch , args.epochs , end_cpu_secs - start_cpu_secs))
    tot_time+=end_cpu_secs - start_cpu_secs
    test()

print("Total time= {:.3f}s".format(tot_time))
````

## Conclusion:

1. If you want to try a larger batch size in one GPU machine, try <span style="color:blue">gradient accumulation</span>;
2. If you want to try a very very deep model on one GPU machine and want to fit samples in sequence model, try <span style="color:red">gradient checkpoint</span>;
3. If you have multi-GPU, try <span style="color:red">DataParallel</span> from Pytorch or provided link;
4. If you are lucky, servers and multi-gpu machine, and want to try batch_size like 10000, try <span style="color:blue">Distributed Computing</span>.


Good luck for all of you to any case you would like to implement deep learning algorithms. 
