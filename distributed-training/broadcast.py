import torch
import time


def check_broadcast(x, a, b, c, d):
    return x.max().item() == a.max().item() == b.max().item() == c.max().item() == d.max().item()


def broadcast1(x, a, b, c, d):
    # For example, use a.copy_(x) to copy the data from x to a
    a.copy_(x)
    b.copy_(x)
    c.copy_(x)
    d.copy_(x)


def broadcast2(x, a, b, c, d):
    a.copy_(x)
    b.copy_(a)
    c.copy_(a)
    d.copy_(a)


def broadcast3(x, a, b, c, d):
    # x_pinned = x.pin_memory()
    a.copy_(x, non_blocking=True)

    for tensor in [b, c, d]:
        device = tensor.device
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            tensor.copy_(a, non_blocking=True)
        # tensor.copy_(a, non_blocking=True)

def timeit(boradcast):

    if boradcast.__name__ == 'broadcast3':
        x = torch.randn((100000, 10000), pin_memory=True)
    else:
        x = torch.randn((100000, 10000))
    a = torch.zeros((100000, 10000), device="cuda:0")
    b = torch.zeros((100000, 10000), device="cuda:1")
    c = torch.zeros((100000, 10000), device="cuda:2")
    d = torch.zeros((100000, 10000), device="cuda:3")

    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    torch.cuda.synchronize(2)
    torch.cuda.synchronize(3)

    tic = time.time()

    boradcast(x, a, b, c, d)

    torch.cuda.synchronize(0)
    torch.cuda.synchronize(1)
    torch.cuda.synchronize(2)
    torch.cuda.synchronize(3)

    toc = time.time()

    assert check_broadcast(x, a, b, c, d)
    return toc - tic


print('Running time for broadcast1:', timeit(broadcast1), '(s)')
print('Running time for broadcast2:', timeit(broadcast2), '(s)')
print('Running time for broadcast3:', timeit(broadcast3), '(s)')
