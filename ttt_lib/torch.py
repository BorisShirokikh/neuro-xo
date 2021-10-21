from dpipe.torch import to_device


def optimizer_step(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


def masked_mse_loss(x, y, mask):
    return (x.masked_select(to_device(mask.type(dtype=torch.BoolTensor), device=x)) - y) ** 2
