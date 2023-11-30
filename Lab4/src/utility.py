import torch


def setup_device(model, use_cuda):
    if torch.cuda.is_available() and use_cuda.lower() == 'y':
        device = torch.device('cuda')
        model.cuda()
        print('Using CUDA-capable GPU...')
    else:
        device = torch.device('cpu')
        model.cpu()
        print('Using CPU...')
    return device
