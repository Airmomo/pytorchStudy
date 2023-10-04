

def print_named_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Parameter name: {name}")
            print(f"Parameter shape: {param.shape}")
            print(f"Parameter values: {param.data}")
