import os
from torch import nn

def get_leaf_files(
    path:str, 
    ender:str|tuple[str]='', 
    starter:str|tuple[str]='', 
    container:str='')->tuple[int,list[str]]:
    
    """
    Returns all the leaf-files in a folder
    """
    
    path = rf'{path}'
    files = []
    with os.scandir(path=path) as entries:
        
        for entry in entries:
            
            name = os.path.join(path, entry.name)
            if entry.is_file():
                
                files += [name] if (name.startswith(starter) and name.endswith(ender) and container in name) else []
                
            elif entry.is_dir():
                
                files += get_leaf_files(
                    path= name,
                    starter=starter,
                    ender=ender,
                    container=container)[1]
                
            else: files += []
            
    return len(files), files

def count_params(model:nn.Module):
    """Prints a brief summary of the trainable parameter count per module."""
    print("\n" + "="*45)
    print("Model Parameter Summary")
    print("="*45)
    
    total_params = 0
    module_counts = {}
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Group by the top-level module name (e.g., 'macro_encoder', 'alpha_branch')
            module_name = name.split('.')[0]
            module_counts[module_name] = module_counts.get(module_name, 0) + param.numel()
            total_params += param.numel()
            
    for mod, count in module_counts.items():
        print(f"{mod:<25}: {count:,}")
        
    print("-" * 45)
    print(f"{'Total Trainable Params':<25}: {total_params:,}")
    print("="*45 + "\n")
        
    return total_params

if __name__ == "__main__":
    leg, fil = get_leaf_files('Data/VCTKCorpus', ender=('.mp4', '.wav'))
    print(leg == len(fil), leg)