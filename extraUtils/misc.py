import os

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

if __name__ == "__main__":
    leg, fil = get_leaf_files('Data/VCTKCorpus', ender=('.mp4', '.wav'))
    print(leg == len(fil), leg)