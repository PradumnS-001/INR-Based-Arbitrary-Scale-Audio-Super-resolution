import os

def count_files(
    path:str, 
    ender:str|tuple[str]='', 
    starter:str|tuple[str]='', 
    container:str='')->int:
    
    """
    Recursively counts the number of files in a folder
    """
    
    path = rf'{path}'
    count = 0
    with os.scandir(path=path) as entries:
        
        for entry in entries:
            
            if entry.is_file():
                
                count += 1 if (entry.name.startswith(starter) and entry.name.endswith(ender) and container in entry.name) else 0
                
            elif entry.is_dir():
                
                count += count_files(
                    path= os.path.join(path, entry.name),
                    starter=starter,
                    ender=ender,
                    container=container)
                
            else: count += 0
            
    return count

def get_leaf_files(
    path:str, 
    ender:str|tuple[str]='', 
    starter:str|tuple[str]='', 
    container:str='')->list[str]:
    
    """
    Returns all the leaf-files in a folder
    """
    
    path = rf'{path}'
    files = []
    with os.scandir(path=path) as entries:
        
        for entry in entries:
            
            if entry.is_file():
                
                files += [os.path.join(path, entry.name)] if (entry.name.startswith(starter) and entry.name.endswith(ender) and container in entry.name) else []
                
            elif entry.is_dir():
                
                files += get_leaf_files(
                    path= os.path.join(path, entry.name),
                    starter=starter,
                    ender=ender,
                    container=container)
                
            else: files += []
            
    return files

if __name__ == "__main__":
    pass