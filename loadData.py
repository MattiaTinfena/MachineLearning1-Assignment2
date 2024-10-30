def loadData(dataset):

    col =[]
    coord = []
    coord = dataset.columns
    for i in range(len(coord)):
        col.append(dataset[coord[i]].tolist())
    
    return col