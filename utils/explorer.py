# Data Exploration
def explore_data():

    print('Total train images:',len(filenames))
    print('Images with pneumonia:', len(pneumonia_locations))

    ns = [len(value) for value in pneumonia_locations.values()]
    plt.figure()
    plt.hist(ns)
    plt.xlabel('Pneumonia per image')
    plt.xticks(range(1, np.max(ns)+1))
    plt.show()

    heatmap = np.zeros((1024, 1024))
    ws = []
    hs = []
    for values in pneumonia_locations.values():
        for value in values:
            x,y,w,h = value
            heatmap[y:y+h, x:x+w] +=1
            ws.append(w)
            hs.append(h)

    plt.figure()
    plt.title('Pneumonia location heatmap')
    plt.imshow(heatmap)
    plt.figure()
    plt.title('Pneumonia height lengths')
    plt.hist(hs, bins=np.linspace(0,1000,50))
    plt.show()
    plt.figure()
    plt.title('Pneumonia width lengths')
    plt.hist(ws, bins=np.linspace(0,1000,50))
    plt.show()

    print('Minimum pneumonia height:', np.min(hs))
    print('Minimum pneumonia width:', np.min(ws))
    