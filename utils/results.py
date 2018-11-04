def plot_results(history):
    plt.figure(figsize=(12,4))
    plt.subplot(131)
    plt.plot(history.epoch, history.history["loss"], label="Train loss")
    plt.plot(history.epoch, history.history["val_loss"], label="Valid loss")
    plt.legend()

    plt.subplot(132)
    plt.plot(history.epoch, history.history["acc"], label="Train accuracy")
    plt.plot(history.epoch, history.history["val_acc"], label="Valid accuracy")
    plt.legend()

    plt.subplot(133)
    plt.plot(history.epoch, history.history["mean_iou"], label="Train iou")
    plt.plot(history.epoch, history.history["val_mean_iou"], label="Valid iou")
    plt.legend()
    plt.show()


def plot_valid_prediction(valid_gen, model):
   
    for imgs, msks in valid_gen:
    # predict batch of images
    preds = model.predict(imgs)
    # create figure
    f, axarr = plt.subplots(4, 8, figsize=(20,15))
    axarr = axarr.ravel()
    axidx = 0
    # loop through batch
    for img, msk, pred in zip(imgs, msks, preds):
        # plot image
        axarr[axidx].imshow(img[:, :, 0])
        # threshold true mask
        comp = msk[:, :, 0] > 0.5
        # apply connected components
        comp = measure.label(comp)
        # apply bounding boxes
        predictionString = ''
        for region in measure.regionprops(comp):
            # retrieve x, y, height and width
            y, x, y2, x2 = region.bbox
            height = y2 - y
            width = x2 - x
            axarr[axidx].add_patch(patches.Rectangle((x,y),width,height,linewidth=2,edgecolor='b',facecolor='none'))
        # threshold predicted mask
        comp = pred[:, :, 0] > 0.5
        # apply connected components
        comp = measure.label(comp)
        # apply bounding boxes
        predictionString = ''
        for region in measure.regionprops(comp):
            # retrieve x, y, height and width
            y, x, y2, x2 = region.bbox
            height = y2 - y
            width = x2 - x
            axarr[axidx].add_patch(patches.Rectangle((x,y),width,height,linewidth=2,edgecolor='r',facecolor='none'))
        axidx += 1
    plt.show()
    # only plot one batch
    break