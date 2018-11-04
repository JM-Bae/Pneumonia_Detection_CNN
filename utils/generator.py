#Data Generator
class generator(keras.utils.Sequence):

    def __init__(self, folder, filenames, pneumonia_locations=None, batch_size=32,
                 image_size=256, shuffle=True, augment=False, predict=False):
        self.folder = folder
        self.filenames = filenames
        self.pneumonia_locations = pneumonia_locations
        self.batch_size = batch_size
        self.image_size = image_size
        self.shuffle = shuffle
        self.augment = augment
        self.predict = predict
        self.on_epoch_end()

    def __load__(self, filename):
        # load dicom images into np.array
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array

        # create empty mask
        msk = np.zeros(img.shape)
        filename = filename.split('.')[0]

        # if img contains pneumonia
        if filename in self.pneumonia_locations:
            for location in self.pneumonia_locations[filename]:
                x,y,w,h = location
                msk[y:y+h, x:x+w] = 1

        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        msk = resize(msk, (self.image_size, self.image_size), mode='reflect') > 0.5

        if self.augment and random.random() >0.5:
            img = np.fliplr(img)
            msk = np.fliplr(msk)

        # trailing channel dimension?
        img = np.expand_dims(img, -1)
        msk = np.expand_dims(msk, -1)
        return img, msk

    def __loadpredict__(self, filename):
        img = pydicom.dcmread(os.path.join(self.folder, filename)).pixel_array
        img = resize(img, (self.image_size, self.image_size), mode='reflect')
        img = np.expand_dims(img, -1)
        return img

    def __getitem__(self, index):
        filenames = self.filenames[index*self.batch_size:(index+1)*self.batch_size]

        if self.predict:
            # load files
            imgs = [self.__loadpredict__(filename) for filename in filenames]
            # create np batch
            imgs = np.array(imgs)
            return imgs, filenames

        # train mode: return img & masks
        else:
            # load files
            items = [self.__load__(filename) for filename in filenames]
            # unzip imgs and msks
            imgs, msks = zip(*items)
            # create np batch
            imgs = np.array(imgs)
            msks = np.array(msks)
            return imgs, msks

    def on_epoch_end(self):
        if self.shuffle:
            random.shuffle(self.filenames)

    def __len__(self):
        if self.predict:
            # return everything
            return int(np.ceil(len(self.filenames)/self.batch_size))
        else:
            # return full batches only
            return int(len(self.filenames)/self.batch_size)            