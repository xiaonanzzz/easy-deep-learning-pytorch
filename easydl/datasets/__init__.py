from torchvision.datasets.folder import default_loader

class ImageLoader(object):
    def __init__(self, *args, image_transform=None, **kwargs):
        super(ImageLoader, self).__init__(*args, **kwargs)
        self.image_transform = image_transform

    def load_image(self, path):
        im = default_loader(path)
        if self.image_transform is not None:
            im = self.image_transform(im)
        return im

