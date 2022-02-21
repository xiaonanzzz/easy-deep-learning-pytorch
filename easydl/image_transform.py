from torchvision import transforms

resnet_mean = [0.485, 0.456, 0.406]
resnet_std = [0.229, 0.224, 0.225]


def make_transform_train_v1(image_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=resnet_mean, std=resnet_std)
    ])

def make_transform_test_v1(image_size):
    return transforms.Compose([
        transforms.Resize(int(image_size / 0.875)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=resnet_mean, std=resnet_std)
    ])


resnet_transform_train = make_transform_train_v1(image_size=224)
resnet_transform_test = make_transform_test_v1(image_size=224)