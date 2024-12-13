"""Dataset image transformations.

Taken from the Pytorch Resnet page: https://pytorch.org/hub/pytorch_vision_resnet/
"""

from torchvision import transforms

"""
- RandomResizedCrop: focus on random aspects of the image
- RandomHorizontalFlip: double the dataset size by flipping some images horizontally
- Normalize: normalize the image to the pretrained ImageNet mean and standard deviation
"""
RESNET_TRAIN_TRANSFORM = transforms.Compose(
    [
        transforms.RandomResizedCrop(256 / 0.875),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)


"""
- Resize + CenterCrop: eliminate the outer part of the image to remove background noise
- Normalize: normalize the image to the pretrained ImageNet mean and standard deviation
"""
RESNET_TEST_TRANSFORM = transforms.Compose(
    [
        transforms.Resize(int(256 / 0.875)),
        transforms.CenterCrop(256),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ]
)
