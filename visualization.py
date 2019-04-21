from PIL import Image
import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from cnn import Net, load_dataset

CLASSES_BY_NR = ['Apple Braeburn', 'Apple Golden 1', 'Apple Golden 2', 'Apple Golden 3', 'Apple Granny Smith', 'Apple Red 1', 'Apple Red 2', 'Apple Red 3', 'Apple Red Delicious', 'Apple Red Yellow 1', 'Apple Red Yellow 2', 'Apricot', 'Avocado', 'Avocado ripe', 'Banana', 'Banana Lady Finger', 'Banana Red', 'Cactus fruit', 'Cantaloupe 1', 'Cantaloupe 2', 'Carambula', 'Cherry 1', 'Cherry 2', 'Cherry Rainier', 'Cherry Wax Black', 'Cherry Wax Red', 'Cherry Wax Yellow', 'Chestnut', 'Clementine', 'Cocos', 'Dates', 'Granadilla', 'Grape Blue', 'Grape Pink', 'Grape White', 'Grape White 2', 'Grape White 3', 'Grape White 4', 'Grapefruit Pink', 'Grapefruit White', 'Guava', 'Hazelnut', 'Huckleberry', 'Kaki', 'Kiwi', 'Kumquats', 'Lemon', 'Lemon Meyer', 'Limes', 'Lychee', 'Mandarine', 'Mango', 'Mangostan', 'Maracuja', 'Melon Piel de Sapo', 'Mulberry', 'Nectarine', 'Orange', 'Papaya', 'Passion Fruit', 'Peach', 'Peach 2', 'Peach Flat', 'Pear', 'Pear Abate', 'Pear Kaiser', 'Pear Monster', 'Pear Williams', 'Pepino', 'Physalis', 'Physalis with Husk', 'Pineapple', 'Pineapple Mini', 'Pitahaya Red', 'Plum', 'Plum 2', 'Plum 3', 'Pomegranate', 'Pomelo Sweetie', 'Quince', 'Rambutan', 'Raspberry', 'Redcurrant', 'Salak', 'Strawberry', 'Strawberry Wedge', 'Tamarillo', 'Tangelo', 'Tomato 1', 'Tomato 2', 'Tomato 3', 'Tomato 4', 'Tomato Cherry Red', 'Tomato Maroon', 'Walnut']
CLASSES_BY_NAME = {CLASSES_BY_NR[nr]: nr for nr in range(len(CLASSES_BY_NR))}

def plot_heatmap(img):
    """img is a tensor of shape (channels, height, width)"""
    nrows, ncols = 4, 4
    fig = plt.figure(figsize=(nrows, ncols))
    channels = img.shape[0]
    for plot_nr in range(min(channels, nrows*ncols) ):
        fig.add_subplot(nrows, ncols, plot_nr+1)
        #plt.imshow(img[plot_nr])
        plt.pcolor(img[plot_nr], cmap=plt.cm.Reds)
    plt.show()

def example_heatmap():
    x = np.random.randint(0, 256, (100, 100))
    plt.pcolor(x, cmap=plt.cm.Reds)
    plt.show()

# def occlusion_heatmap(model, image_path, target_cls, device):
#     if target_cls not in CLASSES_BY_NAME:
#         raise Exception("Invalid target_cls name")
#     target_nr = CLASSES_BY_NAME[target_cls]
#
#     target = torch.tensor([target_nr], dtype=torch.int64)
#     data = torch.tensor(np.array(Image.open(image_path), dtype=np.float32)).permute(2,0,1).unsqueeze(0)
#     #import  ipdb; ipdb.set_trace()
#     print(data.shape)
#     with torch.no_grad():
#         data, target = data.to(device), target.to(device)
#         output = model(data)
#         #import ipdb; ipdb.set_trace()
#         pred = output.argmax(dim=1, keepdim=True).item()
#         print("Predicted class: ", CLASSES_BY_NR[pred])

def plot_image(img, channel_first=True):
    """Plots image"""
    img = np.array(img)
    assert len(img.shape) == 3
    if channel_first:
        Image.fromarray(np.moveaxis(255*img, 0,-1).astype(np.uint8), mode='RGB').show()
        return
    else:
        Image.fromarray((255*img).astype(np.uint8), mode='RGB').show()

def occlude_img(img, height, width, occ_kernel, value=None):
    """
    It is assumed img is in the format C,H,W
    Returns a new occluded image around the position x,y with a given kernel
    """
    im_chanel, im_height, im_width = img.shape
    top = max(height - occ_kernel//2, 0)
    left = max(width - occ_kernel//2, 0)
    bottom = min(height + occ_kernel//2, im_height-1)
    right = min(width + occ_kernel//2, im_width-1)
    occluded = img.clone()
    occluded[:, top:bottom, left:right] = value
    return occluded


def heatmap_example(model, loader, device):
    data, target = next(iter(loader))

    # data, target = data.to(device), target.to(device)
    # output = model(data)
    # pred = output.argmax(dim=1, keepdim=True).item()
    # print("Ground truth: ", CLASSES_BY_NR[target.item()])
    # print("Predicted class: ", CLASSES_BY_NR[pred])

    occluded = occlude_img(data[0], 20,40, 10, 0)
    #plot_image(data[0])
    #plot_image(occluded)
    import ipdb; ipdb.set_trace()
    im_chanel, im_height, im_width = data[0].shape
    losses = np.zeros((im_height, im_width))

    #with torch.no_grad():
    for pos_height in range(im_height):
        for pos_width in range(im_width):

            occluded_img = occlude_img(data[0], pos_height, pos_width, 100, 1)
            output = model(occluded_img.unsqueeze(0))
            loss = F.cross_entropy(output, target)
            loss.backward()
            #import ipdb; ipdb.set_trace()
            losses[pos_height, pos_width] = loss


    # normalize and plot heatmap
    xmax, xmin = losses.max(), losses.min()
    losses = (losses - xmin) / (xmax - xmin)
    plt.pcolor(losses, cmap=plt.cm.Reds)
    plot_image(data[0])
    plt.show()
    #import ipdb; ipdb.set_trace()

def plot_occlusion_heatmap(image, target_cls):
    """
    For each position in the input image consider a square of size 10
    centered inthat pixel (you can chage the size if you want).
    Fill that square with pixels withconstant value
    and compute the loss function for that image.
    Create a heatmap out of thosevalue
    """
    im_chanel, im_height, im_width = image.shape
    losses = np.zeros((im_height, im_width))
    target = torch.tensor([CLASSES_BY_NAME[target_cls]], dtype=torch.int64)
    for pos_height in range(im_height):
        for pos_width in range(im_width):
            occluded_img = occlude_img(image, pos_height, pos_width, occ_kernel=10, value=1)
            output = model(occluded_img.unsqueeze(0))
            loss = F.cross_entropy(output, target)
            loss.backward()
            losses[pos_height, pos_width] = loss
    plt.pcolor(losses, cmap=plt.cm.Reds)
    plt.show()

def plot_pixelwise_gradients(model, image, target_cls):
    """
    Create heatmaps from pixelwise gradients of the loss function.
    """
    output = model(image.unsqueeze(0))
    target = torch.tensor([CLASSES_BY_NAME[target_cls]], dtype=torch.int64)
    loss = F.cross_entropy(output, target)
    loss.backward()
    for layer_grad in model.grads:
        plot_heatmap(layer_grad[0])


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Visualization tool for CNN Pytorch model')
    parser.add_argument('--use_cuda', action='store_true', default=False,
                        help='enables CUDA training')
    parser.add_argument('--image', help='path to the image', required=True)
    parser.add_argument('--image_cls', help='cllass of the image', required=True)
    parser.add_argument('--load_weights', help='path to saved model\'s file', required=True)
    parser.add_argument('--dataset', help='Path to dataset from which to load the images for visualizatiom')
    args = parser.parse_args()

    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    model = Net().to(device)
    model.load_state_dict(torch.load(args.load_weights))
    model.eval()

    image = np.array(Image.open(args.image)).astype(np.float32) / 255
    image_tensor = torch.tensor(image).permute(2, 0, 1)

    plot_pixelwise_gradients(model, image_tensor, args.image_cls)
    plot_occlusion_heatmap(image_tensor, args.image_cls)