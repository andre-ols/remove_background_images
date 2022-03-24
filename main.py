import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def show(img):
    plt.imshow(img), plt.axis('off')
    plt.show()


def load_image_with_cv2(path):
    image_bgr = cv2.imread(path)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    return image_rgb


def remove_background(image):

    height, width = image.shape[:2]

    # define rectangle coordinates
    rectangle = (20, 10, width - 50, height)

    mask = np.zeros(image.shape[:2], np.uint8)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)

    # removing background in RGB
    cv2.grabCut(image, mask, rectangle, bgdModel,
                fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    mask_2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    image_no_bg = image * mask_2[:, :, np.newaxis]

    # Convert cv2 image to PIL image
    image = Image.fromarray(image_no_bg).convert("RGBA")

    # Change black background to transparent
    datas = image.getdata()

    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    image.putdata(newData)

    return image


def load_background(path, width, height):
    background = Image.open(path).convert("RGBA")
    # background should be the same size as the image
    background = background.resize((width, height))
    return background


def join_images(image_path, background_path):
    image = load_image_with_cv2(image_path)

    height, width = image.shape[:2]

    print(image.shape)

    image_no_bg = remove_background(image)

    background = load_background(
        background_path, width, height)

    # Join images
    saida = Image.alpha_composite(background, image_no_bg)

    show(saida)


if __name__ == '__main__':
    join_images('gabigol-debochando-sorrindo-flamengo.jpg',
                'Torcida-do-Flamengo-lota-Maracana-para-jogo-do-time-no-Brasileiro.jpg')
