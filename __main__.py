import skimage
from skimage.io import imread, imsave, imshow
from skimage.color import rgba2rgb
from skimage.util import img_as_ubyte
from skimage import segmentation
from skimage.transform import resize
from inpainter import Inpainter
# from inpainter_Origion import Inpainter_Origion
import cv2

def main():
    # args = parse_args()

    original_image = imread("/home/kow/CutOutWiz/Projects/pythonProject/exemplar_inpainting/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/resources/beautiful-caucasian-woman-with-makeup.png")
    original_mask = imread('/home/kow/CutOutWiz/Projects/pythonProject/exemplar_inpainting/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/resources/marked_mask_before_old.png', as_gray=True)
    # original_image = imread("/home/kow/CutOutWiz/Projects/pythonProject/exemplar_inpainting/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/resources/beauty-portrait-young-brunettete-woman.png")
    # original_mask = imread('/home/kow/CutOutWiz/Projects/pythonProject/exemplar_inpainting/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/resources/marked_mask_before.png', as_gray=True)
    print(" image type:", original_image.dtype)
    # original_image = imread("/home/kow/CutOutWiz/Projects/pythonProject/exemplar_inpainting/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/resources/image8.jpg")
    # original_mask = imread('/home/kow/CutOutWiz/Projects/pythonProject/exemplar_inpainting/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/resources/mask8.jpg', as_gray=True)

    if original_image is not None and original_mask is not None:
        # print('got inputs', original_image.shape)
        h, w, ch = original_image.shape
        print(original_image.shape, original_mask.shape)
        if ch == 4:
            original_image = rgba2rgb(original_image)
            original_image = img_as_ubyte(original_image)

            # print(original_image.shape, "rgb")
        th, tw = 400, 800
        image = original_image
        mask = original_mask
        print(" image type again:", original_image.dtype)
        # image = resize(image, (th, tw), order=0, preserve_range=True, anti_aliasing=False).astype('uint8')
        # mask = resize(mask, (th, tw), order=0, preserve_range=True, anti_aliasing=False).astype('uint8')
        # image = cv2.resize(image, (th, tw))
        # mask = cv2.resize(mask, (th, tw))
        # imshow(image)
        # skimage.io.show()

        output_image = Inpainter(
            image,
            mask,
            patch_size=5,
            plot_progress=True
        ).inpaint()
        result = resize(output_image, (th, tw), order=0, preserve_range=True, anti_aliasing=False).astype('uint8')
        # cv2.imwrite('test.png', output_image)
        # imsave('test.jpg', output_image)
        imsave('test.jpg', result)
    else:
        print('error in loading image....')


if __name__ == '__main__':
    main()
