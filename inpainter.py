import numpy as np
import matplotlib.pyplot as plt
import time
import os

import skimage
from skimage.io import imread, imsave, imshow
from skimage.color import rgb2gray, rgb2lab
from skimage.filters import laplace
from scipy.ndimage.filters import convolve
from math import pi
import cv2
from PIL import Image


class Inpainter():

    def __init__(self, image, mask, patch_size, plot_progress=False):
        self.image = image.astype('uint8')
        self.mask = mask.astype('uint8')
        self.patch_size = patch_size
        self.plot_progress = plot_progress

        # Non initialized attributes
        self.plot_image_path = '/home/kow/CutOutWiz/Projects/pythonProject/exemplar_inpainting/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/Improved-inpaint-object-remover-c9aaf60350fdce0abc4508c4bf26f30ef7fc33bb/resources/output/'
        self.working_image = None
        self.working_mask = None
        self.front = None
        self.confidence = None
        self.data = None
        self.H_p = None
        self.priority = None
        self.new_source_patch = None
        self.new_mask_patch = None
        self.new_source_patch_copy = None
        self.x = None
        self.y = None
        self.h = None
        self.w = None
        self.pixel_extension = 5        # imshow(self.image)
        # # imshow(self.new_mask_patch)
        # skimage.io.show()

    def inpaint(self):
        """ Compute the new image and return it """

        self._validate_inputs()
        self._initialize_attributes()

        start_time = time.time()
        keep_going = True
        while keep_going:
            self._find_front()
            if self.plot_progress:
                self._plot_image()

            self._update_priority()
            target_pixel = self._find_highest_priority_pixel()
            self.update_patchsize(target_pixel)
            find_start_time = time.time()
            # self.new_find_source_patch()
            source_patch = self._find_source_patch(target_pixel)
            print('Time to find best: %f seconds'
                  % (time.time() - find_start_time))

            self._update_image(target_pixel, source_patch)

            keep_going = not self._finished()
            if not keep_going:
                if self.plot_progress:
                    self._plot_image()

        # print("now generate result gif")
        # os.system('convert -delay 30 -loop 0 ' + self.plot_image_path + '*.jpg ' + self.plot_image_path + 'result.gif')
        print('Took %f seconds to complete' % (time.time() - start_time))
        self.working_image[self.y - self.pixel_extension: self.y + self.pixel_extension + self.h, self.x - self.pixel_extension: self.x + self.w + self.pixel_extension] = self.new_source_patch_copy
        # bg = np.copy(self.working_image)
        # bg = Image.fromarray(bg.astype('uint8'), 'RGB')
        # #y - 7: y + h + 7, x - 7: x + w + 7
        # ROI = self.working_image[self.y - 7: self.y + self.h + 7, self.x - 7: self.x + self.w + 7]
        # bg.paste(self.new_source_patch_copy, ROI[0][0], ROI[0][1])
        # bg.save('final.jpg')
        return self.working_image

    def _validate_inputs(self):
        if self.image.shape[:2] != self.mask.shape:
            raise AttributeError('mask and image must be of the same size')

    def _plot_image(self):
        height, width = self.new_mask_patch.shape  # 480*360
        # print('plot image:', height, width)
        # Remove the target region from the image
        inverse_mask = 1 - self.new_mask_patch
        rgb_inverse_mask = self._to_rgb(inverse_mask)
        image = self.new_source_patch_copy * rgb_inverse_mask

        # Fill the target borders with red
        image[:, :, 0] += self.front * 255  # Red
        # imshow(image)
        # skimage.io.show()

        # Fill the inside of the target region with white
        white_region = (self.new_mask_patch) * (255)
        rgb_white_region = self._to_rgb(white_region)

        # imshow(rgb_white_region)
        # skimage.io.show()

        image += rgb_white_region
        # imshow(image)
        # skimage.io.show()

        plot_image_path = self.plot_image_path
        remaining = self.new_mask_patch.sum()
        plot_image_path = plot_image_path + str(int(height * width - remaining)) + '.jpg'

        image = image.astype('uint8')
        imsave(plot_image_path, image)
        # cv2.imwrite(plot_image_path, image)

    def _initialize_attributes(self):
        """ Initialize the non initialized attributes

        The confidence is initially the inverse of the mask, that is, the
        target region is 0 and source region is 1.

        The data starts with zero for all pixels.

        The working image and working mask start as copies of the original
        image and mask.
        """
        height, width = self.image.shape[:2]  # image.shape[:2] = 【480, 360】
        # print(height, width, "image shape inpainter")

        self.working_image = np.copy(self.image)
        self.working_mask = np.copy(self.mask)

        self.new_source_patch, self.new_mask_patch, self.x, self.y, self.w, self.h = self.new_find_source_patch()

        height_new, width_new = self.new_source_patch_copy.shape[:2]
        print("new mask size:", self.new_mask_patch.shape)
        self.new_mask_patch = rgb2gray(self.new_mask_patch)
        self.confidence = (1 - self.new_mask_patch).astype('uint8') #changed

        # imshow(self.image)
        # # imshow(self.new_mask_patch)
        # skimage.io.show()

        self.data = np.zeros([height_new, width_new])
        self.H_p = np.zeros([height_new, width_new])
        self.lamda = np.zeros([height_new, width_new, 2])

        print(self.working_image.shape, "in plot progress")
        # self.new_find_source_patch()

    def _find_front(self):
        """ Find the front using laplacian on the mask

        The laplacian will give us the edges of the mask, it will be positive
        at the higher region (white) and negative at the lower region (black).
        We only want the the white region, which is inside the mask, so we
        filter the negative values.

        """

        # self.front = (laplace(self.working_mask) > 0).astype('uint8')
        self.front = (laplace(self.new_mask_patch) > 0).astype('uint8')
        # imshow(self.front)
        # skimage.io.show()
        # cv2.imshow('fill_2', self.working_mask)
        # cv2.waitKey(0)
        # TODO: check if scipy's laplace filter is faster than scikit's

    def update_patchsize(self,target_pixel):
        target_patch = self._get_patch(target_pixel)
        mask = 1 - self._patch_data(self.new_mask_patch, target_patch)
        target_lamda = self._patch_data(self.lamda, target_patch)
        lamda1_avg = sum(sum(target_lamda[:, :, 0] * mask)) / sum(sum((mask == 1)))
        lamda2_avg = sum(sum(target_lamda[:, :, 1] * mask)) / sum(sum((mask == 1)))
        ave = ((lamda1_avg - lamda2_avg)/(lamda1_avg + lamda2_avg))**2
        print('average,', ave)
        if ave >= 0.9:
            self.patch_size = 5
        elif ave >= 0.8:
            self.patch_size = 7
        elif ave >= 0.7:
            self.patch_size = 9
        else:
            self.patch_size = self.patch_size #11
        print('updated patch size:', self.patch_size)

    def _update_priority(self): # priority function
        self._update_confidence()
        gradient = self._update_data()
        self._structure_tensor(gradient)
        # self.priority = self.confidence * self.data * self.front
        self.priority = (3/5 * self.confidence + 1/5 * self.data + 1/5 * self.H_p) * self.front
        # self.priority = (2 / 5 * self.confidence + 3 / 5 * self.H_p) * self.front

    def _update_confidence(self):

        new_confidence = np.copy(self.confidence)
        front_positions = np.argwhere(self.front == 1)
        # print("front position :",len(front_positions), front_positions)
        for point in front_positions:
            patch = self._get_patch(point)
            # print('patch:', patch)
            # k = self._patch_data(self.confidence, patch)
            # kk = sum(self._patch_data(self.confidence, patch))
            # kkk = sum(sum(self._patch_data(self.confidence, patch)))
            new_confidence[point[0], point[1]] = sum(sum(self._patch_data(self.confidence, patch))) / self._patch_area(patch)

        self.confidence = new_confidence

    def _update_data(self):

        normal = self._calc_normal_matrix()
        gradient, max_gradient = self._calc_gradient_matrix()
        normal_gradient = normal * max_gradient

        # self.data = np.sqrt(normal_gradient[:, :, 0] ** 2 + normal_gradient[:, :, 1] ** 2) + 0.0001
        # To be sure to have a greater than 0 data

        self.data = np.abs(normal_gradient[:, :, 0] + normal_gradient[:, :, 1]) + 0.00001
        # To be sure to have a greater than 0 data
        return gradient

    def _calc_normal_matrix(self):
        # x_kernel = np.array([[.25, 0, -.25], [.5, 0, -.5], [.25, 0, -.25]])
        # y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        x_kernel = np.array([[.25, 0, -.25], [.2, 0, -.5], [.25, 0, -.25]])
        y_kernel = np.array([[-.25, -.5, -.25], [0, 0, 0], [.25, .5, .25]])

        x_normal = convolve(self.new_mask_patch.astype(float), x_kernel)
        y_normal = convolve(self.new_mask_patch.astype(float), y_kernel)
        normal = np.dstack((x_normal, y_normal))

        height, width = normal.shape[:2]
        norm = np.sqrt(y_normal ** 2 + x_normal ** 2).reshape(height, width, 1).repeat(2, axis=2)
        norm[norm == 0] = 1

        unit_normal = normal / norm
        return unit_normal

    def _calc_gradient_matrix(self):
        # TODO: find a better method to calc the gradient
        #### specify the region to calculate gradient --TRY 1
        height, width = self.new_source_patch_copy.shape[:2]  # 480*360 # changed

        # image = np.copy(self.working_image)
        # gradient_r = np.nan_to_num(np.array(np.gradient(image[:, :, 0])))
        # gradient_g = np.nan_to_num(np.array(np.gradient(image[:, :, 1])))
        # gradient_b = np.nan_to_num(np.array(np.gradient(image[:, :, 2])))
        # gradient = np.zeros([2, height, width])
        # gradient[0] = (gradient_r[0] + gradient_g[0] + gradient_b[0])/3
        # gradient[1] = (gradient_r[1] + gradient_g[1] + gradient_b[1])/3

        grey_image = rgb2gray(self.new_source_patch_copy)
        grey_image[self.new_mask_patch == 1] = None

        # gradient for entire new working image
        gradient = np.nan_to_num(np.array(np.gradient(grey_image)))
        gradient_val = np.sqrt(gradient[0] ** 2 + gradient[1] ** 2)
        max_gradient = np.zeros([height, width, 2])

        front_positions = np.argwhere(self.front == 1)
        for point in front_positions:
            patch = self._get_patch(point)
            patch_y_gradient = self._patch_data(gradient[0], patch)
            patch_x_gradient = self._patch_data(gradient[1], patch)
            patch_gradient_val = self._patch_data(gradient_val, patch)
            patch_max_pos = np.unravel_index(patch_gradient_val.argmax(), patch_gradient_val.shape)

            # max_gradient[point[0], point[1], 0] = patch_y_gradient[patch_max_pos]  #
            # max_gradient[point[0], point[1], 1] = patch_x_gradient[patch_max_pos]  #

            max_gradient[point[0], point[1], 0] = patch_x_gradient[patch_max_pos]  #
            # max_gradient[point[0], point[1], 1] = patch_y_gradient[patch_max_pos] * -1  #
            max_gradient[point[0], point[1], 1] = patch_y_gradient[patch_max_pos]

        return gradient, max_gradient

    def _structure_tensor(self, gradient):
        height, width = gradient.shape[1:]
        h = np.zeros([height, width])
        # ro = 0.2
        ro = 0.2
        # structure tensor of complete gradient........(try2.............)
        for i in range(height):
            for j in range(width):
                str_tensor = np.array([[gradient[0, i, j]*gradient[0, i, j], gradient[0, i, j]*gradient[1, i, j]],
                                       [gradient[0, i, j]*gradient[1, i, j], gradient[1, i, j]*gradient[1, i, j]]])\
                                       / (2*pi*ro**2)*np.exp(-(i**2+j**2)/(2*ro**2))

                eigenvalue, featurevector = np.linalg.eig(str_tensor)
                h[i, j] = (eigenvalue[0] - eigenvalue[1])**2
                self.lamda[i, j, :] = eigenvalue
        self.H_p = 0.8 * h + np.exp(-h)

# structure tensor of fill front only........................................
        # front_positions = np.argwhere(self.front == 1)
        # for point in front_positions:
        #     i = point[0]
        #     j = point[1]
        #     str_tensor = np.array([[gradient[0, i, j] * gradient[0, i, j], gradient[0, i, j] * gradient[1, i, j]],
        #                           [gradient[0, i, j]*gradient[1, i, j], gradient[1, i, j]*gradient[1, i, j]]])\
        #                           / (2 * pi * ro ** 2) * np.exp(-(i ** 2 + j ** 2) / (2 * ro ** 2))
        #     eigenvalue, featurevector = np.linalg.eig(str_tensor)
        #     h[i, j] = (eigenvalue[0] - eigenvalue[1])**2
        #     self.H_p = 0.8 * h + np.exp(-h)

    def _find_highest_priority_pixel(self):
        point = np.unravel_index(self.priority.argmax(), self.priority.shape)

        return point

    def new_find_source_patch(self):
        # TODO: change the contour finding global in future
        cnts, _ = cv2.findContours(self.mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        image_copy = np.copy(self.working_image)
        print(image_copy.shape, 'working.....')
        mask_copy = np.copy(self.working_mask)
        # imshow(self.working_image)
        # # imshow(mask_copy)
        # skimage.io.show()
        for c in cnts:
            (self.x, self.y, self.w, self.h) = cv2.boundingRect(c)
            print(self.x, self.y, self.w, self.h)
            bbox = self.x, self.y, self.x + self.w, self.y + self.h
            col_min, col_max = bbox[3] + 3, bbox[1] - 3
            row_min, row_max = bbox[0] - 3, bbox[2] + 3
            # rect = cv2.rectangle(self.working_image, (row_min, col_min), (row_max, col_max), (255, 255, 255), 1)
            new_source_patch = image_copy[self.y-self.pixel_extension:self.y+self.h+self.pixel_extension, self.x-self.pixel_extension:self.x+self.w+self.pixel_extension]
            self.new_source_patch_copy = new_source_patch

            new_mask_patch = mask_copy[self.y-self.pixel_extension:self.y+self.h+self.pixel_extension, self.x-self.pixel_extension:self.x+self.w+self.pixel_extension]
            new_mask_patch = cv2.merge([new_mask_patch, new_mask_patch, new_mask_patch]) # 3 mode generate of mask

            # on basis of mask, generate original patch from bitwise_OR
            masked_and = cv2.bitwise_or(new_source_patch, new_mask_patch)
            # masked_and = cv2.bitwise_(masked_and)
            new_source_patch = masked_and.copy()
            # cv2.imshow('xx', new_source_patch)
            # cv2.waitKey(0)
            # cv2.imwrite('rect.png', rect)
            print(new_source_patch.shape,'sp....')
            # cv2.imwrite('new.png', new_source_patch)
            # cv2.imwrite('new_mask.png', new_mask_patch)
            # imsave('new.png', self.new_source_patch_copy)
            # imsave('new_mask.png', new_mask_patch)
            return new_source_patch, new_mask_patch, self.x, self.y, self.w, self.h
            # cv2.imshow('xx', new_source_patch)
            # cv2.waitKey(0)

    def _find_source_patch(self, target_pixel): # modified the source patch
        target_patch = self._get_patch(target_pixel)
        # height, width = self.working_image.shape[:2] # old source region
        # self.new_source_patch, self.new_mask_patch = self.new_find_source_patch() # new source region
        height, width = self.new_source_patch_copy.shape[:2]
        # source patch is working image entire (height, width) - target_patch (patch_height, patch_width))
        patch_height, patch_width = self._patch_shape(target_patch)

        best_match = None
        best_match_difference = 0

        # lab_image = rgb2lab(self.working_image)
        # lab_image = self.new_source_patch_copy
        lab_image = rgb2lab(self.new_source_patch_copy)

        print("source path dimension:", height, width, patch_height, patch_width)

        for y in range(height - patch_height+1):
            for x in range(width - patch_width+1):
                source_patch = [
                    [y, y + patch_height-1],
                    [x, x + patch_width-1]
                ]
                # if self._patch_data(self.working_mask, source_patch).sum() != 0:
                if self._patch_data(self.new_mask_patch, source_patch).sum() != 0:
                   continue
                # if self.new_mask_patch[source_patch[0][0]:source_patch[0][1] + 1,
                #            source_patch[1][0]:source_patch[1][1] + 1].sum() !=0:
                #     continue

                difference = self._calc_patch_difference(
                    lab_image,
                    target_patch,
                    source_patch
                )

                if best_match is None or difference < best_match_difference:
                    best_match = source_patch
                    best_match_difference = difference
                # print("best match....", best_match)
        return best_match

    def _update_image(self, target_pixel, source_patch):
        target_patch = self._get_patch(target_pixel)
        target_patch_2 = self._get_patch_2(target_pixel)

        pixels_positions = np.argwhere(self._patch_data(self.new_mask_patch, target_patch) == 1) \
                           + [target_patch[0][0], target_patch[1][0]]

        patch_confidence = self.confidence[target_pixel[0], target_pixel[1]]

        for point in pixels_positions:
            self.confidence[point[0], point[1]] = patch_confidence

        # mask = self._patch_data(self.working_mask, target_patch)
        mask = self._patch_data(self.new_mask_patch, target_patch)
        rgb_mask = self._to_rgb(mask)

        source_data = self._patch_data(self.new_source_patch_copy, source_patch)
        target_data = self._patch_data(self.new_source_patch_copy, target_patch) # working image would be target data

        new_data = source_data * rgb_mask + target_data * (1 - rgb_mask)

        # inpaint with new data..........
        self._copy_to_patch(self.new_source_patch_copy, target_patch, new_data)
        self._copy_to_patch(self.new_mask_patch, target_patch, 0)

        # self._copy_to_patch(self.working_image, target_patch_2, new_data)

    def _get_patch(self, point):
        half_patch_size = (self.patch_size - 1) // 2
        height, width = self.new_source_patch_copy.shape[:2]
        patch = [
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height - 1)
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width - 1)
            ]
        ]
        return patch

    def _get_patch_2(self, point):
        half_patch_size = (self.patch_size - 1) // 2
        height, width = self.working_image.shape[:2]
        patch = [
            [
                max(0, point[0] - half_patch_size),
                min(point[0] + half_patch_size, height - 1)
            ],
            [
                max(0, point[1] - half_patch_size),
                min(point[1] + half_patch_size, width - 1)
            ]
        ]
        return patch

    def _calc_patch_difference(self, image, target_patch, source_patch):
        mask = 1 - self._patch_data(self.new_mask_patch, target_patch)
        rgb_mask = self._to_rgb(mask)
        target_data = self._patch_data(
            image,
            target_patch
        ) * rgb_mask
        source_data = self._patch_data(
            image,
            source_patch
        ) * rgb_mask
        squared_distance = ((target_data - source_data) ** 2).sum()
        euclidean_distance = np.sqrt(
            (target_patch[0][0] - source_patch[0][0]) ** 2 +
            (target_patch[1][0] - source_patch[1][0]) ** 2
        )  # tie-breaker factor
        height, width = mask.shape
        two_dim_mask = mask.reshape(height, width, 1).repeat(2, axis=2)
        target_lamda = self._patch_data(self.lamda, target_patch) * two_dim_mask
        source_lamda = self._patch_data(self.lamda, source_patch) * two_dim_mask
        # lamda_distance = sum(sum(10*(self.moa(target_lamda) - self.moa(source_lamda) ** 2)))
        lamda_distance = sum(sum((target_lamda[:, :, 0]-source_lamda[:, :, 0])**2 +(target_lamda[:, :, 1]-source_lamda[:, :, 1])**2))
        return squared_distance + euclidean_distance + lamda_distance
        # return squared_distance + euclidean_distance
        # return squared_distance

    def moa(self, lamda):
        moa = (np.nan_to_num((lamda[:, :, 0] - lamda[:, :, 1])/(lamda[:, :, 0] + lamda[:, :, 1])))**2
        return moa

    def _finished(self):
        height, width = self.new_source_patch_copy.shape[:2]  # 480*360
        remaining = self.new_mask_patch.sum()
        total = height * width
        print('%d of %d completed' % (total - remaining, total))
        # return remaining == 0
        return remaining <= 0

    @staticmethod
    def _patch_area(patch):
        return (1 + patch[0][1] - patch[0][0]) * (1 + patch[1][1] - patch[1][0])

    @staticmethod
    def _patch_shape(patch):
        return (1 + patch[0][1] - patch[0][0]), (1 + patch[1][1] - patch[1][0])

    @staticmethod
    def _patch_data(source, patch):
        return source[
               patch[0][0]:patch[0][1] + 1,
               patch[1][0]:patch[1][1] + 1
               ]

    @staticmethod
    def _copy_to_patch(dest, dest_patch, data):
        dest[
        dest_patch[0][0]:dest_patch[0][1] + 1,
        dest_patch[1][0]:dest_patch[1][1] + 1
        ] = data

    @staticmethod
    def _to_rgb(image):
        height, width = image.shape  # Height480 * Width360
        return image.reshape(height, width, 1).repeat(3, axis=2)