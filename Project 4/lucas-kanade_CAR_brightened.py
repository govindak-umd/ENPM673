import numpy as np
import cv2
import glob
from scipy.ndimage import affine_transform
import math

def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)])
    return cv2.LUT(image.astype(np.uint8), table.astype(np.uint8))

def img_improve (img,gamma):
    blurred_img = cv2.GaussianBlur(img, (7, 7), 0)
    # converting the image to HSV
    img2hsv = cv2.cvtColor(blurred_img, cv2.COLOR_BGR2HSV)
    hsv_v = img2hsv[:, :, 2]
    # finding the CLAHE
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(16, 16))  # increased values may cause noise
    cl1 = clahe.apply(hsv_v)
    # setting the gamma value, increased values may cause noise
    cl1 = adjust_gamma(cl1, gamma=gamma)
    # adding the last V layer back to the HSV image
    img2hsv[:, :, 2] = cl1
    # converting back from HSV to BGR format
    improved_image = cv2.cvtColor(img2hsv, cv2.COLOR_HSV2BGR)
    return improved_image

img_array = []
for filename in glob.glob('Car4/Car4/img/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)


first_image = img_array[0]
# # now let's initialize the list of reference point
ref_point = []
crop = False

def LucasKanadeAffine(It, It1, threshold=0.005, iters=100):
    '''
    [input]
    * It - Template image
    * It1 - Current image
    * threshold - Threshold for error convergence (default: 0.005)
    * iters - Number of iterations for error convergence (default: 50)

    [output]
    * M - Affine warp matrix [2x3 numpy array]
    '''

    # Initial parameters
    M = np.asarray([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    p = np.asarray([0.0] * 6)
    I = M

    # Iterate
    # for i in range(iters):
        # Step 1 - Warp image
    for i in range(iters):
        warp_img = affine_transform(It1, np.flip(M)[..., [1, 2, 0]])

        # Step 2 - Compute error image with common pixels
        mask = affine_transform(np.ones(It1.shape), np.flip(M)[..., [1, 2, 0]])
        error_img = (mask * It) - (mask * warp_img)
        # Step 3 - Compute and warp the gradient
        gradient = np.dstack(np.gradient(It1)[::-1])
        gradient[:, :, 0] = affine_transform(gradient[:, :, 0], np.flip(M)[..., [1, 2, 0]])
        gradient[:, :, 1] = affine_transform(gradient[:, :, 1], np.flip(M)[..., [1, 2, 0]])
        warp_gradient = gradient.reshape(gradient.shape[0] * gradient.shape[1], 2).T

        # Step 4 - Evaluate jacobian parameters
        H, W = It.shape
        Jx = np.tile(np.linspace(0, W - 1, W), (H, 1)).flatten()
        Jy = np.tile(np.linspace(0, H - 1, H), (W, 1)).T.flatten()

        # Step 5 - Compute the steepest descent images
        steepest_descent = np.vstack([warp_gradient[0] * Jx, warp_gradient[0] * Jy,
                                      warp_gradient[0], warp_gradient[1] * Jx, warp_gradient[1] * Jy,
                                      warp_gradient[1]]).T

        # Step 6 - Compute the Hessian matrix
        hessian = np.matmul(steepest_descent.T, steepest_descent)

        # Step 7/8 - Compute delta P
        delta_p = np.matmul(np.linalg.inv(hessian), np.matmul(steepest_descent.T, error_img.flatten()))

        # Step 9 - Update the parameters
        p = p + delta_p
        M = p.reshape(2, 3) + I
        # Test for convergence
        if np.linalg.norm(delta_p) <= threshold:
            break


    return M

pyramid_layers = 3
remultiply_factor = 2**pyramid_layers
divide_factror = float(1/remultiply_factor)
def Average(lst):
    return sum(lst) / len(lst)

##car
rect_coordinates = [(int(divide_factror*81),int(divide_factror*56)), (int(divide_factror*166), int(divide_factror*123))]
rect_coordinates_orig = [(81,56), (166,123)]

MAIN_WIDTH = 166 - 81
MAIN_HEIGHT = 123 - 56

rect = np.array([rect_coordinates[0][0], rect_coordinates[0][1], rect_coordinates[1][0], rect_coordinates[1][1]])

rect1 = np.reshape(np.array([rect_coordinates[0][0], rect_coordinates[0][1], 1]), (3, 1))
rect2 = np.reshape(np.array([rect_coordinates[1][0], rect_coordinates[1][1], 1]), (3, 1))
first_image = img_improve(first_image, 1.4)
first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
first_image = cv2.equalizeHist(first_image)

template = first_image[rect_coordinates_orig[0][1] : rect_coordinates_orig[1][1], rect_coordinates_orig[0][0] : rect_coordinates_orig[1][0]]

#finding average pixel values in the template:
pixel_vals = []
for i in first_image:
    for c in i:
        pixel_vals.append(c)
average_template_val = Average(pixel_vals)
print ('average_template_val pixel values : ',average_template_val)


first_image = cv2.pyrDown(first_image)
first_image = cv2.GaussianBlur(first_image,(7,7),0)
first_image = cv2.pyrDown(first_image)
first_image = cv2.GaussianBlur(first_image,(3,3),0)
first_image = cv2.pyrDown(first_image)



main_count = 0

img_list = []

for next_img in img_array:

    next_img_untouched = next_img.copy()


    next_img_BGR = next_img.copy()
    next_img_BGR_grayed = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

    pixel_vals =[]
    for i in next_img_BGR_grayed:
        for c in i:
            pixel_vals.append(c)
    average_image_brightness = Average(pixel_vals)
    # print('CURRENT: average_template_val pixel values : ', average_image_brightness)

    count_up = 1.0

    #BRIGHTNESS CHANGED BY LOOPING THROUGH UNTIL THE BRIGHTNESS OF THE TEMPLATE IS REACHED

    while average_image_brightness < 126:
        # print('LOOPING .. >',average_image_brightness)
        next_img_BGR = img_improve(next_img_BGR,count_up)
        pixel_vals_loop =[]
        next_img_BGR_grayed = cv2.cvtColor(next_img_BGR, cv2.COLOR_BGR2GRAY)
        for i in next_img_BGR_grayed:
            for c in i:
                pixel_vals_loop.append(c)
        average_image_brightness = Average(pixel_vals_loop)
        # print('average_image_brightness',average_image_brightness,'gamma is now > ',count_up)
        count_up+=0.2

    count_down = 1.0
    while average_image_brightness > 132.0:
        # print('LOOPING .. >',average_image_brightness)
        next_img_BGR = img_improve(next_img_BGR,count_down)
        pixel_vals_loop =[]
        next_img_BGR_grayed = cv2.cvtColor(next_img_BGR, cv2.COLOR_BGR2GRAY)
        for i in next_img_BGR_grayed:
            for c in i:
                pixel_vals_loop.append(c)
        average_image_brightness = Average(pixel_vals_loop)
        # print('average_image_brightness',average_image_brightness, 'gamma is now > ',count_down)
        count_down-=0.2



    next_img = cv2.pyrDown(next_img_BGR_grayed)
    next_img = cv2.pyrDown(next_img)
    next_img = cv2.pyrDown(next_img)
    next_img = cv2.equalizeHist(next_img)

    if main_count ==230:
        rect_coordinates = [(int(divide_factror * 158), int(divide_factror * 68)),
                            (int(divide_factror * 228), int(divide_factror * 119))]
        rect = np.array(
            [rect_coordinates[0][0], rect_coordinates[0][1], rect_coordinates[1][0], rect_coordinates[1][1]])

        rect1 = np.reshape(np.array([rect_coordinates[0][0], rect_coordinates[0][1], 1]), (3, 1))
        rect2 = np.reshape(np.array([rect_coordinates[1][0], rect_coordinates[1][1], 1]), (3, 1))

    if main_count ==248:
        rect_coordinates = [(int(divide_factror * 171), int(divide_factror * 67)),
                            (int(divide_factror * 239), int(divide_factror * 119))]
        rect = np.array(
            [rect_coordinates[0][0], rect_coordinates[0][1], rect_coordinates[1][0], rect_coordinates[1][1]])

        rect1 = np.reshape(np.array([rect_coordinates[0][0], rect_coordinates[0][1], 1]), (3, 1))
        rect2 = np.reshape(np.array([rect_coordinates[1][0], rect_coordinates[1][1], 1]), (3, 1))

    if main_count ==331:
        rect_coordinates = [(int(divide_factror * 216), int(divide_factror * 72)),
                            (int(divide_factror * 272), int(divide_factror * 118))]
        rect = np.array(
            [rect_coordinates[0][0], rect_coordinates[0][1], rect_coordinates[1][0], rect_coordinates[1][1]])

        rect1 = np.reshape(np.array([rect_coordinates[0][0], rect_coordinates[0][1], 1]), (3, 1))
        rect2 = np.reshape(np.array([rect_coordinates[1][0], rect_coordinates[1][1], 1]), (3, 1))

    if main_count ==385:
        rect_coordinates = [(int(divide_factror * 228), int(divide_factror * 71)),
                            (int(divide_factror * 289), int(divide_factror * 119))]
        rect = np.array(
            [rect_coordinates[0][0], rect_coordinates[0][1], rect_coordinates[1][0], rect_coordinates[1][1]])

        rect1 = np.reshape(np.array([rect_coordinates[0][0], rect_coordinates[0][1], 1]), (3, 1))
        rect2 = np.reshape(np.array([rect_coordinates[1][0], rect_coordinates[1][1], 1]), (3, 1))


    if main_count ==596:
        rect_coordinates = [(int(divide_factror * 225), int(divide_factror * 63)),
                            (int(divide_factror * 308), int(divide_factror * 126))]
        rect = np.array(
            [rect_coordinates[0][0], rect_coordinates[0][1], rect_coordinates[1][0], rect_coordinates[1][1]])

        rect1 = np.reshape(np.array([rect_coordinates[0][0], rect_coordinates[0][1], 1]), (3, 1))
        rect2 = np.reshape(np.array([rect_coordinates[1][0], rect_coordinates[1][1], 1]), (3, 1))

    if main_count ==637:
        rect_coordinates = [(int(divide_factror * 260), int(divide_factror * 62)),
                            (int(divide_factror * 344), int(divide_factror * 137))]
        rect = np.array(
            [rect_coordinates[0][0], rect_coordinates[0][1], rect_coordinates[1][0], rect_coordinates[1][1]])

        rect1 = np.reshape(np.array([rect_coordinates[0][0], rect_coordinates[0][1], 1]), (3, 1))
        rect2 = np.reshape(np.array([rect_coordinates[1][0], rect_coordinates[1][1], 1]), (3, 1))

    p = LucasKanadeAffine(first_image, next_img)
    newrect1 = np.matmul(p, rect1)
    newrect2 = np.matmul(p, rect2)
    x_1 = int(remultiply_factor*newrect1[0])
    y_1 = int(remultiply_factor*newrect1[1])
    cv2.rectangle(next_img_untouched, (x_1,y_1), (x_1 + MAIN_WIDTH,y_1+MAIN_HEIGHT), (0, 0, 255), 2)
    cv2.imwrite("all_new_imgs_CAR/%d.jpg" % main_count, next_img_untouched)

    img_list.append(next_img_untouched)
    print(main_count)
    main_count += 1



out = cv2.VideoWriter('all_new_imgs_CAR/CAR.avi', cv2.VideoWriter_fourcc(*'XVID'), 15.0, (360, 240))
for image in img_list:
    out.write(image)
    cv2.waitKey(10)

out.release()