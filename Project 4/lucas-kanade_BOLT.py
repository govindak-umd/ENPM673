import numpy as np
import cv2
import glob
from scipy.ndimage import affine_transform
import math

img_array = []
for filename in glob.glob('Bolt2/Bolt2/img/*.jpg'):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    img_array.append(img)


first_image = img_array[0]
# # now let's initialize the list of reference point
ref_point = []
crop = False

def LucasKanadeAffine(It, It1, threshold=0.005, iters=10):
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
        #error_img = It-warp_img
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
if pyramid_layers ==1:
    remultiply_factor = 1
    divide_factror = 1
else:
    remultiply_factor = 2**pyramid_layers
    divide_factror = float(1/remultiply_factor)

def Average(lst):
    return sum(lst) / len(lst)

##bolt
rect_coordinates = [(int(divide_factror*269),int(divide_factror*75)), (int(divide_factror*303), int(divide_factror*139))]
rect_coordinates_orig = [(269,75), (303,139)]


MAIN_WIDTH = 303 - 269
MAIN_HEIGHT = 139 - 75

rect = np.array([rect_coordinates[0][0], rect_coordinates[0][1], rect_coordinates[1][0], rect_coordinates[1][1]])

rect1 = np.reshape(np.array([rect_coordinates[0][0], rect_coordinates[0][1], 1]), (3, 1))
rect2 = np.reshape(np.array([rect_coordinates[1][0], rect_coordinates[1][1], 1]), (3, 1))

first_image = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
template = first_image[rect_coordinates_orig[0][1] : rect_coordinates_orig[1][1], rect_coordinates_orig[0][0] : rect_coordinates_orig[1][0]]

first_image = cv2.pyrDown(first_image)
first_image = cv2.pyrDown(first_image)
first_image = cv2.pyrDown(first_image)

main_count = 0

img_list = []


for next_img in img_array:

    next_img_BGR = next_img.copy()
    next_img = cv2.cvtColor(next_img, cv2.COLOR_BGR2GRAY)

    next_img = cv2.pyrDown(next_img)
    next_img = cv2.pyrDown(next_img)
    next_img = cv2.pyrDown(next_img)


    if main_count ==16:
        rect_coordinates = [(int(divide_factror * 251), int(divide_factror * 65)),
                            (int(divide_factror * 289), int(divide_factror * 129))]
        rect = np.array(
            [rect_coordinates[0][0], rect_coordinates[0][1], rect_coordinates[1][0], rect_coordinates[1][1]])

        rect1 = np.reshape(np.array([rect_coordinates[0][0], rect_coordinates[0][1], 1]), (3, 1))
        rect2 = np.reshape(np.array([rect_coordinates[1][0], rect_coordinates[1][1], 1]), (3, 1))


    if main_count ==135:
        rect_coordinates = [(int(divide_factror * 308), int(divide_factror * 82)),
                            (int(divide_factror * 342), int(divide_factror * 146))]
        rect = np.array(
            [rect_coordinates[0][0], rect_coordinates[0][1], rect_coordinates[1][0], rect_coordinates[1][1]])

        rect1 = np.reshape(np.array([rect_coordinates[0][0], rect_coordinates[0][1], 1]), (3, 1))
        rect2 = np.reshape(np.array([rect_coordinates[1][0], rect_coordinates[1][1], 1]), (3, 1))

    if main_count ==193:
        rect_coordinates = [(int(divide_factror * 335), int(divide_factror * 79)),
                            (int(divide_factror * 369), int(divide_factror * 143))]
        rect = np.array(
            [rect_coordinates[0][0], rect_coordinates[0][1], rect_coordinates[1][0], rect_coordinates[1][1]])

        rect1 = np.reshape(np.array([rect_coordinates[0][0], rect_coordinates[0][1], 1]), (3, 1))
        rect2 = np.reshape(np.array([rect_coordinates[1][0], rect_coordinates[1][1], 1]), (3, 1))

    if main_count ==279:
        rect_coordinates = [(int(divide_factror * 359), int(divide_factror * 104)),
                            (int(divide_factror * 393), int(divide_factror * 168))]
        rect = np.array(
            [rect_coordinates[0][0], rect_coordinates[0][1], rect_coordinates[1][0], rect_coordinates[1][1]])

        rect1 = np.reshape(np.array([rect_coordinates[0][0], rect_coordinates[0][1], 1]), (3, 1))
        rect2 = np.reshape(np.array([rect_coordinates[1][0], rect_coordinates[1][1], 1]), (3, 1))

    if main_count ==289:
        rect_coordinates = [(int(divide_factror * 368), int(divide_factror * 110)),
                            (int(divide_factror * 392), int(divide_factror * 174))]
        rect = np.array(
            [rect_coordinates[0][0], rect_coordinates[0][1], rect_coordinates[1][0], rect_coordinates[1][1]])

        rect1 = np.reshape(np.array([rect_coordinates[0][0], rect_coordinates[0][1], 1]), (3, 1))
        rect2 = np.reshape(np.array([rect_coordinates[1][0], rect_coordinates[1][1], 1]), (3, 1))

    p = LucasKanadeAffine(first_image, next_img)
    newrect1 = np.matmul(p, rect1)
    newrect2 = np.matmul(p, rect2)

    x_1 = int(remultiply_factor*newrect1[0])
    y_1 = int(remultiply_factor*newrect1[1])

    cv2.rectangle(next_img_BGR, (x_1,y_1), (x_1 + MAIN_WIDTH,y_1+MAIN_HEIGHT), (255, 255, 0), 2)
    cv2.imwrite("all_new_imgs_BOLT/%d.jpg" % main_count, next_img_BGR)

    img_list.append(next_img_BGR)
    print(main_count)
    main_count += 1

out = cv2.VideoWriter('all_new_imgs_BOLT/bolt.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (480, 270))
for image in img_list:
    out.write(image)
    cv2.waitKey(10)


out.release()