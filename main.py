import cv2
import numpy as np
import utils

########PARAMS########
path = "1.jpeg"
widthImage = 700
heightImage = 700
number_of_questions = 5
number_of_choices = 5
answers = [1, 2, 0, 0, 3]
########################

image = cv2.imread(path)

########PREPROCESSING########
image = cv2.resize(image, (widthImage, heightImage))
imageContours = image.copy()
image_biggest_contours = image.copy()
imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
imageBlur = cv2.GaussianBlur(imageGray, (5, 5), 1)
imageCanny = cv2.Canny(imageBlur, 10, 50)
########

########FINDING ALL CONTOURS########
contours, hierarchy = cv2.findContours(
    imageCanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

cv2.drawContours(imageContours, contours, -1, (0, 255, 0), 5)

########FIND RECTANGLES########
rectangle_contours = utils.find_rectangle_contours(contours)
biggest_contour = utils.get_corner_points(rectangle_contours[0])
grade_points = utils.get_corner_points(rectangle_contours[1])

if biggest_contour.size != 0 and grade_points.size != 0:
    cv2.drawContours(image_biggest_contours,
                     biggest_contour, -1, (0, 255, 0), 20)
    cv2.drawContours(image_biggest_contours,
                     grade_points, -1, (255, 0, 0), 20)

    biggest_contour = utils.reorder(biggest_contour)
    grade_points = utils.reorder(grade_points)

    points_one = np.float32(biggest_contour)
    points_two = np.float32(
        [[0, 0], [widthImage, 0], [0, heightImage], [widthImage, heightImage]])
    matrix = cv2.getPerspectiveTransform(points_one, points_two)
    imageWarpColored = cv2.warpPerspective(
        image, matrix, (widthImage, heightImage))

    points_grade_one = np.float32(grade_points)
    points_grade_two = np.float32(
        [[0, 0], [325, 0], [0, 150], [325, 150]])
    grade_matrix = cv2.getPerspectiveTransform(
        points_grade_one, points_grade_two)
    image_grade_display = cv2.warpPerspective(
        image, grade_matrix, (325, 150))

    ########APPLY THRESHOLD########
    imageWarpGray = cv2.cvtColor(imageWarpColored, cv2.COLOR_BGR2GRAY)
    imageThresh = cv2.threshold(
        imageWarpGray, 200, 255, cv2.THRESH_BINARY_INV)[1]
    boxes = utils.split_boxes(imageThresh)

    ########GETTING NON ZERO PIXEL VALUES OF EACH BOX########
    my_pixel_values = np.zeros((number_of_questions, number_of_choices))
    count_columns = 0
    count_rows = 0
    for idx, box in enumerate(boxes):
        total_pixels = cv2.countNonZero(box)
        my_pixel_values[count_rows][count_columns] = total_pixels
        count_columns += 1
        if count_columns == number_of_choices:
            count_rows += 1
            count_columns = 0

    print(np.matrix(my_pixel_values))

    my_index = []

    ########FINDING INDEX VALUES OF THE MARKINGS########
    for alternative in range(0, number_of_questions):
        arr = my_pixel_values[alternative]
        my_index_val = np.where(arr == np.amax(arr))
        my_index.append(my_index_val[0][0])

    ########GRADING########
    grading = []

    for x in range(0, number_of_questions):
        if answers[x] == my_index[x]:
            grading.append(1)
        else:
            grading.append(0)
    score = (sum(grading)/number_of_questions)*100

    ########DISPLAYING THE ANSWERS########
    imageResult = imageWarpColored.copy()
    imageResult = utils.show_answers(
        imageResult, my_index, grading, answers, number_of_questions, number_of_choices)
    imageRawDrawing = np.zeros_like(imageWarpColored)
    imageRawDrawing = utils.show_answers(
        imageRawDrawing, my_index, grading, answers, number_of_questions, number_of_choices)
    inverseMatrix = cv2.getPerspectiveTransform(points_two, points_one)
    imageInverseWarp = cv2.warpPerspective(
        imageRawDrawing, inverseMatrix, (widthImage, heightImage))

    imageRawGrade = np.zeros_like(image_grade_display)
    cv2.putText(imageRawGrade, str(int(score)) + "%",
                (50, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 125, 125), 5)
    cv2.imshow("Grade Display", imageRawGrade)
    inverseGradeMatrixGrade = cv2.getPerspectiveTransform(
        points_grade_two, points_grade_one)
    imageInverseGradeDisplay = cv2.warpPerspective(
        imageRawGrade, inverseGradeMatrixGrade, (widthImage, heightImage))

    imageFinal = cv2.addWeighted(image, 1, imageInverseWarp, 1, 0)
    imageFinal = cv2.addWeighted(imageFinal, 1, imageInverseGradeDisplay, 1, 0)


imageBlank = np.zeros_like(image)
imageArray = ([image, imageGray, imageBlur, imageCanny], [
    imageContours, image_biggest_contours, imageWarpColored, imageThresh], [imageResult, imageRawDrawing, imageInverseWarp, imageFinal])
imageStacked = utils.stack_images(imageArray, 0.3)

cv2.imshow("Final Result", imageFinal)
cv2.imshow("Stacked Images", imageStacked)
cv2.waitKey(0)
