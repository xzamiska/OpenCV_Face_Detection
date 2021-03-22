from dataclasses import dataclass
import cv2
from mtcnn import MTCNN

# global glob_deep_tp
# global glob_deep_fp
# global glob_deep_fn
#
# global glob_viola_tp
# global glob_viola_fp
# global glob_viola_fn
glob_deep_tp = 0
glob_deep_fp = 0
glob_deep_fn = 0

glob_viola_tp = 0
glob_viola_fp = 0
glob_viola_fn = 0

@dataclass
class Face:
    x: int
    y: int
    w: int
    h: int


@dataclass
class ImgData:
    filename: str
    count: int
    faces: []


def get_iou(x, y, w, h, true_data: ImgData, isViola):
    scores = []
    founded = []

    for item in true_data.faces:
        x_left = max(x, item.x)
        y_top = max(y, item.y)
        x_right = min(w + x, item.w + item.x)
        y_bottom = min(h + y, item.h + item.y)

        intersection_area = max(x_right - x_left, 0) * max(y_bottom - y_top, 0)

        bb1_area = w * h
        bb2_area = item.w * item.h

        iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
        scores.append(iou)
        founded.append(iou > 0)

    if isViola == False:
        cv2.putText(frame, "IoU deep: {:.2f}".format(max(scores)), (x, y - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 155, 255), 2)
    else:
        cv2.putText(frame, "IoU viola: {:.2f}".format(max(scores)), (x, y + h + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 24, 159), 2)

    return max(scores), founded


def calc_predicsion_and_recall(scores, annot, isViola):
    tp = len(list(filter(lambda x: x >= 0.5, scores)))
    fp = len(scores) - tp
    fn = annot.count - tp
    if tp + fp != 0:
        Precision = tp / (tp + fp)
    else:
        Precision = 0
    Recall = tp / (tp + fn)
    if isViola == False:
        cv2.putText(frame, "DEEP - Precision: {:.2f}".format(Precision) + " Recall: {:.2f}".format(Recall), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 155, 255), 2)
    else:
        cv2.putText(frame, "VIOLA - Precision: {:.2f}".format(Precision) + " Recall: {:.2f}".format(Recall), (10, 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (98, 24, 159), 2)

    return tp, fp, fn


def find_face_DEEP(frame, faces, with_score, annot):
    global glob_deep_tp, glob_deep_fp, glob_deep_fn
    scores = []
    found_ind = []
    founded_arr = []

    if faces != []:
        for person in faces:
            x, y, w, h = person['box']
            # keypoints = person['keypoints']
            cv2.rectangle(frame,
                          (x, y),
                          (x + w, y + h),
                          (0, 155, 255),
                          5)

            if with_score:
                score, found_ind = get_iou(x, y, w, h, annot, False)
                scores.append(score)
                if len(founded_arr) == 0:
                    founded_arr = found_ind
                else:
                    for ind in range(len(founded_arr)):
                        founded_arr[ind] = founded_arr[ind] | found_ind[ind]
    if with_score:
        TP, FP, FN = calc_predicsion_and_recall(scores, annot, False)
        glob_deep_tp += TP
        glob_deep_fp += FP
        glob_deep_fn += FN
    return frame, founded_arr


def find_face_viola(frame, faces, with_score, annot):
    global glob_viola_tp, glob_viola_fp, glob_viola_fn
    scores = []
    found_ind = []
    founded_arr = []
    if with_score:
        frame_copy = frame
    else:
        frame_copy = cv2.GaussianBlur(frame, (101, 101), 0)
    for x, y, w, h in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), (98, 24, 159), 3)

        if with_score:
            score, found_ind = get_iou(x, y, w, h, annot, True)
            scores.append(score)
            if len(founded_arr) == 0:
                founded_arr = found_ind
            else:
                for ind in range(len(founded_arr)):
                    founded_arr[ind] = founded_arr[ind] | found_ind[ind]
        else:
            frame_copy[y:y + h, x:x + w] = frame[y:y + h, x:x + w]
    if with_score:
        TP, FP, FN = calc_predicsion_and_recall(scores, annot, True)
        glob_viola_tp += TP
        glob_viola_fp += FP
        glob_viola_fn += FN
    return frame_copy, founded_arr


def read_data():
    f = open("data-faces-wider/annotations.txt", "r")
    data = []
    for x in f:
        if x.find(".jpg") != -1:
            data.append(ImgData("", 0, []))
            data[len(data) - 1].filename = x.rstrip('\n')
        elif len(x.rstrip('\n')) <= 2:
            data[len(data) - 1].count = int(x)
        else:
            faceData = x.rstrip('\n').split(',')
            face = Face(int(faceData[0]), int(faceData[1]), int(faceData[2]), int(faceData[3]))
            data[len(data) - 1].faces.append(face)
    f.close()
    return data


annot_data = read_data()
# glob_deep_tp = 0
# glob_deep_fp = 0
# glob_deep_fn = 0
#
# glob_viola_tp = 0
# glob_viola_fp = 0
# glob_viola_fn = 0
if __name__ == '__main__':
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    detector = MTCNN()
    while 1:
        k = cv2.waitKey(1) & 0XFF
        if k == 27:
            break

        check, frame_cam = video.read()
        result = detector.detect_faces(frame_cam)
        deep_faces, found_deep = find_face_DEEP(frame_cam, result, False, {})
        faces = face_cascade.detectMultiScale(frame_cam, scaleFactor=1.1, minNeighbors=5)
        viola_faces, found_viola = find_face_viola(frame_cam, faces, False, {})
        cv2.imshow('Face Detector', viola_faces)

    video.release()
    cv2.destroyAllWindows()

    for image in annot_data:
        frame = cv2.imread("data-faces-wider/" + image.filename)
        result = detector.detect_faces(frame)
        deep_faces, found_deep = find_face_DEEP(frame, result, True, image)
        faces = face_cascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=5)
        detectFaceViola, found_viola = find_face_viola(frame, faces, True, image)
        for face in image.faces:
            cv2.rectangle(frame,
                          (face.x, face.y), (face.x + face.w, face.y + face.h),
                          (210, 10, 52),
                          2)

        if len(found_deep) == len(found_viola):
            for indexis in range(len(found_deep)):
                print(indexis)
                print(found_deep)
                print(found_viola)
                if found_deep[indexis] is False & found_viola[indexis] is False:
                    cv2.imwrite("output/drawn_" + image.filename, detectFaceViola)
                    break
        else:
            cv2.imwrite("output/drawn_" + image.filename, detectFaceViola)

    print(glob_deep_tp)
    print(glob_deep_fp)
    print(glob_deep_fn)
    deep_precision = glob_deep_tp / (glob_deep_tp + glob_deep_fp)
    deep_recall = glob_deep_tp / (glob_deep_tp + glob_deep_fn)
    print('MTCNN Precision:', deep_precision)
    print('MTCNN Recall:', deep_recall)
    print(glob_viola_tp)
    print(glob_viola_fp)
    print(glob_viola_fn)
    viola_precision = glob_viola_tp / (glob_viola_tp + glob_viola_fp)
    viola_recall = glob_viola_tp / (glob_viola_tp + glob_viola_fn)
    print('Viola Precision:', viola_precision)
    print('Viola Recall:', viola_recall)
