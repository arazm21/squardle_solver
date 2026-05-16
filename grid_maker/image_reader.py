import cv2


def read_image(path, debug = False):
    img = cv2.imread(path)

    if img is None:
        print("image read error")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("image read")

    if debug:
        cv2.imshow("Loaded Image", gray)

        cv2.waitKey(0)

        cv2.destroyAllWindows()

    return gray