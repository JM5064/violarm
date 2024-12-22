import cv2


class Video():
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source)
        self.frame = self.update()


    def update(self):
        ret, self.frame = self.cap.read()


    def read(self):
        return self.frame
    
