import cv2
import threading


class Video():
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(source)
        self.frame = None

        self.thread = threading.Thread(target=self.update, daemon=True)
        self.thread.start()


    def isOpened(self):
        return self.cap.isOpened()
    

    def release(self):
        self.cap.release()


    def update(self):
        while self.isOpened():
            ret, self.frame = self.cap.read()


    def read(self):
        return self.frame
    
