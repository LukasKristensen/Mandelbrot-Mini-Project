import cv2
import matplotlib.pyplot as plt


class VideoCreator:
    def __init__(self, output_video_destination, frame_rate, pRE, pIM):
        self.video = cv2.VideoWriter(output_video_destination, cv2.VideoWriter_fourcc(*'DIVX'), frame_rate, (pRE, pIM))
        self.pRE = pRE
        self.pIM = pIM
        self.frame_rate = frame_rate
        self.output_video_destination = output_video_destination

    def create_frame(self, frame, text):
        plt.figure(figsize=(self.pRE, self.pIM), dpi=1)
        plt.imshow(frame, cmap='magma', interpolation='nearest', aspect='auto')
        plt.axis('off')
        plt.bbox_inches = 'tight'
        plt.pad_inches = 0
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(f'tmp_hold.png', bbox_inches='tight', pad_inches=0)
        plt.close()

        loaded_image = cv2.imread('tmp_hold.png')

        for i in range(len(text)):
            cv2.putText(loaded_image, text[i], (10, 20 + 20 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

        self.save_frame(loaded_image)

    def save_frame(self, frame):
        self.video.write(frame)

    def close(self):
        self.video.release()

