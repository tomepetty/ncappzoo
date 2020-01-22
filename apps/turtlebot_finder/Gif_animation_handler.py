import cv2


class Gif_animation_handler:

    def __init__(self, video_window_name, window_width, window_height):
        self.video_window_name = video_window_name
        cv2.namedWindow(self.video_window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.video_window_name, window_width, window_height)
        self.IDLE_GIF = "animations/robot_idle.gif"        # video for animation at idle
        self.DETECTED_GIF = "animations/robot_happy.gif"   # video for animation when detecting an object
        self.video_capture = None
        self.frame_limit = 1
        self.frame_counter = 0           # Integer. Counts the current number of frames read in a GIF.
        self.blink_frame_counter = 0     # Integer. A sort of countdown frame counter. See animation_handle.py.


    def determine_animations(self, turtle_bot, obj_x_coord):        
        self.frame_counter += 1
        self.blink_frame_counter += 1 
        
        #### Idle animation
        if (obj_x_coord == -1 or turtle_bot.get_vel() != 0):
            # This code makes the gif loop correctly else the animations just blink repeatedly
            # decide if it is time to play entire animation
                
            if (self.blink_frame_counter >= 50):
                self.frame_limit = 12        # play the entire animation with "blinking" animation
                self.blink_frame_counter = 0
            
            # if the frame limit has been reached, restart the animation 
            if (self.frame_counter >= self.frame_limit):
                self.video_capture = cv2.VideoCapture(self.IDLE_GIF)
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0) 
                self.frame_limit = 1         # only show the first frame
                self.frame_counter = 0
                
        #### Detected an object animation 
        elif (obj_x_coord > -1 and turtle_bot.get_vel() == 0):
            # decide if it is time to play the entire animation

            if (self.blink_frame_counter >= 50):
                self.frame_limit = 12        # plays entire animation
                self.blink_frame_counter = 0       # resets the frame delay counter
            
            # if the frame limit has been reached, restart the animation 
            if (self.frame_counter >= self.frame_limit):
                self.video_capture = cv2.VideoCapture(self.DETECTED_GIF)
                self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                self.frame_limit = 5         # only show first 5 frames
                self.frame_counter = 0
                
        # Read an image from the animation video
        ret_val, self.video_image = self.video_capture.read()
        if (ret_val != True):
            print("Error: Can't read frame from video file!")
            exit(1)
            
        # display the video frame in the video window
        cv2.imshow(self.video_window_name, self.video_image)
        if (cv2.waitKey(1) & 0xFF == ord( 'q' )):
            exit(0)
    

