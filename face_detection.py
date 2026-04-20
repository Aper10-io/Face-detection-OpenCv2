import cv2

face_ref = cv2.CascadeClassifier("face_ref.xml")
camera = cv2.VideoCapture(0)
print("Tekan 'q' untuk keluar")
if not camera.isOpened():
    print("Tidak dapat membuka kamera.")
    exit()

def face_detection(frame):
    grey_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_ref.detectMultiScale(grey_frame, scaleFactor=1.1, minNeighbors=3 ,minSize=(30, 30))
    return faces

def drawer_box(frame, faces_detection, name="Wajah"):
    font = cv2.FONT_HERSHEY_SIMPLEX
    color = (255, 0, 0) 
    thickness = 2
    scale = 0.8

    for x, y, w, h in faces_detection:
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, thickness)
        text_pos = (x, y - 10)
        cv2.putText(frame, name, text_pos, font, scale, color, thickness, cv2.LINE_AA)

def close_window():
    camera.release()
    cv2.destroyAllWindows()
    exit()
    
def main():
    while True:
        frame = camera.read()[1]
        frame = cv2.flip(frame, 1) 
        faces = face_detection(frame)
        drawer_box(frame, faces, name="Haesam") 
        
        cv2.imshow("Face Detection", frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            close_window() 
            break

if __name__ == "__main__":
    main()