import cv2 as cv
import numpy as np

def menu():
    while True:
        print("Menu:")
        print("1. Create a white background")
        print("2. Draw rectangle")
        print("3. Translation transformation")
        print("4. Rotation transformation")
        print("5. Scaling transformation")
        print("6. Exit")
        print("Please select an function:", end = '')
        try:
            choice = int(input())
            if 0 < choice < 7 :
                return choice
            else:
                print("Invalid choice")
        except:
            print("Invalid choice")

def init_window(name, bg = (0,0,0),width = 600, height = 800):
    cv.namedWindow(name, cv.WINDOW_NORMAL)
    cv.imshow(name, bg)

def refresh_window():
    global name, bg, rect_coor
    #cv.destroyWindow(frame)
    bg = np.ones((600, 800, 3), np.uint8) * 255
    if rect_coor[0][0] != -1:
        pts = np.array(rect_coor)
        cv.fillPoly(bg, pts = [pts], color=(0, 255, 0))
    cv.imshow(name,bg)

def white():
    global name, bg, rect_coor
    bg = np.ones((600,800,3), np.uint8)*255
    rect_coor = [[-1,-1],
                 [-1,-1],
                 [-1,-1],
                 [-1,-1]]
    cv.imshow(name, bg)

def draw_rect(event,x,y,flags,param):
    global rect_coor,bg, draw, rect_coor
    if(event == cv.EVENT_LBUTTONDOWN):
        bg = np.ones((600, 800, 3), np.uint8)*255
        bg = cv.putText(bg, "Enter to confirm", (488, 561), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2, cv.LINE_AA)
        draw = True
        rect_coor[0] = x,y
    elif (event == cv.EVENT_MOUSEMOVE):
        if draw == True:
            bg = np.ones((600, 800, 3), np.uint8) * 255
            bg = cv.putText(bg, "Enter to confirm", (488, 561), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2, cv.LINE_AA)
            cv.rectangle(bg,rect_coor[0],(x,y),(0,255,0),-1)
    elif(event == cv.EVENT_LBUTTONUP):
        draw = False
        rect_coor[2] = x,y
        bg = np.ones((600, 800, 3), np.uint8) * 255
        bg = cv.putText(bg, "Enter to confirm", (488, 561), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2, cv.LINE_AA)
        cv.rectangle(bg,rect_coor[0],rect_coor[2],(0,255,0),-1)
        print(rect_coor)

def rectangle():
    global name,bg
    cv.setMouseCallback(name, draw_rect)
    while True:
        text = True
        bg = cv.putText(bg, "Enter to confirm", (488, 561), cv.FONT_HERSHEY_PLAIN, 1.5, (255, 0, 0), 2, cv.LINE_AA)
        cv.imshow(name,bg)
        k = cv.waitKey(1) & 0xFF
        if k == ord('\r'):
            calc_rect()
            break

def translation():
    global rect_coor
    if rect_coor[0][0] == -1:
        print("Please draw a rectangle first")
        return
    tx = int(input("Enter tx: "))
    ty = int(input("Enter ty: "))
    (x1,y1),(x2,y2),(x3,y3),(x4,y4) = rect_coor
    x1 += tx
    x2 += tx
    x3 += tx
    x4 += tx
    y1 += ty
    y2 += ty
    y3 += ty
    y4 += ty
    rect_coor = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]

def rotation():
    global rect_coor
    if rect_coor[0][0] == -1:
        print("Please draw a rectangle first")
        return
    (x1,y1),(x2,y2),(x3,y3),(x4,y4) = rect_coor
    angle = np.radians(int(input("Enter angle: ")))
    center = [int((x1 + x3) / 2), int((y1 + y3) / 2)]
    rot = [[np.cos(angle), np.sin(angle)],
           [-np.sin(angle), np.cos(angle)]]
    vector = [[x1-center[0],y1-center[1]],
                [x2-center[0],y2-center[1]],
                [x3-center[0],y3-center[1]],
                [x4-center[0],y4-center[1]]]
    print(vector)
    rot_v = [[vector[0][0]*rot[0][0]+vector[0][1]*rot[0][1],vector[0][0]*rot[1][0]+vector[0][1]*rot[1][1]],
            [vector[1][0]*rot[0][0]+vector[1][1]*rot[0][1],vector[1][0]*rot[1][0]+vector[1][1]*rot[1][1]],
            [vector[2][0]*rot[0][0]+vector[2][1]*rot[0][1],vector[2][0]*rot[1][0]+vector[2][1]*rot[1][1]],
            [vector[3][0]*rot[0][0]+vector[3][1]*rot[0][1],vector[3][0]*rot[1][0]+vector[3][1]*rot[1][1]]]
    print(rot_v)
    print(rect_coor)
    rect_coor = [[int(rot_v[0][0]+center[0]),int(rot_v[0][1]+center[1])],
                [int(rot_v[1][0]+center[0]),int(rot_v[1][1]+center[1])],
                [int(rot_v[2][0]+center[0]),int(rot_v[2][1]+center[1])],
                [int(rot_v[3][0]+center[0]),int(rot_v[3][1]+center[1])]]
    print(rect_coor)

def scaling():
    global rect_coor
    if rect_coor[0][0] == -1:
        print("Please draw a rectangle first")
        return
    sx = float(input("Enter sx: "))
    sy = float(input("Enter sy: "))
    (x1,y1),(x2,y2),(x3,y3),(x4,y4) = rect_coor
    center = [int((x1+x3)/2), int((y1+y3)/2)]
    rect_coor = [[int((x1-center[0])*sx+center[0]), int((y1-center[1])*sy+center[1])],
                 [int((x2-center[0])*sx+center[0]), int((y2-center[1])*sy+center[1])],
                 [int((x3-center[0])*sx+center[0]), int((y3-center[1])*sy+center[1])],
                 [int((x4-center[0])*sx+center[0]), int((y4-center[1])*sy+center[1])]]

def calc_rect():
    global rect_coor
    pts = rect_coor
    rect_coor= [pts[0],
                [pts[0][0], pts[2][1]],
                pts[2],
                [pts[2][0], pts[0][1]]]
    print(rect_coor)

def main():
    global name, bg
    init_window(name)
    cv.waitKey(1)
    while True:
        refresh_window()
        cv.waitKey(10)
        choice = menu()
        if choice == 1:
            white()
        elif choice == 2:
            rectangle()
        elif choice == 3:
            translation()
        elif choice == 4:
            rotation()
        elif choice == 5:
            scaling()
        elif choice == 6:
            cv.destroyAllWindows()
            break
        cv.waitKey(1)

name = "Lab1"
bg = np.zeros((600,800,3), np.uint8)
rect_coor = [[-1,-1],
             [-1,-1],
             [-1,-1],
             [-1,-1]]
draw = False
main()