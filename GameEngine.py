import cv2
import numpy as np
from ultralytics import YOLO
from time import sleep

model = YOLO("D:/courses/MIA_training/task13/group_repo/best.pt")# Replace this with the path of the downloaded model


X_OFFSET = 20
Y_OFFSET = 20

xFound = False
oFound = False
x_cords = (0,0)
o_cords = (0,0)

Xs = []
Os = []
turn = 1 # X turn in the beginning
winner = -1 # -1 refers to no winner

grid = [[-1, -1, -1],
        [-1, -1, -1],
        [-1, -1, -1]]
# A simulated grid to check the conditions of the game

winner_disp_counter = 0

def checkGameOver(): ## checks for game over using the simulated grid

    for row in grid:
        if row[0] != -1 and row[0] == row[1] == row[2]:
            return row[0]  
    for col in range(3):
        if grid[0][col] != -1 and grid[0][col] == grid[1][col] == grid[2][col]:
            return grid[0][col]
    
    if grid[0][0] != -1 and grid[0][0] == grid[1][1] == grid[2][2]:
        return grid[0][0]
    
    if grid[0][2] != -1 and grid[0][2] == grid[1][1] == grid[2][0]:
        return grid[0][2]

    if (all(cell != -1 for row in grid for cell in row) ):
        return 2

    
    return -1

def checkPos(cords):  # gets the position in the simulated grid with respect to its actual position in the frame
    for i in range (3):
        for j in range(3):
            if (cords[0] >= j*x_lines and cords[0] <= (j+1)*x_lines) and (cords[1] >= i*y_lines and cords[1] <= (i+1)*y_lines):
                return (i,j)
    return (-500,-500)


def DrawX(posX, posY):  # Draws an X in the given position
    cv2.line(frame, (x_lines*posY+X_OFFSET, y_lines*posX+Y_OFFSET), (x_lines*(posY+1)-X_OFFSET, y_lines*(posX+1)-Y_OFFSET ), (10,80,240), 7)
    cv2.line(frame, (x_lines*posY+X_OFFSET, y_lines*(posX+1)-Y_OFFSET), (x_lines*(posY+1)-X_OFFSET, y_lines*posX+Y_OFFSET ), (10,80,240), 7)
    grid[posX][posY] = 1

def DrawO(posY, posX): # draws O in the given position
    x_center = (posX*x_lines + (posX+1)*x_lines)//2
    y_center = (posY*y_lines + (posY+1)*y_lines)//2
    r = min([x_lines, y_lines])//2 - 10
    cv2.circle(frame, (x_center, y_center), r, (240,80,10), 7)
    grid[posY][posX] = 0



##////////////////////////////////////////////////////////////////////////
# Main loop

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error")
    exit()

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(1100,800)) ## setting a relatively high ressolution to overcome the inaccuaracy in our model

    if not ret:
        print("Error no frame")
        break
    X_FRAME, Y_FRAME = frame.shape[1], frame.shape[0]
    
    x_lines = X_FRAME // 3
    y_lines = Y_FRAME // 3

#############################################

    if winner == -1: 

        results = model.predict(frame, conf = 0.833,  vid_stride = 20, verbose=False, max_det = 10)

        for result in results:
                boxes = result.boxes

                for box in boxes:
                    conf = box.conf.cpu().numpy()
                    cls = box.cls.cpu().numpy().astype(int)
                    if  cls == 1 and turn == 1: # cls 1 refer to the x class 
                        label = model.names[cls[0]]
                        print(cls)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        
                        # cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)
                        xFound = True # declairing we detected am x gesture 
                        x_cords = (cx, cy) # setting the X coordinates
                        

                    elif cls == 0 and turn == 0 and conf > 0.877:  # cls 0 refer to the x class 
                        label = model.names[cls[0]]
                        print(cls)
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        
                        cx = (x1 + x2) // 2
                        cy = (y1 + y2) // 2
                        
                        cv2.circle(frame, (cx, cy), 15, (0, 0, 255), -1)
                        oFound = True # declairing we detected am O gesture 
                        o_cords = (cx, cy) # setting the O coordinates
                            


    ######################################################
        
        for x in range(x_lines, X_FRAME-3, x_lines ):
            cv2.line(frame, (x,0+Y_OFFSET), (x,Y_FRAME-Y_OFFSET), (100,150,190), 5)

        for y in range(y_lines, Y_FRAME-3, y_lines ):
            cv2.line(frame, (0+X_OFFSET, y), (X_FRAME-X_OFFSET, y), (100,150,190), 5)


        if xFound: # in case an X gesture was detected we check the cell if its empty and if it is
            (row, col) = checkPos(x_cords) # we draw an X in it and switch turns then we checks for game over 
            if row < 0: break 
            if grid[row][col] == -1:
                grid[row][col] = 1
                Xs.append((row, col))
                turn = 0
                winner = checkGameOver()
                sleep(0.005)

            xFound = False
        

        if oFound:  # in case an O gesture was detected
            (row, col) = checkPos(o_cords)
            if row < 0: break 
            if grid[row][col] == -1:
                grid[row][col] = 0
                Os.append((row, col))
                turn = 1
                winner = checkGameOver()
                sleep(0.005)


            oFound = False
        

        for x in Xs:
            DrawX(x[0], x[1])  # We draw all the X symbols in our grid 
        for o in Os:
            DrawO(o[0], o[1])  # We draw all the O symbols in our grid 


    else: # in case a winner was declaired 
        winner_disp_counter+=1
        if winner == 1: winner_text = f"X Won The Game!"
        if winner == 0: winner_text = f"O Won The Game!"
        if winner == 2: winner_text = f"It's a Draw"
        cv2.putText(frame, winner_text, (X_FRAME//10, Y_FRAME//2), cv2.FONT_HERSHEY_SIMPLEX, 2, (150, 170, 90), 5)
        cv2.putText(frame, f"The game will repeat in {(300 - winner_disp_counter)//40}", (X_FRAME//8, Y_FRAME-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 170, 90), 3)
        cv2.putText(frame, f"Press 'q' to quit", (X_FRAME//8, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 170, 90), 3)

        if winner_disp_counter >= 300:
            winner = -1
            grid = [[-1, -1, -1],
                    [-1, -1, -1],
                    [-1, -1, -1]]
            
            Xs = []
            Os = []
            turn = 1
            winner_disp_counter = 0
            xFound = False
            oFound = False

            
            



    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()