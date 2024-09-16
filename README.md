## This branch shows our work in task 13.1 which should work as follows:

- Develop a real-time Tic-Tac-Toe game using computer vision, powered by an Object Detection model (YOLO). The game should recognize hand gestures and convert them into moves on the grid.

## Repo's content

- A description for our code in this README file
- Python code used as the game engine
- **[Drive link](https://drive.google.com/drive/folders/1c8QZ-NIJ0zkRA0k8OvTMj14AMF7L-dT7)** for the dataset we made and used to train the model

----
----


# Summary Documentation of the Code

> This Python script serves as the game engine for a tic tac toe game that uses gesture detection to play. The engine is designed to detect X and O gestures in real-time using OpenCV and YOLOv8. While the engine itself is complete, the missing piece is training the YOLO model to detect the specific X and O gestures. Once the model is trained, it can be used to control the gameplay by detecting player moves. For trying the engine I have used a small toy refering to class 77 and it works perfectly well.

## Key Components in the code:

### YOLOv8 Object Detection:

- The current class i was using (cls == 77) need to be replaced with the correct class numbers after the model is trained on custom data for detecting X and O gestures.
Bounding boxes are used to identify the location of the gesture within the frame, and the center of the detected gesture is mapped to a position on the tic-tac-toe grid.

### Tic-Tac-Toe Game Logic:

>The game is played on a 3x3 grid (grid) where:
>1 represents X,
>0 represents O,
>-1 represents an empty cell.
>After each gesture is detected and mapped to the grid, the engine checks for a winner or a draw using the checkGameOver() function.

#### Turn Management:

>The game alternates between players:
>X’s turn is represented by turn = 1,
>O’s turn is represented by turn = 0.
>Each player's gesture is detected and only applied if it’s their turn.

#### Drawing the Board:

X symbols are drawn using DrawX().
O symbols are drawn using DrawO().
Gridlines are dynamically drawn to divide the frame into 3x3 sections representing the game board.

#### Declaring a Winner:

- When a player wins, the game displays a message (X Won, O Won, or Draw) on the screen. After a brief countdown the game resets for a new round.