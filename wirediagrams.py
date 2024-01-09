import numpy as np #import NumPy library

def generate_wirediagram(): #function to generate one wiring diagram, which is one 20x20 array full of numbers 1-4 to represent the colors

    # create a 20x20 array called 'wire_diagram' to represent the 20x20 pixel image, where each array element corresponds to a 'pixel'
    # each color Red, Blue, Yellow, Green will be represented by numbers 1, 2, 3, and 4 respectively

    # initialize our wire_diagram with 0's
    wire_diagram = np.zeros((20, 20), dtype=int)

    # create boolean variable 'rows_or_cols' which has a 50/50 chance of returning True or False, if True -> start gen process w/ picking a row, if False -> start gen process w/ picking a column
    rows_or_cols = np.random.choice([True, False])

    while np.any(wire_diagram == 0):  # while loop that will continue until all the array elements in 'wire_diagram' are "colored" with num 1-4, or not equal to 0 

        if rows_or_cols: # True --> so we start with rows

            row = np.random.randint(1, 21) # pick a random row number from 1-20 and set it in 'row'

            while np.any(wire_diagram[row - 1, :]): # while loop checks if the row has already been colored, if it has, we will generate a row number continously until it is unique
                row = np.random.randint(1, 21)

            row_color = np.random.randint(1, 5) # retrieve our random color 1-4 
            wire_diagram[row - 1, :] = row_color # populate that 'row' number in 'wire_diagram' with that color

            # moving on to columns, same general process

            col = np.random.choice([c for c in range(1, 21) if c != row]) # makes sure the newly generated column number is not the same as the previously generated row number
            while np.any(wire_diagram[:, col - 1]): # also checks that newly generated column number is not the same as previously generated column number
                col = np.random.choice([c for c in range(1, 21) if c != row])

            col_color = np.random.choice([c for c in range(1, 5) if c != row_color]) # retrieves a new, unique and random color 
            wire_diagram[:, col - 1] = col_color # populate that column with new color

        else: # False --> we start with cols

            #Same process as above, instead we first pick a column number 

            col = np.random.randint(1, 21)
            while np.any(wire_diagram[:, col - 1]):
                col = np.random.randint(1, 21)

            col_color = np.random.randint(1, 5)
            wire_diagram[:, col - 1] = col_color

            # switch to rows...

            row = np.random.choice([r for r in range(1, 21) if r != col])
            while np.any(wire_diagram[row - 1, :]):
                row = np.random.choice([r for r in range(1, 21) if r != col])

            row_color = np.random.choice([c for c in range(1, 5) if c != col_color])
            wire_diagram[row - 1, :] = row_color


    is_dangerous = (
    # checks if the wire at that position is red, and the color randomly picked happens to be yellow, this means that red would be laid before yellow, so it would be dangerous
    (wire_diagram[row - 1, col - 1] == 1 and row_color == 3)  or  (wire_diagram[row - 1, col - 1] == 1 and col_color == 3)  
    or
    (row_color == 3 and np.any(wire_diagram[row - 1, :] == 1))  # checks if a red wire is located in the same row (laid before) as the yellow wire
    or
    (col_color == 3 and np.any(wire_diagram[:, col - 1] == 1))  # check if the red wire is located in the same column (laid before) as the yellow wire
    )

    return wire_diagram, "Dangerous" if is_dangerous else "Safe", col_color if is_dangerous else None # return the generated 'wire_diagram' configuration, followed with a "Dangerous" or "Safe" label, followed by the column color to cut if the diagram is "Dangerous"

wire_diagram_dataset = [] # create 'wire_diagram_dataset' list to store each wire diagram, this will store all 10,000 wire diagrams.

num_wire_diagrams = 10000 # num of wire diagrams to generate

for i in range(num_wire_diagrams): # call the 'generate_wirediagram' function 10,000 times
    wire_diagram, label, cut_color = generate_wirediagram() # set our generated diagram to 'wire_diagram', its Danger/Safe label to 'label' and color of wire to cut to 'cut_color' retrieved from the 'generate_wirediagram' function
    wire_diagram_dataset.append((wire_diagram, label, cut_color)) # append these values to our dataset

for i, (wire_diagram, label, cut_color) in enumerate(wire_diagram_dataset): #prints out the entire wire diagram along with its associated Safe/Dangerous label
    print(f"\nwire_diagram {i + 1} - Label: {label}")
    print(wire_diagram)
