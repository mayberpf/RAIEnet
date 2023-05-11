"""
#######################################
    @ Author : The DemonWolf
#######################################
"""

# Import the necessary libraries
import cv2 as cv
import numpy as np

# Define global variable to store unique edges
SimilarityMap = {}


class Graph:
    # init function to declare class variables
    def __init__(self, V):
        self.V = V
        self.adj = [[] for _ in range(V)]

    def DFSUtil(self, temp, v, visited):
        # Mark the current vertex as visited
        visited[v] = True
        # Store the vertex to list
        temp.append(v)

        # Repeat for all vertices adjacent to this vertex v
        for i in self.adj[v]:
            if not visited[i]:
                # Update the list
                temp = self.DFSUtil(temp, i, visited)
        return temp

    # Function to add an undirected edge
    def addEdge(self, v, w):
        self.adj[v].append(w)
        self.adj[w].append(v)

    # Function to retrieve connected components in an undirected graph
    def connectedComponents(self):
        visited, connectedComp = [], []
        for i in range(self.V):
            visited.append(False)
        for v in range(self.V):
            if not visited[v]:
                temp = []
                connectedComp.append(self.DFSUtil(temp, v, visited))
        return connectedComp


def updateSet(first, second):
    SimilarityMap.setdefault(first, set())
    SimilarityMap[first].add(second)
    SimilarityMap.setdefault(second, set())
    SimilarityMap[second].add(first)


def firstPhase(BinImage, ResultBinImage):
    Label = 0
    for row in range(len(BinImage)):
        for col in range(len(BinImage[0])):
            ResWest, ResNorth, ResNow = ResultBinImage[row][col-1], ResultBinImage[row-1][col], ResultBinImage[row][col]
            # Starting cell which is upper left corner cell
            if row == 0 and col == 0:
                Now = BinImage[row][col]
                if Now != 0:
                    Label += 1
                ResultBinImage[row][col] = Label

            # Entire first row except first cell
            elif row == 0 and col != 0:
                West, Now = BinImage[row][col-1], BinImage[row][col]
                if Now != West:
                    Label += 1
                ResultBinImage[row][col] = Label

            # Entire first column except first cell
            elif row != 0 and col == 0:
                North, Now = BinImage[row-1][col], BinImage[row][col]
                if Now != North:
                    Label += 1
                    ResultBinImage[row][col] = Label
                else:
                    ResultBinImage[row][col] = ResNorth

            # Except first row & column
            else:
                West, North, Now = BinImage[row][col-1], BinImage[row - 1][col], BinImage[row][col]
                if West == North and Now != West:
                    Label += 1
                    ResultBinImage[row][col] = Label
                    # Update SimilarityMap set
                    if Now == BinImage[row-1][col-1] and BinImage[row][col] != 0:
                        updateSet(ResultBinImage[row][col], ResultBinImage[row-1][col-1])

                if West == Now and North == Now:
                    ResultBinImage[row][col] = min(ResWest, ResNorth)
                    # Update SimilarityMap set
                    updateSet(ResNorth, ResWest)

                elif West == Now and North != Now:
                    ResultBinImage[row][col] = ResWest

                elif West != Now and North == Now:
                    ResultBinImage[row][col] = ResNorth

    return ResultBinImage, Label+1


def secondPhase(image, ResultBinImage):
    # Output after first pass
    ResultBinImage, maxLabel = firstPhase(image, ResultBinImage)
    # Create a graph to identify the connected components
    graph = Graph(maxLabel)
    for key, val in SimilarityMap.items():
        for ele in val:
            graph.addEdge(key, ele)
    # Remove similar labels
    components = graph.connectedComponents()
    for row in range(len(ResultBinImage)):
        for col in range(len(ResultBinImage[0])):
            if ResultBinImage[row][col] > 0:
                for index in range(len(components)):
                    if ResultBinImage[row][col] in components[index]:
                        ResultBinImage[row][col] = index

    return ResultBinImage


def ImageShowComponents(labels):
    # Map component labels to hue val
    label_hue = np.uint8(179*labels/np.max(labels))
    blank_ch = 255*np.ones_like(label_hue)
    labeled_img = cv.merge([label_hue, blank_ch, blank_ch])

    # cvt to BGR for display
    labeled_img = cv.cvtColor(labeled_img, cv.COLOR_HSV2BGR)

    # set bg label to black
    labeled_img[label_hue == 0] = 0

    cv.imwrite('labeled.png', labeled_img)
    # Print the final image and set wait until user exit the program (Close the image).
    cv.imshow('labeled.png', labeled_img)

    cv.waitKey(0)


if __name__ == '__main__':
    # Create a Resources folder and put the images to it.
    # Before start this program, make sure the image file named as fig1.png or
    # Put image name which is used to test the program, on the following list
    import pdb;pdb.set_trace()
    img = cv.imread("/home/ktd/cv_lane/res_mask.png", 0)
    # import pdb;pdb.set_trace()
    # Ensure binary
    img = cv.threshold(img, 127, 255, cv.THRESH_BINARY)[1]
    # Get row & col size of the img array
    size = list(img.shape)
    # Create a empty 2D array, which as similar size as img array with initial value = 0
    ResBinImage = np.array([[0] * size[1] for _ in range(size[0])])
    # Run the twoPass algorithms
    labeledImage = secondPhase(img, ResBinImage)
    # Using openCV, colored different labels with different colors
    ImageShowComponents(labeledImage)
