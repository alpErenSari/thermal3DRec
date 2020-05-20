import xml
import xml.etree.cElementTree as ET

class xmlOutput:
    def __init__(self):
        pass

    def print_out(self, filename, param):
        surfaceNumber = param['surfaceNumber']
        surfacePoints = param['surfacePoints']
        surfaceTypes = param['surfaceTypes']
        numberOfPoints = param['numberOfPoints']
        numberOfWindows = param['numberOfWindows']
        windowPoints = param['windowPoints']
        numberOfDoors = param['numberOfDoors']
        doorPoints = param['doorPoints']
        directions = param['directions']
        rValues = param['rValues']
        root = ET.Element("spaces")
        doc = ET.SubElement(root, "space", attrib={"id": "1", "numberOfSurfaces": "6"})

        for i in range(surfaceNumber):
            srfc = ET.SubElement(doc, "surface", attrib={"id": str(i+1), "type": surfaceTypes[i], "direction": directions[i],
                                                         "R": str(round(rValues[i], 2)), "numberOfPoints": str(numberOfPoints[i]), "numberOfWindows": str(numberOfWindows[i]),
                                                         "numberOfDoors": str(numberOfDoors[i])})
            surface_pts = surfacePoints[i]
            for j in range(4):
                ET.SubElement(srfc, "point", attrib={"x": str(surface_pts[j][0]), "y": str(surface_pts[j][1]),
                                                     "z": str(surface_pts[j][2])})
            surface_win_number = numberOfWindows[i]
            for j in range(surface_win_number):
                win_sub = ET.SubElement(srfc, "window", attrib={"id": str(j+1), "numberOfPoints": "4"})
                current_window = windowPoints[j]
                for k in range(4):
                    ET.SubElement(win_sub, "point", attrib={"x": str(current_window[0][k]), "y": str(current_window[1][k]),
                                                         "z": str(current_window[2][k])})

            surface_door_number = numberOfDoors[i]
            for j in range(surface_door_number):
                door_sub = ET.SubElement(srfc, "door", attrib={"id": str(j + 1), "numberOfDoors": "4"})
                current_door = doorPoints[j]
                for k in range(4):
                    ET.SubElement(door_sub, "point",
                                  attrib={"x": str(current_door[0][k]), "y": str(current_door[1][k]),
                                          "z": str(current_door[2][k])})

        tree = ET.ElementTree(root)
        tree.write(filename + ".xml", encoding='utf8')